import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import models as models 
import object as obj
import adaptive_optics as ao
import helpers as helpers


device = "mps" 

# --- 1. Setup Simulation Parameters ---

# General Parameters
wavelength = 500e-9
z_dist = 0.2 #
grid_size = 2048
object_extent = 32e-3 #
object_fraction = 0.75 # the object takes up a smaller part of the entire object. 
grid_spacing = object_extent / grid_size
probe_radius = 15e-3 

total_incident_photons = 10000000 # Set the total number of photons in the entire beam

# Detector Parameters
detector_size = 32e-3 # 
detector_pixels = 1024

# Mirror Parameters
mirror_radius = probe_radius
actuator_max_stroke = 125e-9
actuator_jitter = 0.0 # careful with this, its in the normalized actuator commands [-1,1]
num_actuators = 48

# Optimization
num_measurements = 16
num_epochs = 60


# Setup coordinate grid
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

# make the object 
core_grid_size = int(grid_size * object_fraction)
gt_object = obj.make_text_mask("ALS", core_grid_size, margin=0.01, stroke_frac=0.01, font_name="Helvetica", device=device)
gt_object = obj.make_siemens_star_mask(core_grid_size, num_sectors=32, margin=0.01, line_fraction=1.0, center_radius_frac=0.0, device=device)



# Zero-pad the object up to the full simulation grid size
pad_pixels = (grid_size - core_grid_size) // 2
gt_object = torch.nn.functional.pad(gt_object, (pad_pixels, pad_pixels, pad_pixels, pad_pixels), mode='constant', value=0.0)


# The beam is now the same size as the sample (or at least covers it entirely)
probe_distance = torch.sqrt(X**2 + Y**2)
phi0 = 0.0
probe_amplitude_mask = (probe_distance <= probe_radius).to(torch.float32)

# Normalize the probe so the total sum of intensities (squared amplitude) equals our photon budget
probe_intensity_sum = torch.sum(probe_amplitude_mask**2)
probe_amplitude_mask = probe_amplitude_mask * torch.sqrt(total_incident_photons / probe_intensity_sum)

# Set up the deformable mirror
mirror = ao.deformableMirror(
    wavelength=wavelength,
    number_actuators=num_actuators,
    max_stroke=actuator_max_stroke,
    mirror_radius=mirror_radius,
    grid_size_wavefront=grid_size,
    grid_spacing_wavefront=grid_spacing,
    device=device
)

# Generate synthetic measurements by applying random commands to the DM instead of shifting the sample
print(f"Generating synthetic data for {num_measurements} random DM configurations...")

# We use the true physics to generate our "experimental" data
gt_model = models.DiptychModel(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device,
    detector_size=detector_size,
    detector_pixels=detector_pixels
).to(device)

gt_model.set_object(gt_object)

# Collect measurements
raw_intensities = []
dm_commands = []

with torch.no_grad():
    for i in range(num_measurements):

        # Generate random actuator commands [-1, 1]
        commands = (torch.rand(num_actuators, device=device) * 2) - 1
        dm_commands.append(commands)
        dm_commands_error = helpers.add_actuator_jitter(commands, actuator_jitter)
        
        # Keep illumination in arbitrary units (amplitude of 1 inside the probe)
        illumination = probe_amplitude_mask.to(torch.complex64)
        
        # No shift (0,0) - diptych relies entirely on the probe phase variations from DM
        detector_intensity = gt_model(illumination, dm_commands_error, mirror)
        raw_intensities.append(detector_intensity)

raw_intensities = torch.stack(raw_intensities)

measurements = []
with torch.no_grad():
    for raw_int in raw_intensities:
        # Add Poisson shot noise naturally on the photon counts
        noisy_photons = helpers.add_shot_noise(raw_int)
        measurements.append(noisy_photons)
        
measurements = torch.stack(measurements)
dm_commands = torch.stack(dm_commands)

global_max_intensity = raw_intensities.max()
print(f"Total incident photons in beam: {total_incident_photons:.4e}")
print(f"Max detector pixel intensity (photons): {global_max_intensity:.4e}")
print("Measurements generated. Starting optimization...")

# --- 2. Initialize the Reconstruction Model ---
model = models.DiptychModel(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device,
    detector_size=detector_size,
    detector_pixels=detector_pixels
).to(device)

model.object_amplitude.data = torch.ones((grid_size, grid_size), dtype=torch.float32, device=device)
model.object_phase.data = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)

# --- 3. Setup Optimizer ---
optimizer = optim.AdamW(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

loss_history = []

# --- 4. Optimization Loop ---
for epoch in range(num_epochs):
    epoch_loss = 0
    
    # Process each DM configuration
    for idx in range(num_measurements):
        optimizer.zero_grad()
        
        # Base illumination in arbitrary units
        illumination = probe_amplitude_mask.to(torch.complex64)
        
        # Forward pass using the DM commands (the model natively handles the mirror and propagation)
        predicted_intensity = model(illumination, dm_commands[idx], mirror)
        
        # Calculate loss against the measurement (using amplitude instead of raw photon intensity)
        # This stabilizes optimization when dealing with very large integer photon counts.
        loss = criterion(torch.sqrt(predicted_intensity + 1e-12), torch.sqrt(measurements[idx] + 1e-12))
        epoch_loss += loss.item()
        
        # Backward pass and optimize per position
        loss.backward()
        optimizer.step()
        
        # Apply constraints to the amplitude
        with torch.no_grad():
            model.object_amplitude.data = torch.clamp(model.object_amplitude.data, 0, 1)
            
    loss_val = epoch_loss / num_measurements
    loss_history.append(loss_val)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.4e}")

print("Optimization complete.")

# --- 5. Visualization ---
plt.figure(figsize=(15, 10))

# Plot 1: Loss History
plt.subplot(2, 2, 1)
plt.plot(loss_history)
plt.title('Optimization Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')

# Ground truth extent for plotting
plot_extent = (-object_extent/2, object_extent/2, -object_extent/2, object_extent/2)

# Plot 2: Ground Truth Amplitude
plt.subplot(2, 2, 2)
plt.imshow(torch.abs(gt_object).cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Object plane transmission')
plt.xlabel('(m)')
plt.ylabel('(m)')
plt.colorbar()

# Plot 3: Reconstructed Amplitude
plt.subplot(2, 2, 3)
plt.imshow(model.object_amplitude.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed transmission')
plt.xlabel('(m)')
plt.ylabel('(m)')
plt.colorbar()

# Plot 4: Reconstructed Phase
plt.subplot(2, 2, 4)
plt.imshow(model.object_phase.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed Phase')
plt.xlabel('(m)')
plt.ylabel('(m)')
plt.colorbar()

plt.tight_layout()
plt.savefig('diptych_reconstruction.png', dpi=500)
print("Saved reconstruction results to 'diptych_reconstruction.png'")


# --- 6. Plot first 5 measurements and their probes ---
print("Generating tableau of first 5 measurements...")

# Get the extent of the detector for plotting
detector_extent = (-detector_size/2, detector_size/2, -detector_size/2, detector_size/2)

fig, axes = plt.subplots(2, 5, figsize=(20, 12))

with torch.no_grad():
    for i in range(5):
        # 1. Recreate the probe
        mirror_phase = mirror.forward(dm_commands[i])
        probe = probe_amplitude_mask.to(torch.complex64)
        
        # When amplitude is exactly 0, phase is mathematically not very well defined but torch.angle
        # might return numerical noise or 0. We force it to 0 for a cleaner plot.
        probe_phase_np = torch.angle(probe).cpu().numpy()
        probe_phase_np[probe_amplitude_mask.cpu().numpy() == 0] = 0.0
        
        diffraction_intensity = measurements[i].cpu().numpy()
        
        # Plot Probe Phase
        ax = axes[0, i]
        im1 = ax.imshow(probe_phase_np, extent=plot_extent, origin='lower', cmap='twilight')
        ax.set_title(f'Probe Phase {i+1}')
        if i == 0: ax.set_ylabel('y (m)')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        

        # Plot Detector Diffraction Pattern (Intensity)
        ax = axes[1, i]
        im3 = ax.imshow(np.log10(diffraction_intensity + 1e-12), extent=detector_extent, origin='lower', cmap='inferno')
        ax.set_title(f'Diffraction {i+1} (log10)')
        if i == 0: ax.set_ylabel('y (m)')
        ax.set_xlabel('x (m)')
        # im3.set_clim(0, diffraction_intensity.max()/2)
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('diptych_measurements_tableau.png')
print("Saved measurement tableau to 'diptych_measurements_tableau.png'")
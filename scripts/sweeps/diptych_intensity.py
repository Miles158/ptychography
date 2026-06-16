import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Ensure imports work regardless of where the script is run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import models as models 
from src import object as obj
from src import adaptive_optics as ao
from src import helpers as helpers


device = "mps" 



# --- 1. Setup Simulation Parameters ---

# General Parameters
wavelength = 500e-9
z_dist = 250e-3 #
grid_size = 1024
object_extent = 32e-3 #
object_fraction = 0.75 # the object takes up a smaller part of the entire object. 
grid_spacing = object_extent / grid_size
probe_radius = 15e-3 


# Detector Parameters
detector_size = 1.5e-3 # 
detector_pixels = 1024

# Mirror Parameters
mirror_radius = probe_radius
actuator_max_stroke = 500e-9
actuator_jitter = 0.01 # careful with this, its in the normalized actuator commands [-1,1]
num_actuators = 24

# Optimization
num_measurements = 25


source_intensities = [1e1, 1e2, 1e3, 1e4, 1e5] # Sweep through peak photon counts
num_epochs_list = [150, 150, 150, 150, 150]   # Number of epochs per intensity run
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

# Prepare lists to store sweep results
all_loss_histories = []
reconstruction_mse_list = []
all_recon_amplitudes = [] # To store the final image from each run

# Crop coordinates to calculate MSE only on the core object area, not the massive padded region
crop_start = pad_pixels
crop_end = pad_pixels + core_grid_size
gt_amp_core = torch.abs(gt_object)[crop_start:crop_end, crop_start:crop_end]

for sweep_idx, (source_intensity, num_epochs) in enumerate(zip(source_intensities, num_epochs_list)):
    print(f"\n=========================================================")
    print(f"Starting Sweep {sweep_idx+1}/{len(source_intensities)}: Intensity = {source_intensity:.1e} photons")
    print(f"=========================================================")
    
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
            
    # Find global max across all measurements
    raw_intensities = torch.stack(raw_intensities)
    global_max_intensity = raw_intensities.max()

    # Scale all intensities such that the maximum pixel across all measurements matches source_intensity
    scale_factor = source_intensity / global_max_intensity

    measurements = []
    with torch.no_grad():
        for raw_int in raw_intensities:
            scaled_intensity = raw_int * scale_factor
            # Add Poisson shot noise naturally on the photon counts
            noisy_photons = helpers.add_shot_noise(scaled_intensity)
            measurements.append(noisy_photons)
            
    measurements = torch.stack(measurements)
    dm_commands = torch.stack(dm_commands)

    print(f"Global max before scaling: {global_max_intensity:.4e}")
    print(f"Scaling factor (alpha): {scale_factor:.4e}")
    print("Measurements generated. Starting optimization...")

    # --- 2. Initialize the Reconstruction Model (Fresh for each sweep) ---
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
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
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
            predicted_intensity_raw = model(illumination, dm_commands[idx], mirror)
            
            # Scale the predicted intensity by the exact same alpha factor so it matches the photon count domain
            predicted_intensity = predicted_intensity_raw * scale_factor
            
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
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1:03d}/{num_epochs}, Loss: {loss_val:.4e}")

    print("Optimization complete.")
    all_loss_histories.append(loss_history)
    
    # Store the final reconstructed amplitude for the grid plot
    all_recon_amplitudes.append(model.object_amplitude.detach().cpu().numpy())
    
    # Calculate MSE strictly on the core object area
    recon_amp_core = model.object_amplitude.detach()[crop_start:crop_end, crop_start:crop_end]
    mse_core = torch.mean((gt_amp_core - recon_amp_core)**2).item()
    reconstruction_mse_list.append(mse_core)
    print(f"Final Core Object MSE: {mse_core:.4e}")

# --- 5. Visualization ---
plt.figure(figsize=(12, 5))

# Plot 1: Optimization Loss vs Epochs (All runs)
plt.subplot(1, 2, 1)
for i, (intensity, history) in enumerate(zip(source_intensities, all_loss_histories)):
    plt.plot(history, label=f'Intensity: {intensity:.1e}')
plt.title('Optimization Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (Amplitude)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Reconstruction MSE vs Intensity
plt.subplot(1, 2, 2)
plt.plot(source_intensities, reconstruction_mse_list, marker='o', linewidth=2, markersize=8)
plt.title('Reconstruction Quality vs Peak Intensity')
plt.xlabel('Peak Measurment Intensity')
plt.ylabel('Core Object MSE')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/diptych_intensity_sweep.png', dpi=400)
print("\nSaved sweep results to 'plots/diptych_intensity_sweep.png'")


# --- 6. Tableau of Reconstructed Images ---
n_sweeps = len(source_intensities)
# Calculate a nice grid layout (e.g. 2x3 or 3x3 depending on n_sweeps)
cols = min(n_sweeps, 4)
rows = int(np.ceil(n_sweeps / cols))

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
axes = axes.flatten()

# Ground truth extent for plotting
plot_extent = (-object_extent/2, object_extent/2, -object_extent/2, object_extent/2)

for i in range(n_sweeps):
    ax = axes[i]
    im = ax.imshow(all_recon_amplitudes[i], extent=plot_extent, origin='lower', cmap='inferno')
    ax.set_title(f"Intensity: {source_intensities[i]:.1e}\nMSE: {reconstruction_mse_list[i]:.4e}")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Hide any empty subplots
for i in range(n_sweeps, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('plots/diptych_intensity_reconstructions.png', dpi=400)
print("Saved reconstruction tableau to 'plots/diptych_intensity_reconstructions.png'")


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import models as models 
import object as obj
import helpers

# Set device to MPS if available, otherwise CPU
device = "mps"

# --- 1. Setup Simulation Parameters ---
# General Parameters
wavelength = 500e-9
z_dist = 250e-3 #
grid_size = 1024
object_extent = 5e-3 #
object_fraction = 0.5 # the object takes up a smaller part of the entire object. 
grid_spacing = object_extent / grid_size

probe_radius = 0.25e-3 
desired_overlap = 0.72

source_intensity = 10000 
num_epochs = 35

# Detector Parameters
detector_size = 1.5e-3 # 
detector_pixels = 1024

# Setup coordinate grid
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

# make the object 
core_grid_size = int(grid_size * object_fraction)
gt_object = obj.make_siemens_star_mask(core_grid_size, num_sectors=32, margin=0.01, line_fraction=1.0, center_radius_frac=0.0, device=device)




# Zero-pad the object up to the full simulation grid size
pad_pixels = (grid_size - core_grid_size) // 2
gt_object = torch.nn.functional.pad(gt_object, (pad_pixels, pad_pixels, pad_pixels, pad_pixels), mode='constant', value=0.0)



# Generate synthetic measurements by scanning the illumination over the object
# probe = torch.exp(-(X**2 + Y**2) / (2 * probe_radius**2)).to(torch.complex64).to(device)
# probe_phase = torch.exp(1j * 1e6 * (X**2 + Y**2))
# probe = (probe_amplitude * probe_phase).to(torch.complex64).to(device)



r = torch.sqrt(X**2 + Y**2)  # uses your existing X, Y in meters

phi0 = 0.0
mask = (r <= probe_radius).to(torch.float32)  # top-hat aperture

# Scale illumination so that intensity natively matches the photon count world
probe = mask.to(torch.complex64) * torch.sqrt(torch.tensor(source_intensity, device=device)) * torch.exp(torch.tensor(1j * phi0, device=device))


probe_diameter = 2 * probe_radius
shift_step_meters = probe_diameter * (1 - desired_overlap)
shift_step = max(1, int(shift_step_meters / grid_spacing)) # step size in pixels

# Calculate number of scan positions to cover the unpadded core region
core_extent = core_grid_size * grid_spacing
num_positions = int(core_extent / shift_step_meters) + 1

shifts = []
for i in range(-num_positions//2 + 1, num_positions//2 + 1):
    for j in range(-num_positions//2 + 1, num_positions//2 + 1):
        shifts.append((i * shift_step, j * shift_step))

print(f"Generating synthetic data for {len(shifts)} scan positions with overlap {desired_overlap*100}%...")

# We use the true physics to generate our "experimental" data
gt_model = models.PtychoModel(
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
measurements = []
with torch.no_grad():
    for shift_x, shift_y in shifts:
        intensity = gt_model(probe, pixel_shift_x=shift_x, pixel_shift_y=shift_y)
        noisy_intensity = helpers.add_shot_noise(intensity)
        measurements.append(noisy_intensity)
measurements = torch.stack(measurements)

print("Measurements generated. Starting optimization...")

# --- 2. Initialize the Reconstruction Model ---
recon_model = models.PtychoModel(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device,
    detector_size=detector_size,
    detector_pixels=detector_pixels
).to(device)


recon_model.object_amplitude.data = torch.ones((grid_size, grid_size), dtype=torch.float32, device=device)
recon_model.object_phase.data = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)

# --- 3. Setup Optimizer ---
optimizer = optim.AdamW(recon_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

loss_history = []

# --- 4. Optimization Loop ---
for epoch in range(num_epochs):
    
    epoch_loss = 0
    # Process each scan position
    for idx, (shift_x, shift_y) in enumerate(shifts):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_intensity = recon_model(probe, pixel_shift_x=shift_x, pixel_shift_y=shift_y)
        
        # Calculate loss against the measurement (using amplitude and stabilization)
        loss = criterion(torch.sqrt(predicted_intensity + 1e-12), torch.sqrt(measurements[idx] + 1e-12))
        epoch_loss += loss.item()
        
        # Backward pass and optimize per position
        loss.backward()
        optimizer.step()
        
        # Optional: Apply constraints to the amplitude (e.g. amplitude between 0 and 1)
        with torch.no_grad():
            recon_model.object_amplitude.data = torch.clamp(recon_model.object_amplitude.data, 0, 1)
            
    loss_val = epoch_loss / len(shifts)
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
plt.title('Ground Truth Amplitude')
plt.colorbar()

# Plot 3: Reconstructed Amplitude
plt.subplot(2, 2, 3)
plt.imshow(recon_model.object_amplitude.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed Amplitude')
plt.colorbar()

# Plot 4: Reconstructed Phase
plt.subplot(2, 2, 4)
plt.imshow(recon_model.object_phase.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed Phase')
plt.colorbar()

plt.tight_layout()
plt.savefig('ptychography_reconstruction.png')
print("Saved reconstruction results to 'ptychography_reconstruction.png'")
# plt.show()
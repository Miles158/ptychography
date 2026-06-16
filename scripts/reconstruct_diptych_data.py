import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Ensure imports work regardless of where the script is run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import models as models 
from src import adaptive_optics as ao

device = "mps" 

# --- 1. Load the generated dataset ---
data_path = os.path.join(os.path.dirname(__file__), '../data/synthetic/diptych_dataset_001.pt')
print(f"Loading dataset from {data_path}...")
dataset = torch.load(data_path, weights_only=False)

# Extract measurements and configuration parameters
measurements = dataset["measurements"].to(device)
dm_commands = dataset["dm_commands"].to(device)
gt_object = dataset["gt_object"].to(device)
probe_amplitude_mask = dataset["probe_amplitude_mask"].to(device)
scale_factor = dataset["scale_factor"]
if isinstance(scale_factor, torch.Tensor):
    scale_factor = scale_factor.to(device)

source_intensity = dataset["source_intensity"]
wavelength = dataset["wavelength"]
z_dist = dataset["z_dist"]
grid_size = dataset["grid_size"]
object_extent = dataset["object_extent"]
grid_spacing = dataset["grid_spacing"]
detector_size = dataset["detector_size"]
detector_pixels = dataset["detector_pixels"]

mirror_radius = dataset["mirror_radius"]
actuator_max_stroke = dataset["actuator_max_stroke"]
num_actuators = dataset["num_actuators"]

num_measurements = measurements.shape[0]
num_epochs = 100

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

print("Dataset loaded. Starting optimization...")

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
plt.title('Object plane transmission (Ground Truth)')
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
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/diptych_reconstruction.png', dpi=500)
print("Saved reconstruction results to 'plots/diptych_reconstruction.png'")

# --- 6. Plot first 5 measurements and their probes ---
print("Generating tableau of first 5 measurements...")

# Get the extent of the detector for plotting
detector_extent = (-detector_size/2, detector_size/2, -detector_size/2, detector_size/2)

fig, axes = plt.subplots(2, 5, figsize=(20, 12))

with torch.no_grad():
    for i in range(min(5, num_measurements)):
        # 1. Recreate the probe
        mirror_phase = mirror.forward(dm_commands[i])
        probe = (probe_amplitude_mask.to(torch.complex64) * mirror_phase) 
        
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
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('plots/diptych_measurements_tableau.png')
print("Saved measurement tableau to 'plots/diptych_measurements_tableau.png'")
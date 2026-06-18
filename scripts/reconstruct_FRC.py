import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

import models as models 
import adaptive_optics as ao
import helpers

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def reconstruct_dataset(data_path, num_epochs=100):
    print(f"Loading dataset from {data_path}...")
    dataset = torch.load(data_path, weights_only=False)

    measurements = dataset["measurements"].to(device)
    dm_commands = dataset["dm_commands"].to(device)
    gt_object = dataset["gt_object"].to(device)
    probe_amplitude_mask = dataset["probe_amplitude_mask"].to(device)

    wavelength = dataset["wavelength_m"]
    z_dist = dataset["z_dist_m"]
    grid_size = dataset["grid_size"]
    object_extent = dataset["object_extent_m"]
    grid_spacing = dataset["grid_spacing_m"]
    detector_size = dataset["detector_size_m"]
    detector_pixels = dataset["detector_pixels"]

    mirror_radius = dataset["mirror_radius_m"]
    actuator_max_stroke = dataset["actuator_max_stroke_m"]
    num_actuators = dataset["num_actuators"]

    num_measurements = measurements.shape[0]

    mirror = ao.deformableMirror(
        wavelength=wavelength,
        number_actuators=num_actuators,
        max_stroke=actuator_max_stroke,
        mirror_radius=mirror_radius,
        grid_size_wavefront=grid_size,
        grid_spacing_wavefront=grid_spacing,
        device=device
    )

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

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    loss_history = []

    print("Starting optimization...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for idx in range(num_measurements):
            optimizer.zero_grad()
            illumination = probe_amplitude_mask.to(torch.complex64)
            predicted_intensity = model(illumination, dm_commands[idx], mirror)
            
            loss = criterion(torch.sqrt(predicted_intensity + 1e-12), torch.sqrt(measurements[idx] + 1e-12))
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.object_amplitude.data = torch.clamp(model.object_amplitude.data, 0, 1)
                
        loss_val = epoch_loss / num_measurements
        loss_history.append(loss_val)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.4e}")

    complex_reconstruction = model.object_amplitude * torch.exp(1j * model.object_phase)
    return complex_reconstruction.detach(), gt_object, grid_spacing, object_extent, loss_history

# --- 1. Define paths to the two datasets ---
# Update these paths to point to your actual data sets
data_path_A = os.path.join(os.path.dirname(__file__), '../data/synthetic/diptych_dataset_001.pt')
data_path_B = os.path.join(os.path.dirname(__file__), '../data/synthetic/diptych_dataset_002.pt')

num_epochs = 100

# --- 2. Reconstruct both datasets ---
print("--- Reconstructing Dataset A ---")
recon_A, gt_A, grid_spacing, extent, loss_A = reconstruct_dataset(data_path_A, num_epochs=num_epochs)

print("\n--- Reconstructing Dataset B ---")
recon_B, gt_B, _, _, loss_B = reconstruct_dataset(data_path_B, num_epochs=num_epochs)

# --- 3. Compute FRC ---
print("\nComputing FRC...")
num_points = 50
freq_centers, frc_values = helpers.FRC(recon_A, recon_B, grid_spacing, num_points)

# Move to CPU for plotting
freq_centers = freq_centers.cpu().numpy()
frc_values = frc_values.cpu().numpy()

recon_A_amp = torch.abs(recon_A).cpu().numpy()
recon_A_phase = torch.angle(recon_A).cpu().numpy()
recon_B_amp = torch.abs(recon_B).cpu().numpy()
recon_B_phase = torch.angle(recon_B).cpu().numpy()

plot_extent = (-extent/2, extent/2, -extent/2, extent/2)

# --- 4. Plotting ---
plt.figure(figsize=(20, 10))

# Reconstructed Amplitude A
plt.subplot(2, 3, 1)
plt.imshow(recon_A_amp, extent=plot_extent, origin='lower')
plt.title('Reconstruction A (Amplitude)')
plt.xlabel('(m)')
plt.ylabel('(m)')
plt.colorbar()

# Reconstructed Phase A
plt.subplot(2, 3, 2)
plt.imshow(recon_A_phase, extent=plot_extent, origin='lower')
plt.title('Reconstruction A (Phase)')
plt.xlabel('(m)')
plt.ylabel('(m)')
plt.colorbar()

# Reconstructed Amplitude B
plt.subplot(2, 3, 4)
plt.imshow(recon_B_amp, extent=plot_extent, origin='lower')
plt.title('Reconstruction B (Amplitude)')
plt.xlabel('(m)')
plt.ylabel('(m)')
plt.colorbar()

# Reconstructed Phase B
plt.subplot(2, 3, 5)
plt.imshow(recon_B_phase, extent=plot_extent, origin='lower')
plt.title('Reconstruction B (Phase)')
plt.xlabel('(m)')
plt.ylabel('(m)')
plt.colorbar()

# FRC Plot
plt.subplot(2, 3, (3, 6))
plt.plot(freq_centers, frc_values, '-o', label='FRC')
# Add 1/bit threshold or standard thresholds if needed
plt.axhline(0.5, color='gray', linestyle='--', label='0.5 Threshold')
plt.axhline(0.143, color='red', linestyle='--', label='1/7 Threshold')
plt.title('Fourier Ring Correlation (FRC)')
plt.xlabel('Spatial Frequency (1/m)')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
out_path = 'plots/reconstruction_and_frc.png'
plt.savefig(out_path, dpi=300)
print(f"Saved reconstructions and FRC plot to '{out_path}'")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
import models as models

# simulation parameters

device = "mps"

wavelength = 632e-9
z_dist_start = 6e-3
grid_size = 2048 * 2
object_extent = 14e-3
grid_spacing = object_extent / grid_size

probe_radius = 0.1e-3 
source_intensity = 10000

detector_shape_m = (2.38e-3, 2.38e-3)
detector_shape_px = (1080, 1080)





# --- 1. Load Experimental Data ---
data_path = os.path.join(os.path.dirname(__file__), '../data/experimental/stxm_06232036_002.pkl')
print(f"Loading experimental data from {data_path}...")
with open(data_path, 'rb') as f:
    # Load data as numpy arrays first to avoid immediate memory explosion
    data = pickle.load(f)

# Subsample immediately in numpy to keep memory footprint low
subsample_rate = 1
diffraction_patterns_np = np.array(data['diffraction_patterns'])[::subsample_rate, :, :]
x_positions_np = np.array(data['x_positions'])[::subsample_rate]
y_positions_np = np.array(data['y_positions'])[::subsample_rate]
loaded_probe_np = np.array(data['beam_profile_data']['averaged_profile'])
# Convert only the subsampled data to torch tensors
diffraction_patterns = torch.from_numpy(diffraction_patterns_np.astype(np.float32)).to(device)
x_positions_m = torch.from_numpy(x_positions_np.astype(np.float32)).to(device) / 34555 / 1000
y_positions_m = torch.from_numpy(y_positions_np.astype(np.float32)).to(device) / 34555 / 1000
loaded_probe = torch.from_numpy(loaded_probe_np.astype(np.float32)).to(device) 


# Load the probe from the data
# The probe is on the detector plane, we need to map it to the object plane.
# Assuming the planes are the same size/spacing for the starting point.
raw_probe = torch.from_numpy(loaded_probe_np.astype(np.complex64)).to(device)

# Interpolate intensity (abs^2) to avoid complex interpolation issues
probe_intensity = torch.abs(raw_probe).unsqueeze(0).unsqueeze(0)
probe_phase = torch.angle(raw_probe).unsqueeze(0).unsqueeze(0)

# Resize intensity and phase separately
initial_probe_amp = torch.nn.functional.interpolate(
    probe_intensity, 
    size=(grid_size, grid_size), 
    mode='bilinear', 
    align_corners=False
).squeeze()

initial_probe_phase = torch.nn.functional.interpolate(
    probe_phase, 
    size=(grid_size, grid_size), 
    mode='bilinear', 
    align_corners=False
).squeeze()

initial_probe = initial_probe_amp * torch.exp(1j * initial_probe_phase)

spacing = data['metadata']['spacing']
# Clean up large numpy arrays
del data, diffraction_patterns_np, x_positions_np, y_positions_np
import gc
gc.collect()






model = models.PtychoModel(
    object_grid_shape=(grid_size, grid_size), 
    grid_spacing_object=(grid_spacing, grid_spacing), 
    wavelength=wavelength, 
    z_dist_init=0.2,
    device=device,
    detector_shape_m=detector_shape_m,
    detector_shape_px=detector_shape_px,
    probe_init=initial_probe
).to(device)




optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
loss_history = []

# --- 3. Optimization Loop ---
num_epochs = 5
print("Starting ptychographic reconstruction...")

for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(len(diffraction_patterns)):
        optimizer.zero_grad()
        
        # Convert meter positions to pixel shifts
        shift_x = int(x_positions_m[i] / spacing)
        shift_y = int(y_positions_m[i] / spacing)
        
        # Forward pass
        pred = model(pixel_shift_x=shift_x, pixel_shift_y=shift_y)
        
        # Ensure pred is float32
        pred = pred.to(torch.float32)
        
        loss = criterion(torch.sqrt(pred + 1e-12), torch.sqrt(diffraction_patterns[i] + 1e-12))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(diffraction_patterns):.4e}")
    loss_history.append(epoch_loss/len(diffraction_patterns))

print(f"Optimization complete. Optimized z_dist: {model.z_dist.item():.4e} m")

# --- 4. Visualization ---
# Visualization of Diffraction Patterns
os.makedirs('plots', exist_ok=True)
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i in range(min(16, len(diffraction_patterns))):
    ax = axes.flat[i]
    # Take log for better visualization
    pattern = torch.log10(diffraction_patterns[i] + 1e-6).cpu().numpy()
    ax.imshow(pattern, cmap='inferno')
    ax.set_title(f"x: {x_positions_m[i]*1000:.2f}mm, y: {y_positions_m[i]*1000:.2f}mm", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig('plots/diffraction_patterns.png')
print("Saved diffraction patterns to plots/diffraction_patterns.png")
plt.close(fig)

# --- 5. Comparison Visualization ---
model.eval()
with torch.no_grad():
    indices = [0, len(diffraction_patterns)//2, len(diffraction_patterns)-1]
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    
    # Row 1: Optimized Object/Probe
    axes[0, 0].imshow(model.object_amplitude.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Obj Amp')
    axes[0, 1].imshow(model.object_phase.detach().cpu().numpy(), cmap='twilight')
    axes[0, 1].set_title('Obj Phase')
    axes[0, 2].imshow(model.probe_amplitude.detach().cpu().numpy(), cmap='inferno')
    axes[0, 2].set_title('Probe Amp')
    axes[0, 3].imshow(model.probe_phase.detach().cpu().numpy(), cmap='twilight')
    axes[0, 3].set_title('Probe Phase')
    
    # Rows 2-4: Comparison
    for row, idx in enumerate(indices, 1):
        shift_x = int(x_positions_m[idx] / spacing)
        shift_y = int(y_positions_m[idx] / spacing)
        pred = model(pixel_shift_x=shift_x, pixel_shift_y=shift_y).detach().cpu().numpy()
        meas = diffraction_patterns[idx].detach().cpu().numpy()
        
        axes[row, 0].imshow(np.log10(meas + 1e-6), cmap='inferno')
        axes[row, 0].set_title(f'Meas {idx}')
        axes[row, 1].imshow(np.log10(pred + 1e-6), cmap='inferno')
        axes[row, 1].set_title(f'Pred {idx}')
        axes[row, 2].imshow(meas - pred, cmap='bwr')
        axes[row, 2].set_title('Diff')
        axes[row, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig('plots/reconstruction_comparison.png')
    print("Saved reconstruction comparison to plots/reconstruction_comparison.png")
    plt.close(fig)

plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.plot(loss_history)
plt.title('Optimization Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')

plt.subplot(1, 4, 2)
plt.imshow(model.object_amplitude.detach().cpu().numpy(), cmap='gray')
plt.title('Reconstructed Amplitude')

plt.subplot(1, 4, 3)
plt.imshow(model.object_phase.detach().cpu().numpy(), cmap='twilight')
plt.title('Reconstructed Phase')

plt.subplot(1, 4, 4)
plt.imshow(model.probe_amplitude.detach().cpu().numpy(), cmap='inferno')
plt.title('Optimized Probe Amplitude')

plt.tight_layout()
plt.savefig('plots/experimental_reconstruction.png')
print("Saved reconstruction to plots/experimental_reconstruction.png")

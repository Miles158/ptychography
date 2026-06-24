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
grid_size = 2048
object_extent = 8e-3
grid_spacing = object_extent / grid_size

probe_radius = 1e-3 
source_intensity = 10000

detector_shape_m = (2.38e-3, 4.22e-3)
detector_shape_px = (1080, 1920)

# Create a starter probe
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')
r = torch.sqrt(X**2 + Y**2)  # uses your existing X, Y in meters
phi0 = 0.0
mask = (r <= probe_radius).to(torch.float32)  # top-hat aperture

probe = mask.to(torch.complex64) * torch.sqrt(torch.tensor(source_intensity, device=device)) * torch.exp(torch.tensor(1j * phi0, device=device))



model = models.PtychoModel(
    object_grid_shape=(grid_size, grid_size), 
    grid_spacing_object=(grid_spacing, grid_spacing), 
    wavelength=wavelength, 
    z_dist_init=0.2,
    device=device,
    detector_shape_m=detector_shape_m,
    detector_shape_px=detector_shape_px,
    probe_init=probe
).to(device)



# Ensure imports work


# --- 1. Load Experimental Data ---
data_path = os.path.join(os.path.dirname(__file__), '../data/experimental/stxm_06232036_001.pkl')
print(f"Loading experimental data from {data_path}...")
with open(data_path, 'rb') as f:
    # Load data as numpy arrays first to avoid immediate memory explosion
    data = pickle.load(f)

# Subsample immediately in numpy to keep memory footprint low
subsample_rate = 6
diffraction_patterns_np = np.array(data['diffraction_patterns'])[::subsample_rate, :, :, 0]
x_positions_np = np.array(data['x_positions'])[::subsample_rate]
y_positions_np = np.array(data['y_positions'])[::subsample_rate]

# Convert only the subsampled data to torch tensors
diffraction_patterns = torch.from_numpy(diffraction_patterns_np.astype(np.float32)).to(device)
x_positions_m = torch.from_numpy(x_positions_np.astype(np.float32)).to(device) / 34555 / 1000
y_positions_m = torch.from_numpy(y_positions_np.astype(np.float32)).to(device) / 34555 / 1000


metadata = data['metadata']
spacing = metadata['spacing']
# Clean up large numpy arrays
del data, diffraction_patterns_np, x_positions_np, y_positions_np
import gc
gc.collect()



optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
loss_history = []

# --- 3. Optimization Loop ---
num_epochs = 12
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

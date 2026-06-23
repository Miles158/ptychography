import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import models as models

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Load Experimental Data ---
data_path = os.path.join(os.path.dirname(__file__), '../data/experimental/stxm_06232036_001.pkl')
print(f"Loading experimental data from {data_path}...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

intensities = torch.tensor(data['intensities'], dtype=torch.float32, device=device)
diffraction_patterns = torch.tensor(data['diffraction_patterns'], dtype=torch.float32, device=device)
x_positions = torch.tensor(data['x_positions'], dtype=torch.float32, device=device) / 34555 / 1000 # Convert steps to meters
y_positions = torch.tensor(data['y_positions'], dtype=torch.float32, device=device) / 34555 / 1000 # Convert steps to meters
metadata = data['metadata']
spacing = metadata['spacing']

# Assuming grid size from diffraction patterns
grid_size = diffraction_patterns.shape[-1]
detector_pixels = grid_size
detector_size = grid_size * spacing # Placeholder, adjust based on actual physical setup

# --- 2. Initialize Learnable Model ---
# We need an initial probe guess. Using a simple circular aperture as a starting point.
initial_probe = torch.ones((grid_size, grid_size), dtype=torch.complex64, device=device)

model = models.LearnablePtychoModel(
    grid_size_object=grid_size,
    grid_spacing_object=spacing,
    wavelength=632e-9, # Placeholder, update with actual wavelength
    z_dist_init=10e-3, # Placeholder, update with actual distance
    probe_init=initial_probe,
    detector_size=detector_size,
    detector_pixels=detector_pixels,
    device=device
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
loss_history = []

# --- 3. Optimization Loop ---
num_epochs = 50
print("Starting ptychographic reconstruction...")

for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(len(diffraction_patterns)):
        optimizer.zero_grad()
        
        # Convert meter positions to pixel shifts
        shift_x = int(x_positions[i] / spacing)
        shift_y = int(y_positions[i] / spacing)
        
        # Forward pass
        pred = model(pixel_shift_x=shift_x, pixel_shift_y=shift_y)
        
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

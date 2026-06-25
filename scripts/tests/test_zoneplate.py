import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pickle

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import models as models
from src import object as obj

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Load Experimental Data ---
data_path = os.path.join(os.path.dirname(__file__), '../../data/experimental/stxm_06232036_002.pkl')
print(f"Loading experimental data from {data_path}...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Subsample immediately in numpy to keep memory footprint low
subsample_rate = 3
diffraction_patterns_np = np.array(data['diffraction_patterns'])[::subsample_rate, :, :]
x_positions_np = np.array(data['x_positions'])[::subsample_rate]
y_positions_np = np.array(data['y_positions'])[::subsample_rate]
loaded_probe_np = np.array(data['beam_profile_data']['averaged_profile'])

# Convert only the subsampled data to torch tensors
diffraction_patterns = torch.from_numpy(diffraction_patterns_np.astype(np.float32)).to(device)
x_positions_m = torch.from_numpy(x_positions_np.astype(np.float32)).to(device) / 34555 / 1000
y_positions_m = torch.from_numpy(y_positions_np.astype(np.float32)).to(device) / 34555 / 1000

# Zero the positions to center the scan pattern
x_positions_m = x_positions_m - torch.mean(x_positions_m)
y_positions_m = y_positions_m - torch.mean(y_positions_m)

# Swap x and y positions
y_positions_m = - y_positions_m


metadata = data['metadata']
spacing = metadata['spacing']

# Clean up large numpy arrays
del data, diffraction_patterns_np, x_positions_np, y_positions_np
import gc
gc.collect()

# --- 2. Setup Simulation Parameters ---
wavelength = 632e-9
z_dist =  0.075# Placeholder, update with actual distance
grid_size = 2048
object_extent = 6e-3
grid_spacing = object_extent / grid_size

# Assuming grid size from diffraction patterns
object_grid_shape = (diffraction_patterns.shape[-2], diffraction_patterns.shape[-1])
detector_shape_px = object_grid_shape
detector_shape_m = (object_grid_shape[0] * spacing, object_grid_shape[1] * spacing)

# --- 3. Create Zone Plate Object ---
print("Creating Zone Plate object...")
gt_object = obj.make_zone_plate_mask(
    grid_size=grid_size,
    pixel_size=grid_spacing,
    focal_length=0.2, # Adjust as needed
    wavelength=wavelength,
    device=device
)

# Visualize the created object
plt.figure(figsize=(6, 6))
plt.imshow(torch.abs(gt_object).cpu().numpy(), cmap='gray')
plt.title('Created Zone Plate Object')
plt.axis('off')
plt.savefig('plots/zone_plate_object.png')
print("Saved zone plate object to plots/zone_plate_object.png")
plt.close()

# --- 4. Create Probe ---
print("Creating Gaussian probe...")
source_intensity = 100
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(y, x, indexing='xy')
fwhm = 0.75e-3 # Set your desired FWHM in meters
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
probe = torch.exp(-(X**2 + Y**2) / (2 * sigma**2)).to(torch.complex64)
probe = probe * torch.sqrt(torch.tensor(source_intensity, device=device))

# --- 5. Initialize Model ---
print("Initializing PtychoModel...")
model = models.PtychoModel(
    object_grid_shape=(grid_size, grid_size), 
    grid_spacing_object=(grid_spacing, grid_spacing), 
    wavelength=wavelength, 
    z_dist_init=z_dist,
    device=device,
    detector_shape_m=detector_shape_m,
    detector_shape_px=detector_shape_px,
    probe_init=probe
).to(device)

model.set_object(gt_object)

# --- 6. Generate Simulated Patterns and Compare ---
print("Generating simulated patterns and plotting comparisons...")
os.makedirs('plots', exist_ok=True)

# Select a few indices to visualize
num_plots = 5
indices_to_plot = np.linspace(0, len(diffraction_patterns) - 1, num_plots, dtype=int)

fig, axes = plt.subplots(num_plots, 3, figsize=(15, 5 * num_plots))

with torch.no_grad():
    for row, idx in enumerate(indices_to_plot):
        # Get experimental pattern
        exp_pattern = diffraction_patterns[idx].cpu().numpy()
        
        # Calculate shifts
        shift_x = int(x_positions_m[idx] / grid_spacing)
        shift_y = int(y_positions_m[idx] / grid_spacing)
        
        # Generate simulated pattern
        sim_pattern = model(pixel_shift_x=shift_x, pixel_shift_y=shift_y).cpu().numpy()
        
        # Plot Experimental
        ax = axes[row, 0]
        im1 = ax.imshow(exp_pattern + 1e-6, cmap='inferno')
        ax.set_title(f"Experimental (Idx: {idx})\nx: {x_positions_m[idx]*1000:.2f}mm, y: {y_positions_m[idx]*1000:.2f}mm")
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot Simulated
        ax = axes[row, 1]
        im2 = ax.imshow(sim_pattern + 1e-6, cmap='inferno')
        ax.set_title(f"Simulated (Idx: {idx})")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot Difference
        ax = axes[row, 2]
        diff = exp_pattern - sim_pattern
        im3 = ax.imshow(diff, cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        ax.set_title(f"Difference (Exp - Sim)")
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('plots/zone_plate_comparison.png')
print("Saved comparison to plots/zone_plate_comparison.png")
plt.close(fig)

# --- 5. Comparison Visualization ---
model.eval()
with torch.no_grad():
    indices = [0, len(diffraction_patterns)//2, len(diffraction_patterns)-1]
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    
    # Row 1: Optimized Object/Probe
    extent = [-object_extent/2*1000, object_extent/2*1000, -object_extent/2*1000, object_extent/2*1000]
    axes[0, 0].imshow(model.object_amplitude.detach().cpu().numpy(), cmap='gray', extent=extent)
    axes[0, 0].set_title('Obj Amp (mm)')
    axes[0, 0].set_xlabel('x (mm)')
    axes[0, 0].set_ylabel('y (mm)')
    axes[0, 1].imshow(model.object_phase.detach().cpu().numpy(), cmap='twilight', extent=extent)
    axes[0, 1].set_title('Obj Phase')
    axes[0, 1].set_xlabel('x (mm)')
    axes[0, 1].set_ylabel('y (mm)')
    axes[0, 2].imshow(model.probe_amplitude.detach().cpu().numpy(), cmap='inferno')
    axes[0, 2].set_title('Probe Amp')
    axes[0, 3].imshow(model.probe_phase.detach().cpu().numpy(), cmap='twilight')
    axes[0, 3].set_title('Probe Phase')
    
    # Rows 2-4: Comparison
    for row, idx in enumerate(indices, 1):
        shift_x = int(x_positions_m[idx] / grid_spacing)
        shift_y = int(y_positions_m[idx] / grid_spacing)
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
extent = [-object_extent/2*1000, object_extent/2*1000, -object_extent/2*1000, object_extent/2*1000]


plt.subplot(1, 4, 2)
plt.imshow(model.object_amplitude.detach().cpu().numpy(), cmap='gray', extent=extent)
plt.title('Reconstructed Amplitude (mm)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

plt.subplot(1, 4, 3)
plt.imshow(model.object_phase.detach().cpu().numpy(), cmap='twilight', extent=extent)
plt.title('Reconstructed Phase')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

plt.subplot(1, 4, 4)
plt.imshow(model.probe_amplitude.detach().cpu().numpy(), cmap='inferno')
plt.title('Optimized Probe Amplitude')

plt.tight_layout()
plt.savefig('plots/experimental_reconstruction.png')
print("Saved reconstruction to plots/experimental_reconstruction.png")

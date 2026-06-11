import torch
import matplotlib.pyplot as plt
import numpy as np

from src import models as models 
from src import object as objects


# mps is the apple silicon GPU. "cpu" otherwise, or CUDA if its available.
device = "mps"

# --- 1. Setup Simulation Parameters ---
object_extent = 1e-3 # 75 mm
grid_size = 1024
grid_spacing = object_extent / grid_size
wavelength = 500e-9
z_dist = 250e-3 #
detector_size = 10e-3 # 15 mm

# Setup coordinate grid
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')


# --- usage ---
core_grid_size = 256
gt_object = objects.make_text_mask("ALS", core_grid_size, margin=0.1, stroke_frac=0.01, font_name="Helvetica", device=device)

# The model's grid size defines the fully padded simulation space
# We pass the smaller core object to set_object, and the model handles the padding internally.
grid_size = 1024
object_extent = grid_size * grid_spacing

# Re-setup coordinate grid for the padded size
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

###################################################################################################

# Probe parameters
# STXM requires a small, focused probe. Let's use 50um.
probe_radius = 1e-6

# Generate synthetic measurements by scanning the illumination over the unpadded core region
scan_step_meters = 5e-6 # Physical distance (gap) between scan points
shift_step = max(1, int(scan_step_meters / grid_spacing)) # step size in pixels

# Calculate number of scan positions to cover the unpadded core region
core_extent = core_grid_size * grid_spacing
num_positions = int(core_extent / scan_step_meters) + 1

# Model used to generate synthetic data. 
gt_model = models.STXM(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device,
    detector_size=detector_size
).to(device)

gt_model.set_object(gt_object)

# Create the probe using the radius defined above
probe_amplitude = torch.exp(-(X**2 + Y**2) / (2 * probe_radius**2))
# In STXM, phase curvature isn't strictly necessary as we integrate intensity, but keeping it flat is fine.
probe = probe_amplitude.to(torch.complex64).to(device)

# STXM is a direct imaging technique. We don't optimize parameters, we just map the 
# integrated intensity measured at each scan point to a 2D image.
scan_points = num_positions
stxm_image = np.zeros((scan_points, scan_points))

# To correctly plot the image, we map grid positions
start_idx = -(scan_points // 2) + 1
end_idx = (scan_points // 2) + 1

print(f"Running STXM scan: {scan_points}x{scan_points} = {scan_points**2} points...")

with torch.no_grad():
    # Iterate through the grid of scan positions
    for i, shift_y_idx in enumerate(range(start_idx, end_idx)):
        for j, shift_x_idx in enumerate(range(start_idx, end_idx)):
            shift_x = shift_x_idx * shift_step
            shift_y = shift_y_idx * shift_step
            
            # Record the total transmitted intensity
            intensity = gt_model(probe, pixel_shift_x=shift_x, pixel_shift_y=shift_y)
            stxm_image[i, j] = intensity.item()

print("Scan complete.")

# --- 5. Visualization ---
plt.figure(figsize=(10, 5))

# Ground truth extent for plotting
plot_extent = (-object_extent/2, object_extent/2, -object_extent/2, object_extent/2)

# Plot 1: Ground Truth Amplitude
plt.subplot(1, 2, 1)
plt.imshow(torch.abs(gt_model.object_amplitude).detach().cpu().numpy(), extent=plot_extent, origin='lower', cmap='gray')
plt.title('Object Plane')
plt.xlabel('(m)')
plt.ylabel('(m)')

plt.colorbar()

# Plot 2: STXM Image
plt.subplot(1, 2, 2)


# Coordinates of scan points (physical)
coords = np.linspace(start_idx * shift_step * grid_spacing,
                     (end_idx-1) * shift_step * grid_spacing,
                     scan_points)

Xc, Yc = np.meshgrid(coords, coords)

plt.scatter(
    Xc.flatten(), Yc.flatten(),
    c=stxm_image.flatten(),
    s=2, marker='s',
    cmap='gray'
)
plt.gca().set_aspect('equal', 'box')
plt.title('Reconstruction')
plt.xlabel('X scan shift (m)')
plt.ylabel('Y scan shift(m)')
plt.colorbar()



# # Calculate physical extent of the scanned area
# scan_physical_extent = (
#     start_idx * shift_step * grid_spacing,
#     end_idx * shift_step * grid_spacing,
#     start_idx * shift_step * grid_spacing,
#     end_idx * shift_step * grid_spacing
# )



# plt.imshow(stxm_image, extent=scan_physical_extent, origin='lower', cmap='gray')
# plt.title('Detector Plane')
# plt.xlabel('(m)')
# plt.ylabel('(m)')

# plt.colorbar()

plt.tight_layout()
plt.savefig('stxm_reconstruction.png')
print("Saved STXM image to 'stxm_reconstruction.png'")
# plt.show()
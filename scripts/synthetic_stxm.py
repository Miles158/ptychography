import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import src.models as models

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

########################################################################################
# This is an AI-generated function that draws the letters ALS (upside down :/) as a test object. 

from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.font_manager as fm

def make_text_mask(text, grid_size, margin=0., stroke_frac=0.01, font_name="Helvetica", device="cpu"):
    # Find an installed font path using matplotlib
    font_path = fm.findfont(font_name, fallback_to_default=True)
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Could not locate a TTF for font_name={font_name}. "
                                f"Matplotlib returned: {font_path}")

    # Create grayscale canvas
    img = Image.new("L", (grid_size, grid_size), 0)
    draw = ImageDraw.Draw(img)

    stroke_w = max(1, int(grid_size * stroke_frac))  # keep this as-is

    usable = int((1 - 2 * margin) * grid_size)
    lo, hi = 1, grid_size
    best = lo

    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bbox = draw.textbbox((0, 0), text, font=f)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # <-- IMPORTANT: compensate for stroke (textbbox does NOT include stroke)
        if (w + 2 * stroke_w) <= usable and (h + 2 * stroke_w) <= usable:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    font = ImageFont.truetype(font_path, best)

    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x0 = (grid_size - w) // 2
    y0 = (grid_size - h) // 2

    draw.text(
        (x0, y0), text,
        font=font,
        fill=255,
        stroke_width=stroke_w,
        stroke_fill=255
    )

    mask = (np.array(img) > 128).astype(np.float32)
    return torch.from_numpy(mask).to(torch.complex64).to(device)


# --- usage ---
core_grid_size = 256
gt_object = make_text_mask("ALS", core_grid_size, margin=0.1, stroke_frac=0.01, font_name="Helvetica", device=device)

# The model's grid size defines the fully padded simulation space
# We pass the smaller core object to set_object, and the model handles the padding internally.
grid_size = 1024
object_extent = grid_size * grid_spacing

# Re-setup coordinate grid for the padded size
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

###################################################################################################

# Probe parameters (needed early for overlap calculation)
# STXM requires a small, focused probe. Let's use 50um.
probe_radius = 1e-6

# Generate synthetic measurements by scanning the illumination over the unpadded core region
desired_overlap = 0.66 # e.g., 66% overlap
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

print(f"Running STXM scan: {scan_points}x{scan_points} points...")

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
plt.title('Ground Truth Object (Padded)')
plt.colorbar()

# Plot 2: STXM Image
plt.subplot(1, 2, 2)
# Calculate physical extent of the scanned area
scan_physical_extent = (
    start_idx * shift_step * grid_spacing,
    end_idx * shift_step * grid_spacing,
    start_idx * shift_step * grid_spacing,
    end_idx * shift_step * grid_spacing
)
plt.imshow(stxm_image, extent=scan_physical_extent, origin='lower', cmap='gray')
plt.title('STXM Image (Transmitted Intensity)')
plt.colorbar()

plt.tight_layout()
plt.savefig('stxm_reconstruction.png')
print("Saved STXM image to 'stxm_reconstruction.png'")
# plt.show()
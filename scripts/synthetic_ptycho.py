import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import src.models as models



# mps is the apple silicon GPU. "cpu" otherwise, or CUDA if its available.
device = "mps"

# --- 1. Setup Simulation Parameters ---
object_extent = 75e-3 # 75 mm
grid_size = 1028 
grid_spacing = object_extent / grid_size
wavelength = 500e-9
z_dist = 250e-3 #

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
gt_object = make_text_mask("ALS", grid_size, margin=0.25, stroke_frac=0.01, font_name="Helvetica", device=device)

###################################################################################################


# Generate synthetic measurements by scanning the illumination over the object
num_positions = 12
shift_step = int(grid_size / 30) # step size in pixels (smaller step = more overlap)

shifts = []
for i in range(-num_positions//2 + 1, num_positions//2 + 1):
    for j in range(-num_positions//2 + 1, num_positions//2 + 1):
        shifts.append((i * shift_step, j * shift_step))

print(f"Generating synthetic data for {len(shifts)} scan positions...")

# Model used to generate synthetic data. 
gt_model = models.PtychoModel(
    grid_size=grid_size, 
    grid_spacing=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device
).to(device)
gt_model.set_object(gt_object)

# Gaussian illumination for the sample.
probe_radius = 4e-3 # Could sanity check. 
probe = torch.exp(-(X**2 + Y**2) / (2 * probe_radius**2)).to(torch.complex64).to(device)

# Collect measurements
measurements = []
with torch.no_grad():
    for shift_x, shift_y in shifts:
        intensity = gt_model(probe, pixel_shift_x=shift_x, pixel_shift_y=shift_y)
        measurements.append(intensity)
measurements = torch.stack(measurements)

print("Measurements generated. Starting optimization...")

# --- 2. Initialize the Reconstruction Model ---
model = models.PtychoModel(
    grid_size=grid_size, 
    grid_spacing=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device
).to(device)

# Rational starting point for optimiation 
model.object_amplitude.data = torch.ones((grid_size, grid_size), dtype=torch.float32, device=device)
model.object_phase.data = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)

# --- 3. Setup Optimizer ---
optimizer = optim.AdamW(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

num_epochs = 60
loss_history = []

# --- 4. Optimization Loop ---
for epoch in range(num_epochs):
    optimizer.zero_grad()
    total_loss = 0.0

    for idx, (shift_x, shift_y) in enumerate(shifts):
        predicted = model(probe, pixel_shift_x=shift_x, pixel_shift_y=shift_y)
        total_loss = total_loss + criterion(predicted, measurements[idx])

    total_loss = total_loss / len(shifts)
    total_loss.backward()
    optimizer.step()

    loss_val = total_loss.item()
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
plt.imshow(model.object_amplitude.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed Amplitude')
plt.colorbar()

# Plot 4: Reconstructed Phase
plt.subplot(2, 2, 4)
plt.imshow(model.object_phase.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed Phase')
plt.colorbar()

plt.tight_layout()
plt.savefig('ptychography_reconstruction.png')
print("Saved reconstruction results to 'ptychography_reconstruction.png'")
# plt.show()
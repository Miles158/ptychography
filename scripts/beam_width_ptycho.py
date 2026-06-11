import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import src.models as models
import src.object as obj

# ----------------------------
# Config
# ----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Setup Simulation Parameters ---
grid_size = 1024
object_extent = 75e-3
grid_spacing = object_extent / grid_size
wavelength = 500e-9
z_dist = 250e-3
detector_size = 1.5e-3

# Coordinate grid (simulation grid)
x = torch.linspace(-object_extent / 2, object_extent / 2, grid_size, device=device)
y = torch.linspace(-object_extent / 2, object_extent / 2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing="xy")

# --- usage / GT object (core then pad) ---
core_grid_size = 256
gt_object_core = obj.make_text_mask(
    "A",
    core_grid_size,
    margin=0.01,
    stroke_frac=0.01,
    font_name="Helvetica",
    device=device,
)

pad_pixels = (grid_size - core_grid_size) // 2
gt_object = torch.nn.functional.pad(
    gt_object_core,
    (pad_pixels, pad_pixels, pad_pixels, pad_pixels),
    mode="constant",
    value=0.0,
)

# -------------------------------------------------------
# Sweep over probe radii (beam widths)
# -------------------------------------------------------
probe_radii = [2.0e-3, 2.4e-3]  # meters
probe_radii = [1.5e-3, 2.0e-3, 3.0e-3, 4.0e-3, 5.0e-3]
epochs = [50,100,200,400,800]
# Desired overlap and scan configuration
desired_overlap = 0.7

# Ptychography optimization parameters
num_epochs = 75
learning_rate = 0.001
criterion = nn.MSELoss()

# Use a fixed phase for the illumination (top-hat aperture)
phi0 = 0.0

# Precompute r (radius grid) once
r = torch.sqrt(X**2 + Y**2)

# Ground-truth physics model (used only to generate measurements)
gt_model = models.PtychoModel(
    grid_size_object=grid_size,
    grid_spacing_object=grid_spacing,
    wavelength=wavelength,
    z_dist=z_dist,
    device=device,
    detector_size=detector_size,
).to(device)
gt_model.set_object(gt_object)

# Extent for plotting (physical coordinates)
plot_extent = (-object_extent / 2, object_extent / 2, -object_extent / 2, object_extent / 2)

# Storage for all sweep results
sweep_results = {
    "probe_radii_m": [],
    "probe_diameters_m": [],
    "loss_histories": [],       # list of list[float]
    "final_losses": [],         # per sweep element
    "best_losses": [],          # min over epochs per sweep element
    "best_epoch_indices": [],   # per sweep element
    "reconstructions": [],     # list of dict with amplitude/phase arrays + metadata
}

# Optional: keep all reconstructed figures in memory for later plotting
recon_amplitudes = []
recon_phases = []

# ----------------------------
# Main sweep loop
# ----------------------------
for sweep_idx, probe_radius in enumerate(probe_radii):
    print(f"\n=== Sweep {sweep_idx+1}/{len(probe_radii)}: probe_radius = {probe_radius*1e3:.3f} mm ===")

    # Build probe for this radius (top-hat aperture)
    mask = (r <= probe_radius).to(torch.float32)
    probe = mask.to(torch.complex64) * torch.exp(torch.tensor(1j * phi0, device=device))

    # Determine scan step based on overlap
    probe_diameter = 2 * probe_radius
    shift_step_meters = probe_diameter * (1 - desired_overlap)
    shift_step_pixels = max(1, int(shift_step_meters / grid_spacing))

    # Number of scan positions to cover the unpadded core region
    core_extent = core_grid_size * grid_spacing
    num_positions = int(core_extent / shift_step_meters) + 1

    shifts = []
    for i in range(-num_positions // 2 + 1, num_positions // 2 + 1):
        for j in range(-num_positions // 2 + 1, num_positions // 2 + 1):
            # The model's pixel_shift expects an integer (number of pixels)
            shifts.append((i * shift_step_pixels, j * shift_step_pixels))

    print(f"Generating synthetic data for {len(shifts)} scan positions (overlap {desired_overlap*100:.1f}%)...")

    # Generate measurements for this probe
    measurements = []
    with torch.no_grad():
        for shift_x, shift_y in shifts:
            intensity = gt_model(probe, pixel_shift_x=shift_x, pixel_shift_y=shift_y)
            measurements.append(intensity)
    measurements = torch.stack(measurements)

    print("Measurements generated. Starting optimization...")

    # ----------------------------
    # Reconstruction model (fresh init per probe size)
    # ----------------------------
    recon_model = models.PtychoModel(
        grid_size_object=grid_size,
        grid_spacing_object=grid_spacing,
        wavelength=wavelength,
        z_dist=z_dist,
        device=device,
        detector_size=detector_size,
    ).to(device)

    # Re-init object estimate
    recon_model.object_amplitude.data = torch.ones((grid_size, grid_size), dtype=torch.float32, device=device)
    recon_model.object_phase.data = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)

    optimizer = optim.AdamW(recon_model.parameters(), lr=learning_rate)

    loss_history = []
    best_loss = float("inf")
    best_epoch = -1
    best_state = None  # store best recon tensors (CPU later)

    # ----------------------------
    # Optimization loop
    # ----------------------------
    num_epochs = epochs[sweep_idx]
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for idx, (shift_x, shift_y) in enumerate(shifts):
            optimizer.zero_grad()

            predicted_intensity = recon_model(
                probe, pixel_shift_x=shift_x, pixel_shift_y=shift_y
            )

            loss = criterion(predicted_intensity, measurements[idx])
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Amplitude constraint
            with torch.no_grad():
                recon_model.object_amplitude.data = torch.clamp(recon_model.object_amplitude.data, 0, 1)

        loss_val = epoch_loss / len(shifts)
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            # snapshot best state (on device, then we'll move to CPU once per sweep)
            best_state = {
                "object_amplitude": recon_model.object_amplitude.detach().clone(),
                "object_phase": recon_model.object_phase.detach().clone(),
            }

        print(f"  Epoch {epoch+1:03d}/{num_epochs}, Loss: {loss_val:.4e}")

    # ----------------------------
    # Save per-sweep results
    # ----------------------------
    final_loss = float(loss_history[-1])

    # Move best reconstruction to CPU for saving/plotting
    best_amp = best_state["object_amplitude"].detach().cpu().numpy()
    best_phase = best_state["object_phase"].detach().cpu().numpy()

    sweep_results["probe_radii_m"].append(float(probe_radius))
    sweep_results["probe_diameters_m"].append(float(probe_diameter))
    sweep_results["loss_histories"].append(loss_history)
    sweep_results["final_losses"].append(final_loss)
    sweep_results["best_losses"].append(float(best_loss))
    sweep_results["best_epoch_indices"].append(int(best_epoch))
    sweep_results["reconstructions"].append({
        "probe_radius_m": float(probe_radius),
        "probe_diameter_m": float(probe_diameter),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "final_loss": float(final_loss),
        "object_amplitude": best_amp,  # numpy arrays for pickle
        "object_phase": best_phase,    # numpy arrays for pickle
    })

    recon_amplitudes.append(best_amp)
    recon_phases.append(best_phase)

    print(f"Done sweep. Best loss={best_loss:.4e} at epoch {best_epoch+1}.")

# ----------------------------
# Find global best probe size
# ----------------------------
best_idx = int(np.argmin(sweep_results["best_losses"]))
best_probe_radius = sweep_results["probe_radii_m"][best_idx]
best_probe_diameter = sweep_results["probe_diameters_m"][best_idx]
best_probe_loss = sweep_results["best_losses"][best_idx]
print(f"\nGLOBAL BEST: probe_radius={best_probe_radius*1e3:.3f} mm "
      f"(diameter={best_probe_diameter*1e3:.3f} mm), best_loss={best_probe_loss:.4e}")

# ----------------------------
# Save all results to a single pkl
# ----------------------------
out_pkl = "results/ptychography_sweep_results.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump(sweep_results, f)

print(f"Saved all sweep results to: {out_pkl}")

# ----------------------------
# Plot: loss vs beam size (use best loss per sweep)
# ----------------------------
beam_sizes_plot = np.array(sweep_results["probe_diameters_m"]) * 1e3  # mm
best_losses_plot = np.array(sweep_results["best_losses"])

plt.figure(figsize=(7, 4.5))
plt.plot(beam_sizes_plot, best_losses_plot, marker="o")
plt.xlabel("Probe diameter (mm)")
plt.ylabel("Best MSE loss")
plt.title("Loss vs Probe Diameter")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/loss_vs_beam_diameter.png", dpi=200)
print("Saved: loss_vs_beam_diameter.png")

# ----------------------------
# Plot: all reconstructed images (amplitude + phase)
# ----------------------------
n = len(probe_radii)
fig, axes = plt.subplots(n, 2, figsize=(10, 2.6*n), constrained_layout=True)

if n == 1:
    axes = np.array([axes])

for i in range(n):
    # Amplitude
    ax_amp = axes[i, 0]
    im0 = ax_amp.imshow(recon_amplitudes[i], extent=plot_extent, origin="lower")
    ax_amp.set_title(f"Amplitude\nprobe R={probe_radii[i]*1e3:.2f} mm")
    ax_amp.set_xlabel("x (m)")
    ax_amp.set_ylabel("y (m)")
    fig.colorbar(im0, ax=ax_amp, fraction=0.046, pad=0.01)

    # Phase
    ax_ph = axes[i, 1]
    im1 = ax_ph.imshow(recon_phases[i], extent=plot_extent, origin="lower")
    ax_ph.set_title(f"Phase\nprobe R={probe_radii[i]*1e3:.2f} mm")
    ax_ph.set_xlabel("x (m)")
    ax_ph.set_ylabel("y (m)")
    fig.colorbar(im1, ax=ax_ph, fraction=0.046, pad=0.01)

fig.suptitle("All reconstructions from probe-size sweep", y=1.02)
fig.savefig("plots/all_reconstructed_images.png", dpi=200)
print("Saved: all_reconstructed_images.png")

# ----------------------------
# (Optional) Plot loss curves for each probe size
# ----------------------------
plt.figure(figsize=(8, 5))
for i in range(n):
    plt.plot(sweep_results["loss_histories"][i], label=f"R={probe_radii[i]*1e3:.2f}mm")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Loss curves for all probe sizes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/loss_curves_all_probes.png", dpi=200)
print("Saved: loss_curves_all_probes.png")
"""Inspect reconstructed ptychography results and compare simulated vs measured
diffraction at the sample edges.

This script mirrors the data loading and position conversions used in
`scripts/reconstruct_ptycho.py`. It loads the saved reconstruction results
(`plots/reconstruction_result.pkl`) and simulates diffraction patterns at
positions near the left/right edges of the scan using `fresnel_propagate` from
`src/propagation.py`. It then saves a comparison figure showing the probe,
object amplitude, and side-by-side measured vs simulated diffraction patterns.
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import zoom

# Import propagation from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import propagation as prop


def load_data():
	data_path = os.path.join(os.path.dirname(__file__), '../data/experimental/stxm_06252036_001.pkl')
	with open(data_path, 'rb') as f:
		data = pickle.load(f)

	diffraction_patterns = np.array(data['diffraction_patterns']).astype(np.float32)
	x_positions = np.array(data['x_positions']).astype(np.float32)
	y_positions = np.array(data['y_positions']).astype(np.float32)
	spacing_meta = data['metadata'].get('spacing', None)

	# Convert steps -> meters using same factor as reconstruct_ptycho.py (24332 then /1000)
	x_m = x_positions / 24332.0 / 1000.0
	y_m = y_positions / 24332.0 / 1000.0

	# Center and flip y to match reconstruction code
	x_m = x_m - np.mean(x_m)
	y_m = y_m - np.mean(y_m)
	y_m = -y_m

	return diffraction_patterns, x_m, y_m, spacing_meta


def shift_object_np(obj, shift_x, shift_y):
	"""Shift object (numpy complex array) by integer pixel amounts without wrap.

	Positive shift_x shifts the object to the right.
	"""
	H, W = obj.shape
	pad = max(abs(int(shift_x)), abs(int(shift_y)), 1)
	padded = np.zeros((H + 2 * pad, W + 2 * pad), dtype=obj.dtype)
	padded[pad:pad + H, pad:pad + W] = obj

	# Extract region shifted by (shift_x, shift_y)
	sx = pad - int(shift_x)
	sy = pad - int(shift_y)
	shifted = padded[sy:sy + H, sx:sx + W]
	return shifted


def simulate_diffraction_at_positions(results, obj_complex, probe_complex, edge_indices, x_m, grid_spacing, wavelength, z_dist, diffraction_meas):
    """Simulate diffraction for a list of edge scan indices and return simulated intensities.
    Returns list of tuples (meas, sim_resampled, sim_full)
    """
    results_list = []
    obj = obj_complex
    probe = probe_complex

    # figure out pixel spacing scalar
    if isinstance(grid_spacing, (list, tuple, np.ndarray)):
        dx = float(grid_spacing[0])
    else:
        dx = float(grid_spacing)

    H, W = obj.shape

    for scan_idx in edge_indices:
        # Get actual position in meters from x_m array
        pos_x_m = x_m[scan_idx]
        shift_x_pix = int(np.round(pos_x_m / dx))
        shift_y_pix = 0  # evaluate central row / edge, keep y shift zero

        shifted_obj = shift_object_np(obj, shift_x_pix, shift_y_pix)

        # form exit wave and propagate using torch
        exit_wave = torch.tensor(probe * shifted_obj, dtype=torch.complex64)
        det_wave = prop.fresnel_propagate(exit_wave, z_dist, (dx, dx), wavelength)
        sim_intensity = np.abs(det_wave.cpu().numpy()) ** 2

        # resample to measured diffraction shape
        meas = diffraction_meas[scan_idx]
        meas_H, meas_W = meas.shape
        sim_H, sim_W = sim_intensity.shape
        zoom_f = (meas_H / sim_H, meas_W / sim_W)
        sim_resampled = zoom(sim_intensity, zoom_f, order=1)

        results_list.append((meas, sim_resampled, sim_intensity, scan_idx))

    return results_list


def main():
    diffraction_patterns, x_m, y_m, spacing_meta = load_data()

    # Load reconstruction results
    res_path = os.path.join('plots', 'reconstruction_result.pkl')
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Reconstruction results not found at {res_path}")

    with open(res_path, 'rb') as f:
        results = pickle.load(f)

    probe = results['probe']
    obj = results['object']
    z_dist = results.get('z_dist_m', 0.053)
    grid_spacing = results.get('grid_spacing_object_m', None)
    wavelength = 632e-9

    # choose edge positions (leftmost, middle, and rightmost)
    sorted_idx = np.argsort(x_m)
    left_idxs = sorted_idx[:2].tolist()
    mid_idxs = sorted_idx[len(sorted_idx)//2-1:len(sorted_idx)//2+1].tolist()
    right_idxs = sorted_idx[-2:].tolist()
    edge_idxs = left_idxs + mid_idxs + right_idxs

    # simulate
    comparisons = simulate_diffraction_at_positions(
        results, obj, probe, edge_idxs, x_m, grid_spacing, wavelength, z_dist, diffraction_patterns
    )

    # Plot probe, object amplitude, and side-by-side measured vs simulated at edges
    os.makedirs('plots', exist_ok=True)
    fig, axes = plt.subplots(3, max(1, len(edge_idxs)), figsize=(4 * len(edge_idxs), 10), constrained_layout=True)

    # probe amplitude
    if fig is None:
        return

    for col, idx in enumerate(edge_idxs):
        if col == 0:
            ax = axes[0, col]
        else:
            ax = axes[0, col]
        im0 = ax.imshow(np.abs(probe), cmap='inferno')
        ax.set_title(f'Probe Amplitude')
        ax.axis('off')
        fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # object amplitude -> replaced with measured diffraction at same position
    for col, scan_idx in enumerate(edge_idxs):
        ax = axes[1, col]
        im1 = ax.imshow(diffraction_patterns[scan_idx], cmap='inferno')
        ax.set_title(f'Measured idx {scan_idx}')
        ax.axis('off')
        fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # measured vs simulated diffraction per edge column
    for col, (meas, sim_resampled, sim_full, scan_idx) in enumerate(comparisons):
        ax = axes[2, col]
        im2 = ax.imshow(sim_resampled, cmap='inferno')
        ax.set_title(f'Simulated idx {scan_idx}')
        ax.axis('off')
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        # also save measured as separate file next to simulated
        fig2, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))
        im_meas = a1.imshow(meas, cmap='inferno')
        a1.set_title(f'Measured idx {scan_idx}')
        a1.axis('off')
        plt.colorbar(im_meas, ax=a1, fraction=0.046, pad=0.04)
        im_sim = a2.imshow(sim_resampled, cmap='inferno')
        a2.set_title(f'Simulated idx {scan_idx}')
        a2.axis('off')
        plt.colorbar(im_sim, ax=a2, fraction=0.046, pad=0.04)
        small_path = os.path.join('plots', f'edge_compare_idx_{scan_idx}.png')
        fig2.savefig(small_path, dpi=300)
        plt.close(fig2)

    out = os.path.join('plots', 'inspect_edge_overview.png')
    fig.savefig(out, dpi=300)
    print(f'Saved edge inspection overview to {out}')


if __name__ == '__main__':
	main()


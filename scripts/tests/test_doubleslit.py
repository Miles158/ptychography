"""Test double-slit Fresnel propagation against analytical Fresnel diffraction.

This script builds a simple 2D double-slit aperture (slits running along y,
so the problem is effectively 1D in x), propagates it using
`fresnel_propagate` from `src/propagation.py`, and compares the resulting
intensity to an analytical Fresnel double-slit diffraction pattern.

The comparison is made visually via saved plots with physical units.
"""
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import fresnel

# Ensure we can import the propagation module from src
import propagation as prop


def analytical_double_slit_intensity(x, wavelength, z, slit_sep, slit_width):

    # edges: [left1,right1] and [left2,right2]
    centers = np.array([-slit_sep/2, slit_sep/2])
    left_edges  = centers - slit_width/2
    right_edges = centers + slit_width/2

    def integrals(edge):
        arg = np.sqrt(2/(wavelength*z)) * (x - edge) 
        sine_integral, cosine_integral = fresnel(arg)
        return cosine_integral + 1j*sine_integral

    field = (integrals(left_edges[0]) - integrals(right_edges[0])) + (integrals(left_edges[1]) - integrals(right_edges[1]))
    return np.abs(field)**2


# Parameters
wavelength = 632e-9
z = 0.07  

nx = 1024 + 500
ny = 36
object_extent = 10e-3
dx = object_extent / nx

slit_width = 25e-6
slit_sep = 200e-6

# Build 1D slit profile along x, extend uniformly along y
x = np.linspace(-object_extent / 2, object_extent / 2, nx)
aperture_x = np.zeros_like(x)
aperture_x[np.abs(x + slit_sep / 2) < (slit_width / 2)] = 1.0
aperture_x[np.abs(x - slit_sep / 2) < (slit_width / 2)] = 1.0

aperture = np.tile(aperture_x[np.newaxis, :], (ny, 1))

# Convert to torch complex wavefront (plane wave * aperture)
wavefront = torch.tensor(aperture, dtype=torch.complex64)

# Propagate using the Fresnel propagator under test
detector_wave = prop.fresnel_propagate(
    wavefront, z, (dx, dx), wavelength
)

intensity_num = np.abs(detector_wave.cpu().numpy()) ** 2

# With the Fresnel transfer-function implementation, the output grid spacing
# is the same as the input plane spacing, so detector x coordinates are the
# same physical x positions as the object plane.


x_det = x

# Analytical 1D Fresnel intensity (same for every detector row since slits run along y)
intensity_analytic_1d = analytical_double_slit_intensity(
    x_det, wavelength, z, slit_sep, slit_width
)

# Compare the central row of the numerical intensity to the analytical Fresnel pattern
central_row = intensity_num[ny // 2, :]

central_row_norm = central_row / (central_row.max() + 1e-20)
analytic_norm = intensity_analytic_1d / (intensity_analytic_1d.max() + 1e-20)


# Normalize both patterns to peak = 1
central_row_norm = np.fft.fftshift(central_row_norm)
analytic_norm = np.fft.fftshift(analytic_norm)
x_mm = np.fft.fftshift(x_det) * 1e3



plots_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plots')
os.makedirs(plots_dir, exist_ok=True)

analytic_image = np.tile(analytic_norm[np.newaxis, :], (ny, 1))
numeric_log = np.log10(intensity_num + 1e-12)
analytic_log = np.log10(analytic_image + 1e-12)
x_mm = x * 1e3

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

axes[0, 0].imshow(
    aperture,
    cmap='gray',
    extent=[x_mm[0], x_mm[-1], 0, ny],
    aspect='auto',
)
axes[0, 0].set_title(f'Aperture prop dist:{z} (m), wavelength:{wavelength} (m)')
axes[0, 0].set_xlabel('x (mm)')
axes[0, 0].set_ylabel('y index')

im1 = axes[0, 1].imshow(
    numeric_log,
    cmap='inferno',
    extent=[x_mm[0], x_mm[-1], ny / 2, -ny / 2],
    aspect='auto',
)
axes[0, 1].set_title('Numerical Fresnel Diffraction Intensity (log10)')
axes[0, 1].set_xlabel('Detector x position (mm)')
axes[0, 1].set_ylabel('Detector y index')
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im2 = axes[1, 0].imshow(
    analytic_log,
    cmap='inferno',
    extent=[x_mm[0], x_mm[-1], ny / 2, -ny / 2],
    aspect='auto',
)
axes[1, 0].set_title('Analytical Fresnel Diffraction Intensity (log10)')
axes[1, 0].set_xlabel('Detector x position (mm)')
axes[1, 0].set_ylabel('Detector y index')
fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

axes[1, 1].plot(x_mm, central_row_norm, label='Numerical Fresnel')
axes[1, 1].plot(x_mm, analytic_norm, label='Analytical Fresnel', linestyle='--')
axes[1, 1].set_xlabel('Detector x position (mm)')
axes[1, 1].set_ylabel('Normalized intensity')
axes[1, 1].set_title('Central row comparison')
axes[1, 1].legend()
axes[1, 1].grid(True)

fig_path = os.path.join(plots_dir, 'test_doubleslit_comparison.png')
fig.savefig(fig_path, dpi=200)
print(f"Saved plot to {fig_path}")


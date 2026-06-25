"""Test double-slit Fraunhofer pattern against analytical formula.

This script builds a simple 2D double-slit aperture (slits running along y,
so the problem is effectively 1D in x), propagates it using
`fraunhofer_propagate` from `src/propagation.py`, and compares the resulting
intensity (on the detector coordinates returned by the FFT mapping) to the
analytical Fraunhofer double-slit formula

The comparison is made on normalized intensities and asserted to be within a
reasonable mean-squared error tolerance.
"""
import sys
import os
import numpy as np
import torch

# Ensure we can import the propagation module from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import propagation as prop


def analytical_double_slit_intensity(x, wavelength, z, slit_sep, slit_width):
    """Compute 1D Fraunhofer double-slit intensity at positions x (meters).

    Formula (small-angle / Fraunhofer):
      I(x) ~ cos^2( pi * d * x / (wavelength * z) ) * sinc^2( pi * a * x / (wavelength * z) )
    where d = slit center separation, a = slit width.
    """
    arg_cos = np.pi * slit_sep * x / (wavelength * z)
    arg_sinc = np.pi * slit_width * x / (wavelength * z)

    # cos^2 term
    envelope = np.cos(arg_cos) ** 2

    # sinc handling: sinc(x) = sin(x)/x
    sinc = np.ones_like(arg_sinc)
    nz = np.abs(arg_sinc) > 1e-12
    sinc[nz] = np.sin(arg_sinc[nz]) / arg_sinc[nz]

    return envelope * (sinc ** 2)


def run_test():
    # Parameters
    wavelength = 632e-9
    z = 0.5  # long distance to be comfortably in Fraunhofer region (m)

    nx = 2048
    ny = 256
    object_extent = 5.5e-3
    dx = object_extent / nx

    slit_width = 50e-6
    slit_sep = 200e-6

    # Build 1D slit profile along x, extend uniformly along y
    x = np.linspace(-object_extent / 2, object_extent / 2, nx)
    aperture_x = np.zeros_like(x)
    aperture_x[np.abs(x + slit_sep / 2) < (slit_width / 2)] = 1.0
    aperture_x[np.abs(x - slit_sep / 2) < (slit_width / 2)] = 1.0

    aperture = np.tile(aperture_x[np.newaxis, :], (ny, 1))

    # Convert to torch complex wavefront (plane wave * aperture)
    wavefront = torch.tensor(aperture, dtype=torch.complex64)

    # Propagate using the fraunhofer_propagate helper
    detector_wave = prop.fraunhofer_propagate(
        wavefront, z, (dx, dx), wavelength
    )

    intensity_num = np.abs(detector_wave.cpu().numpy()) ** 2

    # Build detector X coordinates used by fraunhofer_propagate
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    x_det = fx * wavelength * z

    # Analytical 1D intensity (same for every detector row since slits run along y)
    intensity_analytic_1d = analytical_double_slit_intensity(
        x_det, wavelength, z, slit_sep, slit_width
    )

    # Compare the central row of the numerical intensity to analytic 1D pattern
    central_row = intensity_num[ny // 2, :]

    # Normalize both patterns to peak=1 for shape comparison
    central_row_norm = central_row / (central_row.max() + 1e-20)
    analytic_norm = intensity_analytic_1d / (intensity_analytic_1d.max() + 1e-20)

    mse = np.mean((central_row_norm - analytic_norm) ** 2)
    print(f"Double-slit Fraunhofer comparison MSE: {mse:.3e}")

    # Loose tolerance to allow sampling & windowing differences
    assert mse < 5e-3, f"MSE too large: {mse:.3e}"


if __name__ == "__main__":
    run_test()

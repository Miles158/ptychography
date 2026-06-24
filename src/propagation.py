"""
propagation.py
Written by Mile Houston

Pytorch functions for optical propagation used in imaging simulations.
"""
import torch.nn.functional as functional
import torch
import numpy as np


# this function is responsible for the fresnel propagation of the wavefront from the mask to the sample plane.
def fresnel_propagate(wavefront, z_dist, grid_spacing, wavelength):
    nx, ny = wavefront.shape
    device = wavefront.device

    # form the spatial frequency bins for the FFT
    Fx = torch.fft.fftfreq(nx, d=grid_spacing, device=device)
    Fy = torch.fft.fftfreq(ny, d=grid_spacing, device=device)
    Vx, Vy = torch.meshgrid(Fy, Fx, indexing="xy")

    k = 2 * torch.pi / wavelength

    wavefront_ft = torch.fft.fft2(wavefront)
    transfer_func = torch.exp(torch.tensor(1j * k * z_dist, device=device)) * torch.exp(
        -1j * torch.pi * wavelength * z_dist * (Vx**2 + Vy**2)
    )
    # Fresnel approximation for near-field propagation
    wavefront_transferred = wavefront_ft * transfer_func

    return torch.fft.ifft2(wavefront_transferred)

def huygens_propagate(
        wavefront, 
        z_dist_m,
        grid_spacing_object_m,
        wavelength_m
):
    ny, nx = wavefront.shape
    device = wavefront.device
    wavelength = torch.tensor(wavelength_m, dtype=torch.float32, device=device)  # or 'cpu'
    k = 2 * torch.pi / wavelength


    wavefront_ft = torch.fft.fftshift(torch.fft.fft2(wavefront))
    fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=grid_spacing_object_m[0], device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=grid_spacing_object_m[1], device=device))
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")

    # 1. Prevent NaNs: Evanescent waves create negative values inside the sqrt.
    # torch.sqrt on negative real numbers yields NaN, which instantly destroys gradients!
    arg = 1.0 - (wavelength_m*FX)**2 - (wavelength_m*FY)**2
    
    # Clamp arg to >= 0 to keep gradients healthy, then mask evanescent frequencies to 0
    sqrt_arg = torch.sqrt(torch.clamp(arg, min=0.0).to(torch.float32))
    phase = 1.0j * k * z_dist_m * sqrt_arg
    transfer_func = torch.exp(phase)
    
    # Apply mask out-of-place by multiplying
    valid_mask = (arg >= 0).to(transfer_func.dtype)
    transfer_func = transfer_func * valid_mask

    output_field = torch.fft.ifft2(torch.fft.ifftshift(wavefront_ft*transfer_func))

    return output_field


    
# this function is responsible for the fraunhofer propagation of the wavefront from the sample plane to the detector plane.
def fraunhofer_propagate(
    wavefront,
    z_dist,
    grid_spacing_object,
    wavelength,
):
    """Propagate a wavefront using the Fraunhofer (far-field) diffraction. 

    Args: 
        wavefront (torch.Tensor): Complex-valued wavefront to be propagated.
        z_dist (float): propagation distance from object to sample. 
        grid_spacing_object (float or tuple): Spacing (in meters) between points in the wavefront tensor. 

    Returns: 
        torch.Tensor: Complex valued propagated wavefront. 
    """
    
    ny, nx = wavefront.shape
    device = wavefront.device
    wavelength = torch.tensor(wavelength, dtype=torch.float32, device=device)  # or 'cpu'
    k = 2 * torch.pi / wavelength



    
    dy, dx = grid_spacing_object

    # Get the spatial frequencies mapped by the FFT
    fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=dx, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=dy, device=device))

    # Map spatial frequencies to physical coordinates at the detector plane
    x_detector = fx * wavelength * z_dist
    y_detector = fy * wavelength * z_dist
    X_det, Y_det = torch.meshgrid(x_detector, y_detector, indexing="xy")

    # FFT
    wavefront_ft = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(wavefront))
    ) * (dx * dy)

    # The quadratic phase term must use the detector coordinates
    quadratic_phase = torch.exp(1j * k * (X_det**2 + Y_det**2) / (2 * z_dist)) / (
        1j * wavelength * z_dist
    )
    phase_factor = torch.exp(torch.tensor(1j * k * z_dist, device=device))

    detector_wave = wavefront_ft * quadratic_phase * phase_factor

    # Note that the detector wave now has a different grid spacing / size than the input wave front.
    return detector_wave


def crop_wavefront(wavefront, grid_spacing_detector, detector_size):

    n_h = int(detector_size[0] / grid_spacing_detector[0])
    n_w = int(detector_size[1] / grid_spacing_detector[1])
    
    sh = (wavefront.shape[-2] - n_h) // 2
    sw = (wavefront.shape[-1] - n_w) // 2
    return wavefront[..., sh : sh + n_h, sw : sw + n_w]


def capture_diffraction(wavefront, grid_spacing_detector, detector_size):
    cropped_wavefront = crop_wavefront(wavefront, grid_spacing_detector, detector_size)
    return torch.abs(cropped_wavefront) ** 2


def capture_intensity(wavefront, grid_spacing_detector, detector_size):
    return capture_diffraction(wavefront, grid_spacing_detector, detector_size).sum()

def downsample_area_exact(intensity, detector_pixels):
    # intensity: [H, W] real tensor
    H, W = intensity.shape[-2], intensity.shape[-1]
    assert H % detector_pixels == 0 and W % detector_pixels == 0, \
        f"{H}x{W} not divisible by {detector_pixels}"

    sh = H // detector_pixels
    sw = W // detector_pixels

    # reshape into bins and sum
    return intensity.reshape(detector_pixels, sh, detector_pixels, sw).sum(dim=(1, 3))


def capture_diffraction_downsample(
    wavefront,
    grid_spacing_detector,
    detector_size,
    detector_pixels=None,
):
    """
    Returns |cropped_wavefront|^2 downsampled to (detector_pixels, detector_pixels).
    Uses binning (summing) to capture the total intensity per detector pixel.
    """
    cropped = crop_wavefront(wavefront, grid_spacing_detector, detector_size)
    intensity = torch.abs(cropped) ** 2

    intensity_downsampled = functional.interpolate(intensity.unsqueeze(0).unsqueeze(1).cpu(), size=detector_pixels, mode="area")
    
    # Scale by the bin area to convert from average to sum (binning)
    bin_area = (intensity.shape[-2] / detector_pixels[0]) * (intensity.shape[-1] / detector_pixels[1])
    intensity_downsampled = intensity_downsampled * bin_area

    return intensity_downsampled.squeeze().squeeze().to(intensity.device)
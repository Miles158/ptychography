import torch

# illuminate 
# creates a propagating wavefront from a designated source 
# --to be propagates to the sample plane.


# this function is responsible for describing the illumination of the wavefront at the source plane.
def source_illumination(grid_size, grid_spacing):

    x = torch.linspace(-grid_spacing * grid_size / 2, grid_spacing * grid_size / 2, grid_size)
    y = torch.linspace(-grid_spacing * grid_size / 2, grid_spacing * grid_size / 2, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    wavefront = torch.ones_like(X, dtype=torch.complex64)  # uniform plane wavefront. Should be changed. 

    return wavefront, X, Y


def propagate_light(wavefront, z_dist, wavelength):
    k = 2 * torch.pi / wavelength
    # just a phase term; no diffraction considered.
    return wavefront * torch.exp(1j * k * z_dist)



# this function is responsible for describing the mask that the wavefront will pass through before the sample plane. 
def mask(wavefront, X, Y, mask_radius, x_cen, y_cen):
    mask = ((X - x_cen)**2 + (Y - y_cen)**2 <= mask_radius**2).to(torch.complex64)
    return wavefront * mask


# this function is responsible for the fresnel propagation of the wavefront from the mask to the sample plane.
def fresnel_propagate(wavefront, z_dist, grid_spacing, wavelength): 
    nx, ny = wavefront.shape
    device = wavefront.device
    
    # form the spatial frequency bins for the FFT 
    Fx = torch.fft.fftfreq(nx, d=grid_spacing, device=device)
    Fy = torch.fft.fftfreq(ny, d=grid_spacing, device=device)
    Vx, Vy = torch.meshgrid(Fy, Fx, indexing='xy')

    k = 2 * torch.pi / wavelength

    wavefront_ft = torch.fft.fft2(wavefront)
    transfer_func = torch.exp(torch.tensor(1j * k * z_dist, device=device)) * torch.exp(-1j * torch.pi * wavelength * z_dist * (Vx**2 + Vy**2))
    # Fresnel approximation for near-field propagation
    wavefront_transferred = wavefront_ft * transfer_func

    return torch.fft.ifft2(wavefront_transferred)


# this function is responsible for the fraunhofer propagation of the wavefront from the sample plane to the detector plane.
def fraunhofer_propagate(wavefront, z_dist, grid_spacing_object, wavelength, detector_size=None, detector_pixels=None):
    ny, nx = wavefront.shape
    device = wavefront.device
    k = 2 * torch.pi / wavelength

    # Get the spatial frequencies mapped by the FFT
    fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=grid_spacing_object, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=grid_spacing_object, device=device))
    
    # Map spatial frequencies to physical coordinates at the detector plane
    x_detector = fx * wavelength * z_dist
    y_detector = fy * wavelength * z_dist
    X_det, Y_det = torch.meshgrid(x_detector, y_detector, indexing='xy')

    # FFT
    wavefront_ft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(wavefront))) * (grid_spacing_object**2)
    
    # The quadratic phase term must use the detector coordinates
    quadratic_phase = torch.exp(1j * k * (X_det**2 + Y_det**2) / (2 * z_dist)) / (1j * wavelength * z_dist)
    phase_factor = torch.exp(torch.tensor(1j * k * z_dist, device=device))
    
    detector_wave = wavefront_ft * quadratic_phase * phase_factor 


    # Note that the detector wave now has a different grid spacing / size than the input wave front. 
    return detector_wave

def crop_wavefront(wavefront, grid_spacing_detector, detector_size):
    n = int(detector_size / grid_spacing_detector) # number of pixels to capture.
    sh = (wavefront.shape[-2] - n) // 2 # number of pixels to chop off the top / bottom
    sw = (wavefront.shape[-1] - n) // 2 # number of pixels to chop off the left / right
    return wavefront[..., sh:sh+n, sw:sw+n]

def capture_diffraction(wavefront, grid_spacing_detector, detector_size):
    cropped_wavefront = crop_wavefront(wavefront, grid_spacing_detector, detector_size)
    return torch.abs(cropped_wavefront) ** 2

def capture_intensity(wavefront, grid_spacing_detector, detector_size):
    return capture_diffraction(wavefront, grid_spacing_detector, detector_size).mean()


def capture_intensity2(wavefront):
    return torch.abs(wavefront)**2
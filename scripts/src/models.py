

import torch
import torch.nn as nn
from . import propagation as prop

class PtychoModel(nn.Module):
    def __init__(self, grid_size, grid_spacing, wavelength, z_dist, detector_size=None, detector_pixels=None, device='cpu'):
        super().__init__()
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.wavelength = wavelength
        self.z_dist = z_dist
        self.detector_size = detector_size
        self.detector_pixels = detector_pixels
        self.device = device
        
        # The object is what we want to reconstruct. We represent it as learnable parameters.
        # Seperate amplitude and phase to avoid rectangular representation of complex numbers which can be harder to optimize.
        self.object_phase = nn.Parameter(torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device))
        self.object_amplitude = nn.Parameter(torch.ones((grid_size, grid_size), dtype=torch.float32, device=device))

    def set_object(self, object):
        self.object_amplitude.data = torch.abs(object)
        self.object_phase.data = torch.angle(object)

    def forward(self, illumination, pixel_shift_x=0, pixel_shift_y=0):
        """
        Forward pass simulates the physics of the ptychography setup.
        """
        # 1. Construct the complex object
        # Convert phase and amplitude to a complex tensor
        complex_object = self.object_amplitude * torch.exp(1j * self.object_phase)
        complex_object = complex_object.to(torch.complex64)
        
        # 2. Simulate the scan shift & consider masking the illuminating wacefront.
        if pixel_shift_x != 0 or pixel_shift_y != 0:
            shifted_object = torch.roll(complex_object, shifts=(pixel_shift_y, pixel_shift_x), dims=(0, 1))
        else:
            shifted_object = complex_object
            
        # 3. interaction with the object
        exit_wavefront = illumination * shifted_object
        

        # 4. Propagate to detector. Assuming far-field propagation. 
        detector_wave = prop.fraunhofer_propagate(
            wavefront=exit_wavefront,
            z_dist=self.z_dist,
            grid_size=self.grid_size,
            grid_spacing=self.grid_spacing,
            wavelength=self.wavelength,

        )
        
        # 5. Capture the expected intensity at the detector
        predicted_intensity = prop.capture_intensity(detector_wave)
        
        return predicted_intensity
    
class STXM(nn.Module):
    def __init__(self, grid_size, grid_spacing, wavelength, z_dist):
        super().__init__()
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.wavelength = wavelength
        self.z_dist = z_dist
        
        # The object is what we want to reconstruct. We represent it as learnable parameters.
        # Seperate amplitude and phase to avoid rectangular representation of complex numbers which can be harder to optimize.
        self.object_phase = nn.Parameter(torch.zeros((grid_size, grid_size), dtype=torch.float32))
        self.object_amplitude = nn.Parameter(torch.ones((grid_size, grid_size), dtype=torch.float32))

    def set_object(self, object):
        self.object_amplitude.data = torch.abs(object)
        self.object_phase.data = torch.angle(object)

    def forward(self, illumination):
        return 0
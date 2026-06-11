"""
models.py
Written by Miles Houston, 6/11/2026

This file provides Ptychography and STXM Pytorch models useful for simulations.
"""

import torch
import torch.nn as nn
import propagation as prop
import adaptive_optics as AO


class PtychoModel(nn.Module):
    """This model simulates a scanning ptychography setup with a learnable object.
    
    The model stores an objects ampltude and phase as learnable parameters and 
    implements a forward model that 
    1) applies a shift to the object plane
    2) multiplies the object plane by a given illumination
    3) propagates the resulting wave to a detector plane
    4) returns the diffraction pattern on the detector plane.
    """

    def __init__(
        self,
        grid_size_object,
        grid_spacing_object,
        wavelength,
        z_dist,
        detector_size=None,
        device="cpu",
    ):
        """Create the ptychography model."""
        super().__init__()
        self.grid_size_object = grid_size_object
        self.grid_spacing_object = grid_spacing_object
        self.wavelength = wavelength
        self.z_dist = z_dist
        self.detector_size = detector_size
        self.device = device

        self.grid_spacing_detector = (
            wavelength * z_dist / (grid_size_object * grid_spacing_object)
        )  # from Fourier optics scaling

        # The object is what we want to reconstruct. We represent it as learnable parameters.
        # Seperate amplitude and phase to avoid rectangular representation of complex numbers which can be harder to optimize.
        self.object_phase = nn.Parameter(
            torch.zeros(
                (grid_size_object, grid_size_object), dtype=torch.float32, device=device
            )
        )

        self.object_amplitude = nn.Parameter(
            torch.ones(
                (grid_size_object, grid_size_object), dtype=torch.float32, device=device
            )
        )

    def set_object(self, object):
        """Set the model's object with a complex tensor. 

            Args:
                object (torch.Tensor): Complex valued tensor of shape 
                (grid_size_object, grid_size_object).
        """
        self.object_amplitude.data = torch.abs(object)
        self.object_phase.data = torch.angle(object)

    def forward(self, illumination, pixel_shift_x=0, pixel_shift_y=0):
        """Computes the diffraction pattern of the object with a given illumination & shift.

            Args: 
                illumination (torch.Tensor): Complex valued illumination field incident on 
                    object plane. E.g. guassian beam, some constant amplitude/phase wavefront.
                pixel_shift_x (int): Horizontal scanning stage shift.
                pixel_shift_y (int): Vertical scanning stage shift. 

            Returns: 
                torch.Tensor: Real-valued detector intensity / diffraction pattern at the detector plane.
        """

        # 1. Construct the complex object
        # Convert phase and amplitude to a complex tensor
        complex_object = self.object_amplitude * torch.exp(1j * self.object_phase)
        complex_object = complex_object.to(torch.complex64)

        # 2. Simulate the scan shift
        if pixel_shift_x != 0 or pixel_shift_y != 0:
            shifted_object = torch.roll(
                complex_object, shifts=(pixel_shift_y, pixel_shift_x), dims=(0, 1)
            )
        else:
            shifted_object = complex_object

        # 3. interaction with the object
        exit_wavefront = illumination * shifted_object

        # 4. Propagate to detector. Assuming far-field propagation.
        detector_wave = prop.fraunhofer_propagate(
            wavefront=exit_wavefront,
            z_dist=self.z_dist,
            grid_spacing_object=self.grid_spacing_object,
            wavelength=self.wavelength,
        )
        if (self.grid_spacing_detector * detector_wave.size(0)) < (self.detector_size):
            print(
                f"Warning: the calculated diffraction pattern {self.grid_spacing_detector * detector_wave.size(0)} is smaller than the specified detector size."
            )

        # 5. Capture the expected intensity at the detector
        predicted_intensity = prop.capture_diffraction(
            wavefront=detector_wave,
            grid_spacing_detector=self.grid_spacing_detector,
            detector_size=self.detector_size,
        )

        return predicted_intensity


class DiptychModel(nn.Module):
    """This model simulates a scanning ptychography setup with a learnable object.
    
    The model stores an objects ampltude and phase as learnable parameters and 
    implements a forward model that 
    1) applies a shift to the object plane
    2) multiplies the object plane by a given illumination
    3) propagates the resulting wave to a detector plane
    4) returns the diffraction pattern on the detector plane.
    """

    def __init__(
        self,
        grid_size_object,
        grid_spacing_object,
        wavelength,
        z_dist,
        detector_size=None,
        device="cpu",
    ):
        """Create the ptychography model."""
        super().__init__()
        self.grid_size_object = grid_size_object
        self.grid_spacing_object = grid_spacing_object
        self.wavelength = wavelength
        self.z_dist = z_dist
        self.detector_size = detector_size
        self.device = device

        self.grid_spacing_detector = (
            wavelength * z_dist / (grid_size_object * grid_spacing_object)
        )  # from Fourier optics scaling

        # The object is what we want to reconstruct. We represent it as learnable parameters.
        # Seperate amplitude and phase to avoid rectangular representation of complex numbers which can be harder to optimize.
        self.object_phase = nn.Parameter(
            torch.zeros(
                (grid_size_object, grid_size_object), dtype=torch.float32, device=device
            )
        )

        self.object_amplitude = nn.Parameter(
            torch.ones(
                (grid_size_object, grid_size_object), dtype=torch.float32, device=device
            )
        )

    def set_object(self, object):
        """Set the model's object with a complex tensor. 

            Args:
                object (torch.Tensor): Complex valued tensor of shape 
                (grid_size_object, grid_size_object).
        """
        self.object_amplitude.data = torch.abs(object)
        self.object_phase.data = torch.angle(object)

    def forward(self, illumination, actuator_commands, mirror_instance):
        """Computes the diffraction pattern of the object with a given illumination & shift.

            Args: 
                illumination (torch.Tensor): Complex valued illumination field incident on 
                    object plane. E.g. guassian beam, some constant amplitude/phase wavefront.
                pixel_shift_x (int): Horizontal scanning stage shift.
                pixel_shift_y (int): Vertical scanning stage shift. 

            Returns: 
                torch.Tensor: Real-valued detector intensity / diffraction pattern at the detector plane.
        """

        # 1. Construct the complex object
        # Convert phase and amplitude to a complex tensor
        complex_object = self.object_amplitude * torch.exp(1j * self.object_phase)
        complex_object = complex_object.to(torch.complex64)

        # 2. act on the wavefront with the deformable mirror


        mirror_heights = mirror_instance.deformableMirror(actuator_commands)  # meters

        phase = (4.0 * torch.pi / self.wavelength) * mirror_heights  # rad

        mirror_phase_factor = torch.exp(1j * phase).to(torch.complex64)

        reflected_illumination = illumination.to(torch.complex64) * mirror_phase_factor


        # 2. act on the wavefront with the deformable mirror

        # 3. interaction with the object
        exit_wavefront = reflected_illumination * object

        # 4. Propagate to detector. Assuming far-field propagation.
        detector_wave = prop.fraunhofer_propagate(
            wavefront=exit_wavefront,
            z_dist=self.z_dist,
            grid_spacing_object=self.grid_spacing_object,
            wavelength=self.wavelength,
        )
        if (self.grid_spacing_detector * detector_wave.size(0)) < (self.detector_size):
            print(
                f"Warning: the calculated diffraction pattern {self.grid_spacing_detector * detector_wave.size(0)} is smaller than the specified detector size."
            )

        # 5. Capture the expected intensity at the detector
        predicted_intensity = prop.capture_diffraction(
            wavefront=detector_wave,
            grid_spacing_detector=self.grid_spacing_detector,
            detector_size=self.detector_size,
        )

        return predicted_intensity


class STXM(nn.Module):
    def __init__(
        self,
        grid_size_object,
        grid_spacing_object,
        wavelength,
        z_dist,
        detector_size=None,
        device="cpu",
    ):
        super().__init__()
        self.grid_size_object = grid_size_object
        self.grid_spacing_object = grid_spacing_object
        self.wavelength = wavelength
        self.z_dist = z_dist
        self.detector_size = detector_size
        self.device = device

        self.grid_spacing_detector = (
            wavelength * z_dist / (grid_size_object * grid_spacing_object)
        )  # from Fourier optics scaling

        # The object is what we want to reconstruct. We represent it as learnable parameters.
        # I find rectangular representation of complex numbers can be harder to optimize, so we seperate amplitude and phase.
        self.object_phase = nn.Parameter(
            torch.zeros(
                (self.grid_size_object, self.grid_size_object),
                dtype=torch.float32,
                device=device,
            )
        )
        self.object_amplitude = nn.Parameter(
            torch.ones(
                (self.grid_size_object, self.grid_size_object),
                dtype=torch.float32,
                device=device,
            )
        )

    def set_object(self, object):
        # Zero-pad the object if it is smaller than the model's grid size
        h, w = object.shape
        pad_h = max(0, (self.grid_size_object - h) // 2)
        pad_w = max(0, (self.grid_size_object - w) // 2)
        if pad_h > 0 or pad_w > 0:
            object = torch.nn.functional.pad(
                object, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0
            )

        self.object_amplitude.data = torch.abs(object)
        self.object_phase.data = torch.angle(object)

    def forward(self, illumination, pixel_shift_x=0, pixel_shift_y=0):
        """
        Forward pass simulates the physics of the STXM setup.
        """
        # 1. Construct the complex object
        complex_object = (self.object_amplitude * torch.exp(1j * self.object_phase)).to(
            torch.complex64
        )

        # 2. Simulate the scan shift
        if pixel_shift_x != 0 or pixel_shift_y != 0:
            shifted_object = torch.roll(
                complex_object, shifts=(-pixel_shift_y, -pixel_shift_x), dims=(0, 1)
            )
        else:
            shifted_object = complex_object

        # 3. interaction with the object
        exit_wavefront = illumination * shifted_object

        # 4. Propagate to detector. Assuming far-field propagation.
        detector_wave = prop.fraunhofer_propagate(
            wavefront=exit_wavefront,
            z_dist=self.z_dist,
            grid_spacing_object=self.grid_spacing_object,
            wavelength=self.wavelength,
        )

        if (self.grid_spacing_detector * detector_wave.size(0)) < (self.detector_size):
            print(
                f"Warning: the calculated diffraction pattern {self.grid_spacing_detector * detector_wave.size(0)} is smaller than the specified detector size."
            )

        # 5. Capture the expected intensity at the detector
        predicted_intensity = prop.capture_intensity(
            wavefront=detector_wave,
            grid_spacing_detector=self.grid_spacing_detector,
            detector_size=self.detector_size,
        )

        return predicted_intensity

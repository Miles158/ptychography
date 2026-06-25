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
    """This model simulates a scanning ptychography setup with a learnable object,
    learnable z_dist, and a learnable probe.
    """

    def __init__(
        self,
        object_grid_shape,
        grid_spacing_object,
        wavelength,
        z_dist_init,
        probe_init,
        detector_shape_m,
        detector_shape_px,
        device="cpu",
    ):
        """Create the learnable ptychography model."""
        super().__init__()
        self.object_grid_shape = object_grid_shape
        self.grid_spacing_object = grid_spacing_object
        self.wavelength = wavelength
        self.detector_shape_m = detector_shape_m
        self.detector_shape_px = detector_shape_px
        self.device = device
        # Learnable parameters
        self.z_dist = nn.Parameter(torch.tensor(z_dist_init, dtype=torch.float32, device=device))
        
        # Probe as learnable complex parameter (amplitude and phase)
        self.probe_amplitude = nn.Parameter(torch.abs(probe_init))
        self.probe_phase = nn.Parameter(torch.angle(probe_init))

        self.object_phase = nn.Parameter(
            torch.zeros(
                (object_grid_shape[0], object_grid_shape[1]), dtype=torch.float32, device=device
            )
        )
        self.object_amplitude = nn.Parameter(
            torch.ones(
               (object_grid_shape[0], object_grid_shape[1]), dtype=torch.float32, device=device
            )
        )

    def set_object(self, object):
         """Set the model's object with a complex tensor. 

             Args:
                 object (torch.Tensor): Complex valued tensor of shape 
                 (object_grid_shape[0], object_grid_shape[1]).
         """
         self.object_amplitude.data = torch.abs(object)
         self.object_phase.data = torch.angle(object)

    def forward(self, pixel_shift_x=0, pixel_shift_y=0):
        # 1. Construct complex object and probe
        complex_object = self.object_amplitude * torch.exp(1j * self.object_phase)
        probe = self.probe_amplitude * torch.exp(1j * self.probe_phase)

        # 2. Simulate the scan shift (Non-periodic)
        if pixel_shift_x != 0 or pixel_shift_y != 0:
            # Pad the object to allow shifting without wrap-around
            # Pad by the maximum possible shift amount
            pad_x = abs(pixel_shift_x)
            pad_y = abs(pixel_shift_y)
            
            # Pad with zeros
            padded_object = torch.nn.functional.pad(
                complex_object.unsqueeze(0).unsqueeze(0), 
                (pad_x, pad_x, pad_y, pad_y), 
                mode='constant', 
                value=0.0
            ).squeeze()
            
            # Shift the padded object
            # Note: adjust indices based on padding
            shifted_object = padded_object[
                pad_y - pixel_shift_y : pad_y - pixel_shift_y + self.object_grid_shape[0],
                pad_x - pixel_shift_x : pad_x - pixel_shift_x + self.object_grid_shape[1]
            ]
        else:
            shifted_object = complex_object
        # ... rest of the forward pass ...

        # # 2. Simulate the scan shift
        # if pixel_shift_x != 0 or pixel_shift_y != 0:
        #     shifted_object = torch.roll(
        #         complex_object, shifts=(pixel_shift_y, pixel_shift_x), dims=(0, 1)
        #     )
        # else:
        #     shifted_object = complex_object

        # 3. interaction with the object
        exit_wavefront = probe * shifted_object

        # 4. Propagate to detector using learnable z_dist
        detector_wave = prop.fresnel_propagate(
            wavefront=exit_wavefront,
            z_dist=self.z_dist,
            grid_spacing=self.grid_spacing_object,
            wavelength=self.wavelength,
        )

        # 5. Capture the expected intensity
        # In Fresnel (near-field) propagation, the grid spacing is preserved.
        # The detector wave has the exact same physical pixel size as the object plane.
        grid_spacing_detector = self.grid_spacing_object
        
        predicted_intensity = prop.capture_diffraction_downsample(
            wavefront=detector_wave,
            grid_spacing_detector=grid_spacing_detector,
            detector_size=self.detector_shape_m,
            detector_pixels=self.detector_shape_px
        )

        return predicted_intensity




# class PtychoModel(nn.Module):
#     """This model simulates a scanning ptychography setup with a learnable object.
    
#     The model stores an objects ampltude and phase as learnable parameters and 
#     implements a forward model that 
#     1) applies a shift to the object plane
#     2) multiplies the object plane by a given illumination
#     3) propagates the resulting wave to a detector plane
#     4) returns the diffraction pattern on the detector plane.
#     """

#     def __init__(
#         self,
#         object_grid_shape,
#         grid_spacing_object,
#         wavelength,
#         z_dist,
#         detector_shape_m,
#         detector_shape_px,
#         device="cpu",
#     ):
#         """Create the ptychography model."""
#         super().__init__()
#         self.object_grid_shape = object_grid_shape
#         self.grid_spacing_object = grid_spacing_object
#         self.wavelength = wavelength
#         self.z_dist = z_dist
#         self.detector_shape_m = detector_shape_m
#         self.detector_shape_px = detector_shape_px
#         self.device = device

#         # The object is what we want to reconstruct. We represent it as learnable parameters.
#         # Seperate amplitude and phase to avoid rectangular representation of complex numbers which can be harder to optimize.
#         self.object_phase = nn.Parameter(
#             torch.zeros(
#                 object_grid_shape, dtype=torch.float32, device=device
#             )
#         )

#         self.object_amplitude = nn.Parameter(
#             torch.ones(
#                 object_grid_shape, dtype=torch.float32, device=device
#             )
#         )

#     def set_object(self, object):
#         """Set the model's object with a complex tensor. 

#             Args:
#                 object (torch.Tensor): Complex valued tensor of shape 
#                 (object_grid_shape[0], object_grid_shape[1]).
#         """
#         self.object_amplitude.data = torch.abs(object)
#         self.object_phase.data = torch.angle(object)

#     def forward(self, illumination, pixel_shift_x=0, pixel_shift_y=0):
#         """Computes the diffraction pattern of the object with a given illumination & shift.

#             Args: 
#                 illumination (torch.Tensor): Complex valued illumination field incident on 
#                     object plane. E.g. guassian beam, some constant amplitude/phase wavefront.
#                 pixel_shift_x (int): Horizontal scanning stage shift.
#                 pixel_shift_y (int): Vertical scanning stage shift. 

#             Returns: 
#                 torch.Tensor: Real-valued detector intensity / diffraction pattern at the detector plane.
#         """

#         # 1. Construct the complex object
#         # Convert phase and amplitude to a complex tensor
#         complex_object = self.object_amplitude * torch.exp(1j * self.object_phase)
#         complex_object = complex_object.to(torch.complex64)

#         # 2. Simulate the scan shift
#         if pixel_shift_x != 0 or pixel_shift_y != 0:
#             shifted_object = torch.roll(
#                 complex_object, shifts=(pixel_shift_y, pixel_shift_x), dims=(0, 1)
#             )
#         else:
#             shifted_object = complex_object

#         # 3. interaction with the object
#         exit_wavefront = illumination * shifted_object

#         # 4. Propagate to detector. Assuming near-field propagation.
#         detector_wave = prop.fraunhofer_propagate(
#             wavefront=exit_wavefront,
#             z_dist=self.z_dist,
#             grid_spacing_object=self.grid_spacing_object,
#             wavelength=self.wavelength,
#         )
        
#         grid_spacing_detector = (
#             self.wavelength * self.z_dist / (self.object_grid_shape[0] * self.grid_spacing_object[0]),
#             self.wavelength * self.z_dist / (self.object_grid_shape[1] * self.grid_spacing_object[1])
#         )

#         # 5. Capture the expected intensity at the detector
#         predicted_intensity = prop.capture_diffraction_downsample(
#             wavefront=detector_wave,
#             grid_spacing_detector=grid_spacing_detector,
#             detector_size=self.detector_shape_m,
#             detector_pixels= self.detector_shape_px
#         )

#         return predicted_intensity


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
        object_grid_shape,
        grid_spacing_object,
        wavelength,
        z_dist,
        detector_shape_m=None,
        detector_shape_px=None,
        device="cpu",
    ):
        """Create the ptychography model."""
        super().__init__()
        self.object_grid_shape = object_grid_shape
        self.grid_spacing_object = grid_spacing_object
        self.wavelength = wavelength
        self.z_dist = z_dist
        self.detector_shape_m = detector_shape_m
        self.detector_shape_px = detector_shape_px
        self.device = device

        # The object is what we want to reconstruct. We represent it as learnable parameters.
        # Seperate amplitude and phase to avoid rectangular representation of complex numbers which can be harder to optimize.
        self.object_phase = nn.Parameter(
            torch.zeros(
                object_grid_shape, dtype=torch.float32, device=device
            )
        )

        self.object_amplitude = nn.Parameter(
            torch.ones(
                object_grid_shape, dtype=torch.float32, device=device
            )
        )

    def set_object(self, object):
        """Set the model's object with a complex tensor. 

            Args:
                object (torch.Tensor): Complex valued tensor of shape 
                (object_grid_shape[0], object_grid_shape[1]).
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

        mirror_response = mirror_instance.forward(actuator_commands)  # meters

        reflected_illumination = illumination.to(torch.complex64) * mirror_response

        # 3. interaction with the object
        exit_wavefront = reflected_illumination * complex_object

        # 4. Propagate to detector. Assuming near-field propagation.
        detector_wave = prop.fraunhofer_propagate(
            wavefront=exit_wavefront,
            z_dist=self.z_dist,
            grid_spacing_object=self.grid_spacing_object,
            wavelength=self.wavelength,
        )
        
        grid_spacing_detector = (
            self.wavelength * self.z_dist / (self.object_grid_shape[0] * self.grid_spacing_object),
            self.wavelength * self.z_dist / (self.object_grid_shape[1] * self.grid_spacing_object)
        )

        predicted_intensity = prop.capture_diffraction_downsample(
            wavefront=detector_wave,
            grid_spacing_detector=grid_spacing_detector,
            detector_size=self.detector_shape_m,
            detector_pixels=self.detector_shape_px
        )


        return predicted_intensity


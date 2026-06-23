
import torch
import torch.nn as nn
import propagation as prop
import adaptive_optics as AO


class LearnablePtychoModel(nn.Module):
    """This model simulates a scanning ptychography setup with a learnable object,
    learnable z_dist, and a learnable probe.
    """

    def __init__(
        self,
        grid_size_object,
        grid_spacing_object,
        wavelength,
        z_dist_init,
        probe_init,
        detector_size,
        detector_pixels,
        device="cpu",
    ):
        """Create the learnable ptychography model."""
        super().__init__()
        self.grid_size_object = grid_size_object
        self.grid_spacing_object = grid_spacing_object
        self.wavelength = wavelength
        self.detector_size = detector_size
        self.detector_pixels = detector_pixels
        self.device = device

        # Learnable parameters
        self.z_dist = nn.Parameter(torch.tensor(z_dist_init, dtype=torch.float32, device=device))
        
        # Probe as learnable complex parameter (amplitude and phase)
        self.probe_amplitude = nn.Parameter(torch.abs(probe_init))
        self.probe_phase = nn.Parameter(torch.angle(probe_init))

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

    def forward(self, pixel_shift_x=0, pixel_shift_y=0):
        # 1. Construct complex object and probe
        complex_object = self.object_amplitude * torch.exp(1j * self.object_phase)
        probe = self.probe_amplitude * torch.exp(1j * self.probe_phase)

        # 2. Simulate the scan shift
        if pixel_shift_x != 0 or pixel_shift_y != 0:
            shifted_object = torch.roll(
                complex_object, shifts=(pixel_shift_y, pixel_shift_x), dims=(0, 1)
            )
        else:
            shifted_object = complex_object

        # 3. interaction with the object
        exit_wavefront = probe * shifted_object

        # 4. Propagate to detector using learnable z_dist
        detector_wave = prop.fraunhofer_propagate(
            wavefront=exit_wavefront,
            z_dist=self.z_dist,
            grid_spacing_object=self.grid_spacing_object,
            wavelength=self.wavelength,
        )

        # 5. Capture the expected intensity
        predicted_intensity = prop.capture_diffraction_downsample(
            wavefront=detector_wave,
            grid_spacing_detector=(self.wavelength * self.z_dist / (self.grid_size_object * self.grid_spacing_object)),
            detector_size=self.detector_size,
            detector_pixels=self.detector_pixels
        )

        return predicted_intensity
import torch

class deformableMirror(torch.nn.Module):
    def __init__(self,
            wavelength,
            number_actuators,
            max_stroke,
            mirror_radius,
            grid_size_wavefront,
            grid_spacing_wavefront,
            device='cpu'
    ):
        super().__init__()
        self.wavelength = wavelength
        self.number_actuators = number_actuators
        self.max_stroke = max_stroke
        self.mirror_radius = mirror_radius
        self.grid_size_wavefront = grid_size_wavefront
        self.grid_spacing_wavefront = grid_spacing_wavefront
        self.device = device
        
        # Coordinate grid for the wavefront
        extent = self.grid_size_wavefront * self.grid_spacing_wavefront
        x = torch.linspace(-extent/2, extent/2, self.grid_size_wavefront, device=self.device)
        y = torch.linspace(-extent/2, extent/2, self.grid_size_wavefront, device=self.device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='xy')
        
        self.actuator_coords = self.generate_fibbonacci_disk()
        
        self.actuator_coordinates = self.generate_fibbonacci_disk()

        # The state of each actuator (from -1 to 1, normalized units)
        self.actuator_position = torch.zeros(self.number_actuators, device=self.device)

    def influence_function(self, x_center, y_center, sigma=None):
        # A simple Gaussian influence function. Sigma controls the width.
        if sigma is None:
            # Scale sigma based on average distance between actuators
            sigma = self.mirror_radius / torch.sqrt(torch.tensor(self.number_actuators, dtype=torch.float32))
            
        return torch.exp(-((self.X - x_center)**2 + (self.Y - y_center)**2) / (2 * sigma**2))

    def generate_fibbonacci_disk(self):
        """Generates a uniform-ish distribution of points on the mirror disk."""
        golden_ratio = (1 + 5**0.5) / 2
        i = torch.arange(0, self.number_actuators, device=self.device)
        
        theta = 2 * torch.pi * i / golden_ratio
        
        r = self.mirror_radius * torch.sqrt((i + 0.5) / self.number_actuators)

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        # List of the placement of each actuator. 
        return torch.stack((x, y), dim=1)

    def forward(self, actuator_commands):
        """
        Takes a list of actuator commands and returns the mirrors response in the phase change 
        """
        self.actuator_commands=actuator_commands
            
        # matrix that represents the positional change of each point ont he mirror surface. 
        mirror_surface = torch.zeros_like(self.X, device=self.device)
        
        # generate the response of each actuator and add them linearly
        for i in range(self.number_actuators):
            actuator_command = self.actuator_commands[i] * self.max_stroke
            x_actuator_cen, y_actuator_cen = self.actuator_coords[i]
            
            # Add this actuator's contribution
            mirror_surface += actuator_command * self.influence_function(x_actuator_cen, y_actuator_cen)

        # 3. Convert the physical response to phase response and clip the mirror. 

        

        mirror_surface_phase = (4.0 * torch.pi / self.wavelength) * mirror_surface  # rad

        mirror_complex_phase = torch.exp(1j * mirror_surface_phase).to(torch.complex64)

        aperture_mask = (self.X**2 + self.Y**2) <= self.mirror_radius**2

        mirror_response = mirror_complex_phase * aperture_mask

        return mirror_response



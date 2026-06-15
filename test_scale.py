import torch
import numpy as np
import sys
sys.path.append('.')
from src import models as models 
from src import object as obj
from src import adaptive_optics as ao

device = "mps"
grid_size = 1024
object_extent = 75e-3
grid_spacing = object_extent / grid_size
wavelength = 500e-9
z_dist = 250e-3
detector_size = 1.5e-3
detector_pixels = 512

x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

gt_object = torch.ones((grid_size, grid_size), dtype=torch.complex64, device=device)
gt_model = models.PtychoModel(grid_size, grid_spacing, wavelength, z_dist, device=device, detector_size=detector_size, detector_pixels=detector_pixels).to(device)
gt_model.set_object(gt_object)

probe_radius = 15e-3
probe = (torch.sqrt(X**2 + Y**2) <= probe_radius).to(torch.complex64)
intensity = gt_model(probe, 0, 0)
print(f"Max intensity: {intensity.max().item():.2e}")
print(f"Mean intensity: {intensity.mean().item():.2e}")

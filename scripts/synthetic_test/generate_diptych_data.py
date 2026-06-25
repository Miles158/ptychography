import torch
import numpy as np
import os

# Ensure imports work regardless of where the script is run
import models as models 
import object as obj
import adaptive_optics as ao
import helpers as helpers

device = "mps" 

# --- 1. Setup Simulation Parameters ---

# General Parameters
wavelength = 500e-9
z_dist = 250e-3 #
grid_size = 1024
object_extent = 32e-3 #
object_fraction = 0.75 # the object takes up a smaller part of the entire object. 
grid_spacing = object_extent / grid_size
probe_radius = 15e-3 

source_intensity = 1000 # Units?? 

# Detector Parameters
detector_size = 1.5e-3 # 
detector_pixels = 1024

# Mirror Parameters
mirror_radius = probe_radius
actuator_max_stroke = 500e-9
actuator_jitter = 0.0 # careful with this, its in the normalized actuator commands [-1,1]
num_actuators = 24

# Optimization
num_measurements = 30

# Setup coordinate grid
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

# make the object 
core_grid_size = int(grid_size * object_fraction)
gt_object = obj.make_text_mask("ALS", core_grid_size, margin=0.01, stroke_frac=0.01, font_name="Helvetica", device=device)
gt_object = obj.make_siemens_star_mask(core_grid_size, num_sectors=32, margin=0.01, line_fraction=1.0, center_radius_frac=0.0, device=device)

# Zero-pad the object up to the full simulation grid size
pad_pixels = (grid_size - core_grid_size) // 2
gt_object = torch.nn.functional.pad(gt_object, (pad_pixels, pad_pixels, pad_pixels, pad_pixels), mode='constant', value=0.0)

# The beam is now the same size as the sample (or at least covers it entirely)
probe_distance = torch.sqrt(X**2 + Y**2)
probe_amplitude_mask = (probe_distance <= probe_radius).to(torch.float32)

# Set up the deformable mirror
mirror = ao.deformableMirror(
    wavelength=wavelength,
    number_actuators=num_actuators,
    max_stroke=actuator_max_stroke,
    mirror_radius=mirror_radius,
    grid_size_wavefront=grid_size,
    grid_spacing_wavefront=grid_spacing,
    device=device
)

# Generate synthetic measurements by applying random commands to the DM instead of shifting the sample
print(f"Generating synthetic data for {num_measurements} random DM configurations...")

# We use the true physics to generate our "experimental" data
gt_model = models.DiptychModel(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device,
    detector_size=detector_size,
    detector_pixels=detector_pixels
).to(device)

gt_model.set_object(gt_object)

# Collect measurements
raw_intensities = []
dm_commands = []

with torch.no_grad():
    for i in range(num_measurements):
        # Generate random actuator commands [-1, 1]
        commands = (torch.rand(num_actuators, device=device) * 2) - 1
        dm_commands.append(commands)
        dm_commands_error = helpers.add_actuator_jitter(commands, actuator_jitter)
        
        # Keep illumination in arbitrary units (amplitude of 1 inside the probe)
        illumination = probe_amplitude_mask.to(torch.complex64)
        
        # No shift (0,0) - diptych relies entirely on the probe phase variations from DM
        detector_intensity = gt_model(illumination, dm_commands_error, mirror)
        raw_intensities.append(detector_intensity)

# Find global max across all measurements
raw_intensities = torch.stack(raw_intensities)
global_max_intensity = raw_intensities.max()

# Scale all intensities such that the maximum pixel across all measurements matches source_intensity
scale_factor = source_intensity / global_max_intensity

measurements = []
with torch.no_grad():
    for raw_int in raw_intensities:
        scaled_intensity = raw_int * scale_factor
        # Add Poisson shot noise naturally on the photon counts
        noisy_photons = helpers.add_shot_noise(scaled_intensity)
        measurements.append(noisy_photons)
        
measurements = torch.stack(measurements)
dm_commands = torch.stack(dm_commands)

print(f"Global max before scaling: {global_max_intensity:.4e}")
print(f"Scaling factor (scale_factor): {scale_factor:.4e}")

# Save the generated data to disk
save_path = os.path.join(os.path.dirname(__file__), f'../data/synthetic/diptych_001_int:{source_intensity}_photos:{num_measurements}.pt')
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save({
    "measurements": measurements.cpu(),
    "dm_commands": dm_commands.cpu(),
    "gt_object": gt_object.cpu(),
    "probe_amplitude_mask": probe_amplitude_mask.cpu(),
    "scale_factor": scale_factor.cpu() if isinstance(scale_factor, torch.Tensor) else scale_factor,
    "measure_max_photon": source_intensity,
    "wavelength_m": wavelength,
    "z_dist_m": z_dist,
    "grid_size": grid_size,
    "object_extent_m": object_extent,
    "grid_spacing_m": grid_spacing,
    "detector_size_m": detector_size,
    "detector_pixels": detector_pixels,
    "mirror_radius_m": mirror_radius,
    "actuator_max_stroke_m": actuator_max_stroke,
    "num_actuators": num_actuators
}, save_path)

print(f"Dataset successfully saved to {save_path}")

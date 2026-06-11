# this script is used for testing the models in src/models.py.
# It does the following:
# 1. Creates a sample object with some known amplitude and phase features. 
# 2. Initializes the ptychography model with a simple object (e.g., a circular aperture) and simulates the forward pass to generate predicted intensity patterns at the detector.
# 3. Visualizes the predicted intensity patterns to verify that the model is simulating the expected physics (e.g., diffraction patterns from the aperture). 
import torch
import matplotlib.pyplot as plt
from src import models as models

wavelength = 500e-9
object_type = 'circular_aperture'
object_extent = 25e-3 
# grid size should be determiend with the pixel pitch of the detector. grid_spacing_det = wavelength * z_dist / (grid_size * grid_spacing_object). 
grid_size = 4096
grid_spacing = object_extent / grid_size
# considerations of the detector plane having enough resolution to capture the diffraction pattern of the object? 
# oversampling condition is pixel_size < lambda * z / 2D, where D is the size of the object. 



object = torch.zeros((grid_size, grid_size), dtype=torch.complex64)
x = torch.linspace(-object_extent/2, object_extent/2, grid_size)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size)
X, Y = torch.meshgrid(x, y, indexing='xy')




if object_type == 'circular_aperture':
    radius = 3e-3 / 10
    aperture = (X**2 + Y**2 <= radius**2).to(torch.complex64)
    object += aperture


plt.figure(figsize=(6, 6))
plt.imshow(torch.abs(object).cpu().numpy(),
           extent=(-object_extent/2, object_extent/2, -object_extent/2, object_extent/2), origin='lower')
plt.title('Object Amplitude')
plt.colorbar(label='Amplitude')
plt.savefig('object_amplitude.png')

z_dist = 250e-3

# simulate the forward pass of the ptychography model


model = models.PtychoModel(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=500e-9, 
    z_dist=z_dist,
    detector_size=1.5e-2,  
) 


model.set_object(object)
predicted_intensity = model.forward(illumination=torch.ones_like(object), pixel_shift_x=0, pixel_shift_y=0)


N = grid_size
dx = grid_spacing
lam = 500e-9
z = z_dist

detector_size = model.grid_spacing_detector * grid_size
print(f"Detector size: {detector_size:.2e} m")
plt.figure(figsize=(6, 6))
extent=(-detector_size/2, detector_size/2, -detector_size/2, detector_size/2)
# plt.imshow(predicted_intensity.detach().cpu().numpy(),
#            extent=extent, origin='lower', norm=plt.matplotlib.colors.LogNorm())

plt.imshow(predicted_intensity.detach().cpu().numpy(),
           extent=extent, origin='lower')
plt.clim(1e-6, predicted_intensity.max().item()/2)  # Set limits for better visualization of the diffraction pattern

# plt.title('Predicted Intensity at Detector')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.colorbar(label='Intensity')
plt.savefig('predicted_intensity.png')



# ####################################################################################################################################
# from scipy.special import j1

# # circular aperture geometry
# diameter = 2 * radius          # diameter

# # --- detector coordinate axis consistent with your FFT mapping ---
# fx = torch.fft.fftshift(torch.fft.fftfreq(grid_size, d=grid_spacing))  # spatial freq [cycles/m]
# x_detector = (fx * wavelength * z_dist).cpu().numpy()                      # detector x-coordinates [m]

# # --- theoretical Airy pattern (normalized) ---
# slice = np.abs(x_detector)  
# u = np.pi * diameter * slice / (wavelength * z_dist)

# airy = np.ones_like(u, dtype=np.float64)
# mask = u != 0
# airy[mask] = (2 * j1(u[mask]) / u[mask])**2

# # --- compare to your simulation center line ---
# sim = predicted_intensity.detach().cpu().numpy()
# center_line = sim[grid_size // 2, :]

# center_line_n = center_line / center_line.max()
# airy_n = airy / airy.max()

# plt.figure()
# plt.plot(x_detector * 1e3, center_line_n, label="Sim (center row)")
# plt.plot(x_detector * 1e3, airy_n, '--', label="Theory Airy")
# plt.xlabel("x_det (mm)")
# plt.ylabel("Normalized intensity")
# plt.legend()
# plt.savefig("airy_compare.png")
# ####################################################################################################################################
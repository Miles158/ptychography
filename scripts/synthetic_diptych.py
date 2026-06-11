import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import src.models as models
import src.object as obj
import src.adaptive_optics as ao

# Set device to MPS if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Setup Simulation Parameters ---
grid_size = 1024
object_extent = 75e-3 #
grid_spacing = object_extent / grid_size
wavelength = 500e-9
z_dist = 250e-3 #
detector_size = 1.5e-3 # 

# Setup coordinate grid
x = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
y = torch.linspace(-object_extent/2, object_extent/2, grid_size, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

# --- usage ---
core_grid_size = 256
gt_object = obj.make_text_mask("A", core_grid_size, margin=0.01, stroke_frac=0.01, font_name="Helvetica", device=device)

# Zero-pad the object up to the full simulation grid size
pad_pixels = (grid_size - core_grid_size) // 2
gt_object = torch.nn.functional.pad(gt_object, (pad_pixels, pad_pixels, pad_pixels, pad_pixels), mode='constant', value=0.0)

# The beam is now the same size as the sample (or at least covers it entirely)
probe_radius = 15e-3 # Large probe
probe_distance = torch.sqrt(X**2 + Y**2)
phi0 = 0.0
probe_amplitude_mask = (probe_distance <= probe_radius).to(torch.float32)

# Set up the deformable mirror
num_actuators = 16
mirror = ao.deformableMirror(
    number_actuators=num_actuators,
    max_stroke=250e-9,
    mirror_radius=probe_radius,
    grid_size_wavefront=grid_size,
    grid_spacing_wavefront=grid_spacing,
    device=device
)

# Generate synthetic measurements by applying random commands to the DM instead of shifting the sample
num_measurements = 36
print(f"Generating synthetic data for {num_measurements} random DM configurations...")

# We use the true physics to generate our "experimental" data
gt_model = models.PtychoModel(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device,
    detector_size=detector_size
).to(device)
gt_model.set_object(gt_object)

# Collect measurements
measurements = []
dm_commands = []

with torch.no_grad():
    for i in range(num_measurements):
        # Generate random actuator commands [-1, 1]
        commands = (torch.rand(num_actuators, device=device) * 2) - 1
        dm_commands.append(commands)
        
        # Calculate the phase delay caused by the DM surface (reflection doubles the path length change)
        dm_surface = mirror.forward(commands)
        dm_phase = torch.exp(1j * (4 * torch.pi / wavelength) * dm_surface)
        
        # Combine base probe amplitude with DM phase
        probe = (probe_amplitude_mask.to(torch.complex64) * dm_phase)
        
        # No shift (0,0) - diptych relies entirely on the probe phase variations
        intensity = gt_model(probe, pixel_shift_x=0, pixel_shift_y=0)
        measurements.append(intensity)
        
measurements = torch.stack(measurements)
dm_commands = torch.stack(dm_commands)

print("Measurements generated. Starting optimization...")

# --- 2. Initialize the Reconstruction Model ---
recon_model = models.PtychoModel(
    grid_size_object=grid_size, 
    grid_spacing_object=grid_spacing, 
    wavelength=wavelength, 
    z_dist=z_dist,
    device=device,
    detector_size=detector_size
).to(device)

recon_model.object_amplitude.data = torch.ones((grid_size, grid_size), dtype=torch.float32, device=device)
recon_model.object_phase.data = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)

# --- 3. Setup Optimizer ---
optimizer = optim.AdamW(recon_model.parameters(), lr=0.05)
criterion = nn.MSELoss()

num_epochs = 50
loss_history = []

# --- 4. Optimization Loop ---
for epoch in range(num_epochs):
    epoch_loss = 0
    
    # Process each DM configuration
    for idx in range(num_measurements):
        optimizer.zero_grad()
        
        # Reconstruct the probe for this specific measurement using the known DM commands
        with torch.no_grad():
            dm_surface = mirror.forward(dm_commands[idx])
            dm_phase = torch.exp(1j * (4 * torch.pi / wavelength) * dm_surface)
            probe = (probe_amplitude_mask.to(torch.complex64) * dm_phase)
        
        # Forward pass (no shift)
        predicted_intensity = recon_model(probe, pixel_shift_x=0, pixel_shift_y=0)
        
        # Calculate loss against the measurement
        loss = criterion(predicted_intensity, measurements[idx])
        epoch_loss += loss.item()
        
        # Backward pass and optimize per position
        loss.backward()
        optimizer.step()
        
        # Apply constraints to the amplitude
        with torch.no_grad():
            recon_model.object_amplitude.data = torch.clamp(recon_model.object_amplitude.data, 0, 1)
            
    loss_val = epoch_loss / num_measurements
    loss_history.append(loss_val)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.4e}")

print("Optimization complete.")

# --- 5. Visualization ---
plt.figure(figsize=(15, 10))

# Plot 1: Loss History
plt.subplot(2, 2, 1)
plt.plot(loss_history)
plt.title('Optimization Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')

# Ground truth extent for plotting
plot_extent = (-object_extent/2, object_extent/2, -object_extent/2, object_extent/2)

# Plot 2: Ground Truth Amplitude
plt.subplot(2, 2, 2)
plt.imshow(torch.abs(gt_object).cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Ground Truth Amplitude')
plt.colorbar()

# Plot 3: Reconstructed Amplitude
plt.subplot(2, 2, 3)
plt.imshow(recon_model.object_amplitude.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed Amplitude')
plt.colorbar()

# Plot 4: Reconstructed Phase
plt.subplot(2, 2, 4)
plt.imshow(recon_model.object_phase.detach().cpu().numpy(), extent=plot_extent, origin='lower')
plt.title('Reconstructed Phase')
plt.colorbar()

plt.tight_layout()
plt.savefig('diptych_reconstruction.png')
print("Saved reconstruction results to 'diptych_reconstruction.png'")

import torch
import matplotlib.pyplot as plt
import numpy as np
from src import adaptive_optics as ao

def plot_actuators(mirror, annotate=False):
    pts = mirror.actuator_coords.detach().cpu().numpy()  # (N,2)
    x, y = pts[:, 0], pts[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")

    # Draw the boundary circle
    circle = plt.Circle((0, 0), mirror.mirror_radius, fill=False, color="k", linewidth=1.5)
    ax.add_patch(circle)

    ax.scatter(x, y, s=25, alpha=0.9)
    if annotate:
        for k, (px, py) in enumerate(pts):
            ax.text(px, py, str(k), fontsize=7, ha="center", va="center")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Fibonacci disk actuator placement (N={mirror.number_actuators}, R={mirror.mirror_radius:.3f}m)")
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig('actuator_placement.png')
    print("Saved actuator placement to 'actuator_placement.png'")
    # plt.show()


def plot_mirror_response(mirror, commands=None):
    if commands is None:
        # Generate some random smooth-ish commands if none provided
        commands = torch.rand(mirror.number_actuators, device=mirror.device) * 0.5
        
    surface = mirror.forward(commands).detach().cpu().numpy()
    
    extent = mirror.grid_size_wavefront * mirror.grid_spacing_wavefront
    plot_extent = [-extent/2, extent/2, -extent/2, extent/2]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(surface, extent=plot_extent, origin='lower', cmap='viridis')
    
    # Overlay the actuator positions
    pts = mirror.actuator_coords.detach().cpu().numpy()
    plt.scatter(pts[:, 0], pts[:, 1], c='red', s=5, alpha=0.5, label='Actuators')
    
    plt.colorbar(im, label='Surface Displacement (m)')
    plt.title('Deformable Mirror Surface Response')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/mirror_response.png')
    print("Saved mirror response to 'mirror_response.png'")
    # plt.show()



device = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. Initialize the mirror
mirror = ao.deformableMirror(
    number_actuators=64,
    max_stroke=1e-6, # 1 micron max stroke
    mirror_radius=5e-3, # 5 mm mirror
    grid_size_wavefront=256,
    grid_spacing_wavefront=(12e-3)/256, # 12 mm total grid
    device=device
)

# 2. Visualize the Fibonacci actuator placement
plot_actuators(mirror, annotate=False)

# 3. Create a test pattern (e.g., push the center up, pull the edges down)
pts = mirror.actuator_coords
r_squared = pts[:, 0]**2 + pts[:, 1]**2
# Normalize r from 0 to 1
r_norm = r_squared / (mirror.mirror_radius**2)
# Command: center is +1, edge is -1
test_commands = torch.cos(r_norm * torch.pi) 

# 4. Visualize the mirror response
plot_mirror_response(mirror, None)


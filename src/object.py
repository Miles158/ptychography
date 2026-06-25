"""
object.py
Written by Miles Houston with the help of Gemini

This file houses functions that can create object planes 
useful to the testing of Ptychography models. 
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.font_manager as fm

def make_text_mask(text, grid_size, margin=0., stroke_frac=0.01, font_name="Helvetica", device="cpu"):
    """
    Creates a binary mask of a text string.
    """
    # Find an installed font path using matplotlib
    font_path = fm.findfont(font_name, fallback_to_default=True)
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Could not locate a TTF for font_name={font_name}. "
                                f"Matplotlib returned: {font_path}")

    # Create grayscale canvas
    img = Image.new("L", (grid_size, grid_size), 0)
    draw = ImageDraw.Draw(img)

    stroke_w = max(1, int(grid_size * stroke_frac))  

    usable = int((1 - 2 * margin) * grid_size)
    lo, hi = 1, grid_size
    best = lo

    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bbox = draw.textbbox((0, 0), text, font=f)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if (w + 2 * stroke_w) <= usable and (h + 2 * stroke_w) <= usable:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    font = ImageFont.truetype(font_path, best)

    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x0 = (grid_size - w) // 2
    y0 = (grid_size - h) // 2

    draw.text(
        (x0, y0), text,
        font=font,
        fill=255,
        stroke_width=stroke_w,
        stroke_fill=255
    )
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    mask = (np.array(img) > 128).astype(np.float32)

    return torch.from_numpy(mask).to(torch.complex64).to(device)


def make_siemens_star_mask(
    grid_size,
    num_sectors=64,        # number of converging "spokes" / angular segments
    margin=0.0,            # keep pattern inside (like make_text_mask's margin)
    line_fraction=1.0,    # 1.0 = fill whole sector (classic star); <1.0 = thinner wedges
    center_radius_frac=0.0, # remove/blank central disk as fraction of radius (0..1)
    device="cpu"
):
    """
    Creates a Siemens-star-like binary mask (0/1) inside an outer circle.

    Returns: complex64 tensor of shape (grid_size, grid_size).
    """
    # Coordinate grid centered at (0,0)
    # Use pixel-center coordinates so the pattern is symmetric.
    coords = torch.arange(grid_size, device=device, dtype=torch.float32) - (grid_size - 1) / 2
    X, Y = torch.meshgrid(coords, coords, indexing='xy')

    r = torch.sqrt(X**2 + Y**2)

    # Outer radius based on margin, matching idea of (1 - 2*margin)*grid_size usable
    outer_radius = (grid_size / 2.0) * (1.0 - margin)

    # Angle in [0, 2pi)
    theta = torch.atan2(Y, X)  # [-pi, pi]
    theta = (theta + 2.0 * np.pi) % (2.0 * np.pi)

    # Map angle to sector index and position within sector
    u = theta / (2.0 * np.pi) * num_sectors  # continuous
    sector_index = torch.floor(u).to(torch.int64)
    frac = u - sector_index.to(torch.float32)  # [0,1)

    # Within-sector angular coordinate centered at 0: d in [-0.5, 0.5]
    d = frac - 0.5

    # If line_fraction=1.0 -> full sector filled.
    # If smaller -> only a central wedge is "ON" (thinner lines).
    half_width = 0.5 * float(line_fraction)
    wedge_on = (d.abs() <= half_width)

    # Classic Siemens star alternates black/white sectors
    # (sector parity chooses which adjacent sector is "white")
    mask = ((sector_index % 2) == 0) & wedge_on

    # Apply outer circle support
    mask = mask & (r <= outer_radius)

    # Optionally blank the center
    if center_radius_frac > 0:
        inner_radius = outer_radius * float(center_radius_frac)
        mask = mask & (r >= inner_radius)

    # Match your text mask orientation (you flip FLIP_TOP_BOTTOM there)
    mask = torch.flip(mask, dims=[0])

    mask = mask.to(torch.float32)
    return mask.to(torch.complex64)


import torch

def make_zone_plate_mask(
    grid_size,
    pixel_size=1e-6,             # physical size of one pixel in meters (e.g., 1 um)
    focal_length=0.01,           # focal length in meters (e.g., 1 cm)
    wavelength=632e-9,           # default to 632 nm (HeNe laser)
    margin=0.0,                  # keep pattern inside an outer circle
    central_transparent=True,    # True = center is 1; False = center is 0
    device="cpu"
):
    """
    Creates a Fresnel zone plate binary mask (0/1) inside an outer circle.

    Returns: complex64 tensor of shape (grid_size, grid_size).
    """
    # Coordinate grid centered at (0,0)
    # Use pixel-center coordinates to keep it symmetric
    coords = torch.arange(grid_size, device=device, dtype=torch.float32) - (grid_size - 1) / 2
    
    # Scale coordinates by pixel size to get physical dimensions
    X, Y = torch.meshgrid(coords * pixel_size, coords * pixel_size, indexing='xy')
    
    r_sq = X**2 + Y**2
    r = torch.sqrt(r_sq)

    # Maximum physical radius based on margin
    max_physical_radius = (grid_size / 2.0) * pixel_size
    outer_radius = max_physical_radius * (1.0 - margin)

    # Zone plate mathematical logic: 
    # A zone switches phase when radius squared crosses multiples of (wavelength * focal_length)
    zone_index = torch.floor(r_sq / (wavelength * focal_length)).to(torch.int64)

    # Alternating zones: even indices are one state, odd are the other
    if central_transparent:
        mask = (zone_index % 2) == 0
    else:
        mask = (zone_index % 2) == 1

    # Apply outer circle support (trim edges based on margin)
    mask = mask & (r <= outer_radius)

    # Match the orientation of your other masks
    mask = torch.flip(mask, dims=[0])

    mask = mask.to(torch.float32)
    return mask.to(torch.complex64)

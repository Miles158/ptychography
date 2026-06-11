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

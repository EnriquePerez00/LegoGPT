import numpy as np
import pytest
import os
from PIL import Image
from src.mosaic_generator import hex_to_rgb, rgb_to_lab, delta_e_cie76, generate_mosaic_from_image

def test_hex_to_rgb():
    assert hex_to_rgb("#ffffff") == (255, 255, 255)
    assert hex_to_rgb("#000000") == (0, 0, 0)
    assert hex_to_rgb("FF0000") == (255, 0, 0)

def test_rgb_to_lab_bounds():
    l, a, b = rgb_to_lab(255, 255, 255)
    # White should have high lightness
    assert l > 95.0
    
    l_k, a_k, b_k = rgb_to_lab(0, 0, 0)
    # Black should have near zero lightness
    assert l_k < 5.0

def test_delta_e():
    lab_white = rgb_to_lab(255, 255, 255)
    lab_black = rgb_to_lab(0, 0, 0)
    
    # Distance between white and black should be large
    dist = delta_e_cie76(lab_white, lab_black)
    assert dist > 80.0
    
    # Distance to self should be 0
    assert delta_e_cie76(lab_white, lab_white) == 0.0

def test_mosaic_generation(tmp_path):
    # Create a simple 8x8 mock image with red, green, and blue pixels
    img_data = np.zeros((8, 8, 3), dtype=np.uint8)
    img_data[0:4, 0:4] = [255, 0, 0] # Red
    img_data[4:8, 4:8] = [0, 255, 0] # Green
    
    temp_img_path = os.path.join(tmp_path, "mock_input.png")
    Image.fromarray(img_data).save(temp_img_path)
    
    # Generate mosaic with grid size 8
    parts = generate_mosaic_from_image(temp_img_path, size=8, use_dithering=False)
    
    # Grid size 8x8 means exactly 64 plates
    assert len(parts) == 64
    
    # Check that they are 1x1 plates (3024.dat)
    for p in parts:
        assert p.part_id == "3024.dat"
        # Y position is flat on ground (0.0)
        assert p.transform[1, 3] == 0.0
        
    # Check that coordinates are centered around origin
    xs = [p.transform[0, 3] for p in parts]
    zs = [p.transform[2, 3] for p in parts]
    
    assert min(xs) == -80.0 # (0 - 4) * 20
    assert max(xs) == 60.0  # (7 - 4) * 20
    assert min(zs) == -80.0
    assert max(zs) == 60.0

import os
import json
import math
import numpy as np
from PIL import Image
from src.parser import ParsedPart

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip('#')
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)

def rgb_to_xyz(r: int, g: int, b: int) -> tuple[float, float, float]:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    return x * 100, y * 100, z * 100

def xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    ref_x, ref_y, ref_z = 95.047, 100.0, 108.883
    x, y, z = x / ref_x, y / ref_y, z / ref_z
    
    def pivot(v):
        return v ** (1/3) if v > 0.008856 else (7.787 * v) + (16 / 116)
        
    x, y, z = pivot(x), pivot(y), pivot(z)
    l = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return l, a, b

def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)

def delta_e_cie76(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    return math.sqrt((lab1[0] - lab2[0])**2 + (lab1[1] - lab2[1])**2 + (lab1[2] - lab2[2])**2)

class LegoColorPalette:
    def __init__(self, catalog_path: str = "public/color_catalog.json"):
        self.colors = []
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for cid, info in data.items():
                    cat = info.get("category", "").lower()
                    mat = info.get("material_type", "").lower()
                    
                    # Filter out transparent and metallic/chrome/glitter/speckle to keep mosaic clean
                    is_transparent = "trans" in cat or mat == "transparent"
                    is_metallic = "chrome" in cat or "glitter" in cat or "speckle" in cat or "metal" in cat
                    
                    if not is_transparent and not is_metallic:
                        ldraw_str = info.get("ldraw_code")
                        if ldraw_str and ldraw_str.isdigit():
                            ldraw_code = int(ldraw_str)
                            hex_str = info.get("hex", "#ffffff")
                            r, g, b = hex_to_rgb(hex_str)
                            lab = rgb_to_lab(r, g, b)
                            self.colors.append({
                                "ldraw_code": ldraw_code,
                                "name": info.get("name"),
                                "rgb": (r, g, b),
                                "lab": lab
                            })
            except Exception as e:
                print(f"Error loading color catalog: {e}")
                
        if not self.colors:
            fallback_palette = {
                0: "#05131D",   # Black
                1: "#0055A2",   # Blue
                2: "#237841",   # Green
                4: "#C91A09",   # Red
                14: "#F2CD37",  # Yellow
                15: "#F2F3F2",  # White
                71: "#A0A5A9",  # Light Bluish Gray
                72: "#5A5F62",  # Dark Bluish Gray
            }
            for code, hex_str in fallback_palette.items():
                r, g, b = hex_to_rgb(hex_str)
                lab = rgb_to_lab(r, g, b)
                self.colors.append({
                    "ldraw_code": code,
                    "name": f"Color {code}",
                    "rgb": (r, g, b),
                    "lab": lab
                })

    def find_closest_color(self, r: int, g: int, b: int) -> dict:
        target_lab = rgb_to_lab(r, g, b)
        best_color = self.colors[0]
        min_dist = float('inf')
        for color in self.colors:
            dist = delta_e_cie76(target_lab, color["lab"])
            if dist < min_dist:
                min_dist = dist
                best_color = color
        return best_color

def generate_mosaic_from_image(image_path: str, size: int = 32, use_dithering: bool = True) -> list[ParsedPart]:
    """
    Converts an input image into a 2D Lego Plate 1x1 mosaic.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=float)
    
    palette = LegoColorPalette()
    grid = [[None for _ in range(size)] for _ in range(size)]
    
    if use_dithering:
        for y in range(size):
            for x in range(size):
                r, g, b = pixels[y, x]
                rc = int(np.clip(r, 0, 255))
                gc = int(np.clip(g, 0, 255))
                bc = int(np.clip(b, 0, 255))
                
                closest = palette.find_closest_color(rc, gc, bc)
                grid[y][x] = closest["ldraw_code"]
                
                r_m, g_m, b_m = closest["rgb"]
                err_r, err_g, err_b = r - r_m, g - g_m, b - b_m
                
                if x + 1 < size:
                    pixels[y, x + 1] += [err_r * 7 / 16, err_g * 7 / 16, err_b * 7 / 16]
                if x - 1 >= 0 and y + 1 < size:
                    pixels[y + 1, x - 1] += [err_r * 3 / 16, err_g * 3 / 16, err_b * 3 / 16]
                if y + 1 < size:
                    pixels[y + 1, x] += [err_r * 5 / 16, err_g * 5 / 16, err_b * 5 / 16]
                if x + 1 < size and y + 1 < size:
                    pixels[y + 1, x + 1] += [err_r * 1 / 16, err_g * 1 / 16, err_b * 1 / 16]
    else:
        for y in range(size):
            for x in range(size):
                r, g, b = pixels[y, x]
                closest = palette.find_closest_color(int(r), int(g), int(b))
                grid[y][x] = closest["ldraw_code"]
                
    parts = []
    for y in range(size):
        for x in range(size):
            color = grid[y][x]
            pos_x = (x - size / 2.0) * 20.0
            pos_z = (y - size / 2.0) * 20.0
            pos_y = 0.0
            
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = [pos_x, pos_y, pos_z]
            
            parts.append(ParsedPart(
                part_id="3024.dat",
                color=color,
                transform=transform,
                step_id=0
            ))
            
    return parts

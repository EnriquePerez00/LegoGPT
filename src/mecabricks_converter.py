import json
import os
import numpy as np
from src.parser import ParsedPart

# Default mappings from Mecabricks parts and colors to LDraw
MECABRICKS_TO_LDRAW_PARTS = {
    # Basic bricks
    "3005": "3005.dat",
    "3004": "3004.dat",
    "3003": "3003.dat",
    "3002": "3002.dat",
    "3001": "3001.dat",
    "3010": "3010.dat",
    "3009": "3009.dat",
    "3008": "3008.dat",
    "3007": "3007.dat",
    "3006": "3006.dat",
    # Plates
    "3024": "3024.dat",
    "3023": "3023.dat",
    "3022": "3022.dat",
    "3021": "3021.dat",
    "3020": "3020.dat",
    "3710": "3710.dat",
    "3666": "3666.dat",
    "3460": "3460.dat",
    "3032": "3032.dat",
    "3031": "3031.dat",
    "3030": "3030.dat",
    "3035": "3035.dat",
    "3034": "3034.dat",
    "3832": "3832.dat",
    "3036": "3036.dat",
    "3958": "3958.dat",
    # Others
    "3062": "3062b.dat",
    "3062b": "3062b.dat",
    "3941": "3941.dat",
}

MECABRICKS_TO_LDRAW_COLORS = {
    # Solid Colors
    "black": 0,
    "blue": 1,
    "green": 2,
    "red": 4,
    "brown": 6,
    "light_gray": 7,
    "dark_gray": 8,
    "light_blue": 9,
    "bright_green": 10,
    "light_green": 11,
    "yellow": 14,
    "white": 15,
    "orange": 25,
    "magenta": 26,
    "pink": 29,
    "medium_blue": 73,
    "dark_blue": 272,
    "sand_green": 378,
    # Common prefixes
    "solid_black": 0,
    "solid_blue": 1,
    "solid_green": 2,
    "solid_red": 4,
    "solid_light_gray": 7,
    "solid_dark_gray": 8,
    "solid_yellow": 14,
    "solid_white": 15,
}

def map_part_id(mb_id: str) -> str:
    """Maps a Mecabricks part ID to an LDraw filename."""
    # Clean string
    clean_id = mb_id.strip().lower()
    # Strip any extension if present
    if clean_id.endswith(".dat"):
        return clean_id
        
    mapped = MECABRICKS_TO_LDRAW_PARTS.get(clean_id)
    if mapped:
        return mapped
        
    # Standard fallback: append .dat
    return f"{clean_id}.dat"

def map_color_id(mb_color: str) -> int:
    """Maps a Mecabricks color name/id to an LDraw color ID code."""
    if not mb_color:
        return 7 # Default light gray
        
    clean_color = mb_color.strip().lower()
    
    # Try direct mapping
    mapped = MECABRICKS_TO_LDRAW_COLORS.get(clean_color)
    if mapped is not None:
        return mapped
        
    # Try finding substring
    for key, val in MECABRICKS_TO_LDRAW_COLORS.items():
        if key in clean_color:
            return val
            
    # Try parsing color if it is a number
    if clean_color.isdigit():
        return int(clean_color)
        
    return 7 # Default fallback

def convert_matrix(mb_matrix: list[float], scale: float = 2500.0) -> np.ndarray:
    """
    Converts a Mecabricks 4x4 matrix (in row-major list format)
    to LDraw's coordinate system (Y-down, scale multiplier).
    """
    if len(mb_matrix) != 16:
        # Fallback to identity matrix
        return np.eye(4, dtype=np.float32)
        
    M_mb = np.array(mb_matrix, dtype=np.float32).reshape(4, 4)
    
    # Create target matrix
    M_ldraw = np.eye(4, dtype=np.float32)
    
    # Apply Y-up to Y-down axis swap to the rotation matrix
    # P = diag(1, -1, -1)
    # R_ld = P * R_mb * P
    R_mb = M_mb[:3, :3]
    P = np.diag([1.0, -1.0, -1.0])
    R_ldraw = P @ R_mb @ P
    M_ldraw[:3, :3] = R_ldraw
    
    # Apply translation with scale and flipped signs for Y and Z
    M_ldraw[0, 3] = M_mb[0, 3] * scale
    M_ldraw[1, 3] = -M_mb[1, 3] * scale
    M_ldraw[2, 3] = -M_mb[2, 3] * scale
    
    return M_ldraw

def parse_mbx_content(json_str: str, scale: float = 2500.0) -> list[ParsedPart]:
    """Parses .mbx JSON string and returns a list of ParsedParts."""
    data = json.loads(json_str)
    parts = []
    
    # Model could be structured as an object with "nodes" or "parts"
    nodes = data.get("nodes", data.get("parts", []))
    
    for idx, node in enumerate(nodes):
        mb_id = node.get("id", node.get("name", "3005"))
        mb_color = node.get("color", node.get("material", "solid_light_gray"))
        mb_matrix = node.get("matrix", [])
        step_id = node.get("step", 0)
        
        # Mappings
        part_id = map_part_id(mb_id)
        color = map_color_id(mb_color)
        transform = convert_matrix(mb_matrix, scale=scale)
        
        parts.append(ParsedPart(
            part_id=part_id,
            color=color,
            transform=transform,
            step_id=step_id
        ))
        
    return parts

def parse_zmbx_file(file_path: str, scale: float = 2500.0) -> list[ParsedPart]:
    """Extracts scene.mbx from .zmbx zip file and parses it."""
    import zipfile
    
    if not zipfile.is_zipfile(file_path):
        # Maybe it's a raw .mbx file
        with open(file_path, "r", encoding="utf-8") as f:
            return parse_mbx_content(f.read(), scale=scale)
            
    with zipfile.ZipFile(file_path, "r") as z:
        # Find the .mbx file inside
        mbx_files = [f for f in z.namelist() if f.endswith(".mbx")]
        if not mbx_files:
            raise FileNotFoundError("No .mbx file found in the .zmbx archive.")
            
        mbx_file = mbx_files[0]
        with z.open(mbx_file) as f:
            content = f.read().decode("utf-8")
            return parse_mbx_content(content, scale=scale)

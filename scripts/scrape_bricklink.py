import os
import zipfile
import requests
import xml.etree.ElementTree as ET
import numpy as np

def convert_io_to_ldr(io_file_path: str, output_ldr_path: str) -> bool:
    """
    Parses a BrickLink Studio .io file (which is a ZIP containing document.xml)
    and converts it to standard LDraw .ldr format.
    """
    if not zipfile.is_zipfile(io_file_path):
        print(f"Error: {io_file_path} is not a valid zip/Studio file.")
        return False
        
    try:
        with zipfile.ZipFile(io_file_path, 'r') as archive:
            if 'document.xml' not in archive.namelist():
                print("Error: document.xml not found inside .io archive.")
                return False
                
            xml_content = archive.read('document.xml')
            
        root = ET.fromstring(xml_content)
        
        # We will parse the parts and their transform matrices
        # In document.xml, instances are typically inside <BuildingInformation><Instances><Part>
        parts_data = []
        for part in root.findall(".//Part"):
            design = part.get("design")  # e.g., "3001"
            color = part.get("color")    # BrickLink color ID
            
            # Rotation and position is stored in matrix format or position tags
            # Let's extract position/rotation from matrix attribute
            matrix_str = part.get("matrix") # Format: "m00,m01,m02,0,m10,m11,m12,0,..." or similar
            tx = float(part.get("tx", 0.0))
            ty = float(part.get("ty", 0.0))
            tz = float(part.get("tz", 0.0))
            
            # Rotation matrix components
            # Studio uses specific tags, let's extract them
            m00 = float(part.get("m00", 1.0))
            m01 = float(part.get("m01", 0.0))
            m02 = float(part.get("m02", 0.0))
            m10 = float(part.get("m10", 0.0))
            m11 = float(part.get("m11", 1.0))
            m12 = float(part.get("m12", 0.0))
            m20 = float(part.get("m20", 0.0))
            m21 = float(part.get("m21", 0.0))
            m22 = float(part.get("m22", 1.0))
            
            part_id = f"{design}.dat"
            # Color conversion: BrickLink color IDs mapping (fallback to LDraw code directly)
            color_id = int(color) if color else 14
            
            parts_data.append({
                "part_id": part_id,
                "color": color_id,
                "pos": [tx, ty, tz],
                "rot": [m00, m01, m02, m10, m11, m12, m20, m21, m22]
            })
            
        # Write LDraw representation
        lines = ["0 BrickLink Studio converted LDraw model"]
        for p in parts_data:
            x, y, z = p["pos"]
            # Convert units if necessary (Studio is in LDraw scale 1:1 generally)
            rot_str = " ".join(f"{val:.6f}" for val in p["rot"])
            lines.append(f"1 {p['color']} {x:.4f} {y:.4f} {z:.4f} {rot_str} {p['part_id']}")
            
        with open(output_ldr_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
        print(f"Successfully converted {io_file_path} to {output_ldr_path}")
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

def download_bricklink_model(model_id: str, output_dir: str = "data/bricklink") -> str:
    """
    Downloads a custom model from the BrickLink Studio gallery.
    """
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.bricklink.com/ajax/clone/studio/gallery/download.ajax?id={model_id}"
    output_path = os.path.join(output_dir, f"{model_id}.io")
    
    print(f"Downloading from BrickLink Studio Gallery: {url}")
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(res.content)
            print(f"Saved Studio file to {output_path}")
            return output_path
        else:
            print(f"BrickLink returned status {res.status_code}")
            return ""
    except Exception as e:
        print(f"Connection error downloading model {model_id}: {e}")
        return ""

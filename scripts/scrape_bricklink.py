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
            # 1. Prioritize direct extraction of nested .ldr files
            ldr_files = [n for n in archive.namelist() if n.lower().endswith('.ldr')]
            if ldr_files:
                # Find the main LDraw model inside zip
                main_ldr = "model.ldr"
                for m_name in ["model.ldr", "modelv2.ldr", "model2.ldr"]:
                    if m_name in ldr_files:
                        main_ldr = m_name
                        break
                if main_ldr not in ldr_files:
                    main_ldr = ldr_files[0]
                    
                try:
                    content = archive.read(main_ldr, pwd=b"soho0909").decode("utf-8", errors="ignore")
                except Exception:
                    # Fallback to no password
                    content = archive.read(main_ldr).decode("utf-8", errors="ignore")
                with open(output_ldr_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Successfully extracted nested {main_ldr} from {io_file_path} to {output_ldr_path}")
                return True
                
            # 2. Fallback to XML parser if no .ldr exists
            if 'document.xml' not in archive.namelist():
                print("Error: No .ldr or document.xml found in .io file.")
                return False
                
            xml_content = archive.read('document.xml')
            
        root = ET.fromstring(xml_content)
        
        parts_data = []
        for part in root.findall(".//Part"):
            design = part.get("design")
            color = part.get("color")
            tx = float(part.get("tx", 0.0))
            ty = float(part.get("ty", 0.0))
            tz = float(part.get("tz", 0.0))
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
            color_id = int(color) if color else 14
            
            parts_data.append({
                "part_id": part_id,
                "color": color_id,
                "pos": [tx, ty, tz],
                "rot": [m00, m01, m02, m10, m11, m12, m20, m21, m22]
            })
            
        lines = ["0 BrickLink Studio converted LDraw model"]
        for p in parts_data:
            x, y, z = p["pos"]
            rot_str = " ".join(f"{val:.6f}" for val in p["rot"])
            lines.append(f"1 {p['color']} {x:.4f} {y:.4f} {z:.4f} {rot_str} {p['part_id']}")
            
        with open(output_ldr_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
        print(f"Successfully converted XML to {output_ldr_path}")
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

def download_bricklink_model(model_id: str, output_dir: str = "data/bricklink_raw") -> str:
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

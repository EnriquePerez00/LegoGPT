import os
import json
import glob
from src.parser import parse_ldraw_file

def categorize_by_theme(theme_name: str) -> str:
    """
    Groups official theme names into broad catalog families.
    """
    theme = theme_name.lower()
    if any(k in theme for k in ["star wars", "space", "mars", "galaxy", "rocket", "shuttle"]):
        return "Space"
    elif any(k in theme for k in ["car", "truck", "vehicle", "speed", "technic", "train", "racer"]):
        return "Vehicles"
    elif any(k in theme for k in ["castle", "house", "building", "modular", "city", "creator", "station", "shop"]):
        return "Structures"
    elif any(k in theme for k in ["bionicle", "mech", "robot", "creature", "beast", "dino", "animal"]):
        return "Creatures"
    return "Other"

def build_catalog(standardized_dir: str = "data/standardized", output_path: str = "data/catalog.json") -> dict:
    """
    Scans the standardized LDraw files, extracts their metadata, and categorizes them.
    """
    catalog = {}
    
    # Check if catalog exists to merge
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            pass
            
    files = glob.glob(os.path.join(standardized_dir, "*.ldr")) + glob.glob(os.path.join(standardized_dir, "*.mpd"))
    
    print(f"Scanning {len(files)} files in {standardized_dir}...")
    for f_path in files:
        file_id = os.path.splitext(os.path.basename(f_path))[0]
        
        # Read headers
        name = "Unknown Lego Set"
        theme = "General"
        
        try:
            with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str.startswith("0"):
                        break # End of LDraw header comment block
                        
                    tokens = line_str.split()
                    if len(tokens) >= 3 and tokens[1].upper() == "THEME":
                        theme = " ".join(tokens[2:])
                    elif len(tokens) >= 2 and tokens[1].upper() == "NAME":
                        name = " ".join(tokens[2:])
                        
            # Parse parts to count
            parts = parse_ldraw_file(f_path)
            num_pieces = len(parts)
            
            # Map theme to family
            family = categorize_by_theme(theme or name)
            
            catalog[file_id] = {
                "source": "Local" if "omr" not in file_id else "OMR",
                "name": name,
                "pieces": num_pieces,
                "theme": theme,
                "family": family,
                "file_path": f_path
            }
        except Exception as e:
            print(f"Error parsing file {f_path}: {e}")
            
    # Save catalog
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    print(f"Catalog saved to {output_path} with {len(catalog)} items.")
    return catalog

if __name__ == "__main__":
    build_catalog()

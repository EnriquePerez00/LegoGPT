import os
import re
import json
import torch
import numpy as np
from src.mpd_parser import flatten_mpd
from src.parser import build_pyg_graph, ParsedPart

# Common vehicle-related parts in LDraw (wheels, tires, axles, steering, mudguards)
VEHICLE_PART_IDS = {
    # Tires and Wheels
    "3139.dat", "42610.dat", "56902.dat", "30027.dat", "18976.dat", "18977.dat", 
    "55981.dat", "55982.dat", "30285.dat", "6014.dat", "6015.dat", "56890.dat",
    "56891.dat", "30285b.dat", "6014b.dat",
    # Axles
    "2926.dat", "4150.dat", "4274.dat",
    # Steering
    "3829.dat", "3829c01.dat",
    # Mudguards / Fenders
    "50745.dat", "98282.dat", "30029.dat"
}

VEHICLE_KEYWORDS = [
    "car", "truck", "rover", "racer", "buggy", "vehicle", "tractor", "kart", 
    "roadster", "formula1", "tow-truck", "jeep", "4wd", "chassis", "auto"
]

def is_vehicle_model(parts: list[ParsedPart], filename: str) -> bool:
    """
    Heuristic to determine if a model is a vehicle:
    - Contains at least 4 parts from our vehicle part set (mostly wheels/tires/axles).
    - OR has a filename/title keyword indicating a vehicle and at least 2 wheels.
    """
    wheel_count = 0
    vehicle_part_count = 0
    for p in parts:
        p_id = p.part_id.lower()
        if p_id in VEHICLE_PART_IDS:
            vehicle_part_count += 1
            if "wheel" in p_id or "tire" in p_id or "rim" in p_id or "tyre" in p_id or p_id in ["3139.dat", "30027.dat", "6015.dat", "18977.dat", "56890.dat", "56891.dat"]:
                wheel_count += 1
                
    filename_lower = filename.lower()
    has_keyword = any(kw in filename_lower for kw in VEHICLE_KEYWORDS)
    
    # Standard car: 4 wheels
    if wheel_count >= 4:
        return True
        
    # Bounded car/kart/motorcycle with keywords: at least 2 wheels
    if has_keyword and (wheel_count >= 2 or vehicle_part_count >= 4):
        return True
        
    return False

def process_vehicle_model(file_path: str, output_dir: str = "data/processed_vehicles") -> bool:
    """
    Processes a single raw vehicle LDraw model, extracting its structure and saving it.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(file_path)
    set_name = os.path.splitext(filename)[0]
    
    # Clean set name to get set number
    set_number_match = re.match(r'^([0-9]+)', set_name)
    set_number = set_number_match.group(1) if set_number_match else set_name
    
    try:
        parts = flatten_mpd(file_path)
        num_parts = len(parts)
        
        if num_parts < 10 or num_parts >= 150:
            # Skip extremely small or large builds to keep model training feasible locally
            return False
            
        if not is_vehicle_model(parts, filename):
            return False
            
        print(f"Detectado vehículo: {filename} ({num_parts} piezas)")
        
        # Sort bottom-up by Y coordinate first, then by step_id
        # In LDraw, higher Y is closer to the ground, so reverse sorting by Y puts wheels first
        parts_sorted = sorted(parts, key=lambda p: (-p.transform[1, 3], p.step_id))
        
        # Save sequence JSON
        parts_json = []
        for idx, p in enumerate(parts_sorted):
            parts_json.append({
                "sequence_index": idx,
                "part_id": p.part_id,
                "color": p.color,
                "transform": p.transform.flatten().tolist(),
                "step_id": p.step_id
            })
            
        assembly_path = os.path.join(output_dir, f"{set_number}_assembly.json")
        with open(assembly_path, "w", encoding="utf-8") as f:
            json.dump({
                "set_name": set_name,
                "set_number": set_number,
                "num_parts": num_parts,
                "parts": parts_json
            }, f, indent=2)
            
        # Create and save graph representation
        # Import allowed parts from src.parser for mapping consistency
        from src.parser import ALLOWED_PARTS
        
        graph_data = build_pyg_graph(parts_sorted, allowed_parts=ALLOWED_PARTS)
        graph_path = os.path.join(output_dir, f"{set_number}_graph.pt")
        torch.save(graph_data, graph_path)
        
        return True
    except Exception as e:
        print(f"Error procesando {filename}: {e}")
        return False

def extract_all_vehicles(raw_dir: str = "data/omr_raw", output_dir: str = "data/processed_vehicles"):
    if not os.path.exists(raw_dir):
        print(f"Directorio de datos raw '{raw_dir}' no encontrado.")
        return
        
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(('.mpd', '.ldr'))]
    print(f"Escaneando {len(files)} archivos para extracción de vehículos...")
    
    count = 0
    for f_path in files:
        if process_vehicle_model(f_path, output_dir):
            count += 1
            
    print(f"\nFinalizado. Se extrajeron {count} modelos de vehículos a '{output_dir}'.")

if __name__ == "__main__":
    extract_all_vehicles()

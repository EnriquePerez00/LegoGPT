import os
import json
import torch
import numpy as np
from typing import List
from src.mpd_parser import flatten_mpd
from src.parser import build_pyg_graph, ParsedPart

def generate_set_dataset(file_path: str, output_dir: str = "data/processed") -> bool:
    """
    Processes a single LDraw (.mpd or .ldr) file, filters it if it has < 100 parts,
    and generates its spatial graph and assembly sequence files.
    
    Returns:
        True if processed and saved successfully, False if filtered out or failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(file_path)
    set_name = os.path.splitext(filename)[0]
    
    # Extract set number (e.g. from 31054-1_Desert-Rover -> 31054)
    if set_name.startswith("9999-"):
        match = re.match(r'^(9999-[0-9]+)', set_name)
        set_number = match.group(1) if match else set_name
    else:
        set_number_match = re.match(r'^([0-9]+)', set_name)
        set_number = set_number_match.group(1) if set_number_match else set_name

    
    print(f"\nProcesando {filename}...")
    
    try:
        # 1. Flatten the MPD model into physical parts
        parts = flatten_mpd(file_path)
        num_parts = len(parts)
        
        print(f"  Total piezas físicas: {num_parts}")
        
        if num_parts == 0:
            print(f"  [OMITIDO] El archivo no contiene piezas físicas.")
            return False
            
        # 2. Sort parts by step_id to represent the correct assembly sequence
        parts_sorted = sorted(parts, key=lambda p: p.step_id)
        
        # 3. Create JSON representation of parts and step sequence
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
        print(f"  Secuencia de montaje guardada en {assembly_path}")
        
        # 4. Create and save PyG Graph of the fully assembled model
        graph_data = build_pyg_graph(parts_sorted)
        graph_path = os.path.join(output_dir, f"{set_number}_graph.pt")
        torch.save(graph_data, graph_path)
        print(f"  Grafo de conectividad PyG guardado en {graph_path}")
        
        return True
        
    except Exception as e:
        print(f"  Error procesando {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

import re

def process_all_raw_models(raw_dir: str = "data/omr_raw", output_dir: str = "data/processed"):
    """
    Loops through all raw MPD/LDR models, filters those with < 100 parts, and processes them.
    """
    if not os.path.exists(raw_dir):
        print(f"Directorio {raw_dir} no existe. Por favor descarga los datos primero.")
        return
        
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(('.mpd', '.ldr'))]
    print(f"Encontrados {len(files)} archivos en {raw_dir}")
    
    processed_count = 0
    for f_path in files:
        success = generate_set_dataset(f_path, output_dir)
        if success:
            processed_count += 1
            
    print(f"\nProceso completado. {processed_count} de {len(files)} sets guardados en {output_dir}")

if __name__ == "__main__":
    process_all_raw_models()

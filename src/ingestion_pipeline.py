import os
import json
import sqlite3
import re
import numpy as np
import networkx as nx
import torch

from src.mpd_parser import flatten_mpd
from src.parser import build_pyg_graph
from scripts.scrape_bricklink import convert_io_to_ldr
from src.validator import check_connection_optimized

def process_and_register_downloaded_model(
    file_path: str,
    source: str,
    source_url: str = None,
    image_url: str = None,
    db_path: str = "data/catalog/models_catalog.db",
    output_standardized_dir: str = "data/standardized",
    output_processed_dir: str = "data/processed"
) -> bool:
    """
    Unified pipeline that:
    1. Converts BrickLink .io files to standard LDraw .ldr files.
    2. Parses/flattens the model structure.
    3. Calculates rich metadata (parts count, connectivity, subassemblies, inventory).
    4. Generates standard assembly sequence (JSON) and graph (.pt) for datasets (< 100 parts).
    5. Registers everything in the local SQLite database including design & image URLs.
    """
    try:
        os.makedirs(output_standardized_dir, exist_ok=True)
        os.makedirs(output_processed_dir, exist_ok=True)
        
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # 1. Handle BrickLink .io conversion to LDraw
        working_file_path = file_path
        if file_path.endswith(".io"):
            ldr_filename = f"{base_name}.ldr"
            ldr_path = os.path.join(output_standardized_dir, ldr_filename)
            print(f"[Pipeline] Convirtiendo .io a LDraw: {file_path} -> {ldr_path}")
            if convert_io_to_ldr(file_path, ldr_path):
                working_file_path = ldr_path
            else:
                print(f"[Pipeline Error] Falló la conversión de {file_path}")
                return False
                
        # Extract set/model ID for database key
        set_id = base_name
        
        # 2. Parse and flatten the LDraw/MPD file structure
        print(f"[Pipeline] Aplanando estructura de {working_file_path}...")
        parts = flatten_mpd(working_file_path)
        parts_count = len(parts)
        print(f"[Pipeline] Piezas físicas encontradas: {parts_count}")
        
        if parts_count == 0:
            print("[Pipeline Error] El modelo no contiene piezas físicas.")
            return False
            
        # 3. Calculate structural connectivity & subassemblies (NetworkX)
        print("[Pipeline] Analizando grafo de conectividad y subensamblajes...")
        G = nx.Graph()
        for idx in range(parts_count):
            G.add_node(idx)
            
        for i in range(parts_count):
            for j in range(i + 1, parts_count):
                if check_connection_optimized(parts[i], parts[j]):
                    G.add_edge(i, j)
                    
        components = list(nx.connected_components(G))
        subassemblies_count = len(components)
        is_fully_connected = 1 if subassemblies_count == 1 else 0
        
        subassemblies_list = []
        for s_idx, comp in enumerate(components):
            comp_parts = [parts[p_idx] for p_idx in comp]
            comp_part_ids = sorted(list(set(p.part_id.lower() for p in comp_parts)))
            # Grounded rule: Y-world coordinate near 0
            grounded = 1 if any(abs(p.transform[1, 3]) <= 1.0 for p in comp_parts) else 0
            subassemblies_list.append({
                "subassembly_id": s_idx,
                "parts_count": len(comp),
                "parts_list": json.dumps(comp_part_ids),
                "is_grounded": grounded
            })
            
        # 4. Generate dataset sequences if part count is < 100
        if parts_count < 100:
            parts_sorted = sorted(parts, key=lambda p: p.step_id)
            parts_json = []
            for idx, p in enumerate(parts_sorted):
                parts_json.append({
                    "sequence_index": idx,
                    "part_id": p.part_id,
                    "color": p.color,
                    "transform": p.transform.flatten().tolist(),
                    "step_id": p.step_id
                })
                
            assembly_path = os.path.join(output_processed_dir, f"{set_id}_assembly.json")
            with open(assembly_path, "w", encoding="utf-8") as f:
                json.dump({
                    "set_name": set_id,
                    "set_number": set_id,
                    "num_parts": parts_count,
                    "parts": parts_json
                }, f, indent=2)
                
            # PyG Graph Generation
            graph_data = build_pyg_graph(parts_sorted)
            graph_path = os.path.join(output_processed_dir, f"{set_id}_graph.pt")
            torch.save(graph_data, graph_path)
            print(f"[Pipeline] Archivos procesados guardados exitosamente para {set_id} (< 100 piezas)")
        else:
            print(f"[Pipeline] Omitiendo generación de secuencia/grafo para {set_id} (piezas: {parts_count} >= 100)")
            
        # 5. Populate local SQLite database
        print(f"[Pipeline] Registrando metadatos en la base de datos local: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Parse LDraw comments/headers for name/theme
        theme = "Unknown"
        name = set_id
        try:
            with open(working_file_path, "r", encoding="utf-8", errors="ignore") as f:
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line.startswith("0 FILE"):
                        continue
                    tokens = line.split()
                    if len(tokens) >= 3 and tokens[0] == "0" and tokens[1].upper() == "THEME":
                        theme = " ".join(tokens[2:])
                    elif len(tokens) >= 3 and tokens[0] == "0" and tokens[1].upper() == "NAME":
                        name = " ".join(tokens[2:])
        except Exception:
            pass
            
        # If BrickLink, theme might be standard
        if source == "BrickLink" and theme == "Unknown":
            theme = "Studio MOC"
            
        # Insert set record
        cursor.execute("""
        INSERT OR REPLACE INTO sets (
            set_id, name, theme, source, file_path, normalized_file_path, 
            parts_count, subassemblies_count, is_fully_connected, source_url, image_url
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            set_id,
            name,
            theme,
            source,
            file_path,
            working_file_path if working_file_path != file_path else None,
            parts_count,
            subassemblies_count,
            is_fully_connected,
            source_url,
            image_url
        ))
        
        # Insert subassemblies
        for sub in subassemblies_list:
            cursor.execute("""
            INSERT OR REPLACE INTO subassemblies (set_id, subassembly_id, parts_count, parts_list, is_grounded)
            VALUES (?, ?, ?, ?, ?)
            """, (
                set_id,
                sub["subassembly_id"],
                sub["parts_count"],
                sub["parts_list"],
                sub["is_grounded"]
            ))
            
        # Insert inventory
        inventory = {}
        for p in parts:
            key = (p.part_id.lower(), p.color)
            inventory[key] = inventory.get(key, 0) + 1
            
        for (part_id, color), qty in inventory.items():
            cursor.execute("""
            INSERT OR REPLACE INTO parts_inventory (set_id, part_id, color, quantity)
            VALUES (?, ?, ?, ?)
            """, (set_id, part_id, color, qty))
            
        conn.commit()
        conn.close()
        print(f"[Pipeline] ¡Modelo {set_id} procesado e integrado exitosamente en BD local!")
        return True
        
    except Exception as e:
        print(f"[Pipeline Error] Fallo en la ingesta del modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

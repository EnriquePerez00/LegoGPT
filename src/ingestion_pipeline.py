import os
import json
import sqlite3
import re
import numpy as np
import networkx as nx
import torch
import threading

db_write_lock = threading.Lock()

from src.mpd_parser import flatten_mpd
from src.parser import build_pyg_graph
from scripts.scrape_bricklink import convert_io_to_ldr
from src.validator import check_connection_optimized, obbs_touching, is_minifig_part, get_part_name_from_db

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
        
        # Default metadata variables
        set_id = "unknown"
        name = "Studio MOC"
        theme = "Studio MOC"
        parts_count = 0
        subassemblies_count = 0
        is_fully_connected = 0
        subassemblies_list = []
        parts = []
        working_file_path = None
        
        if file_path is None:
            # Handle display-only model metadata registration
            set_id = source_url.split("idModel=")[-1] if source_url else "unknown"
            name = "Studio MOC"
        else:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            
            if "Synthetic" in base_name or base_name.startswith("9999-"):
                print(f"[Pipeline] Omitiendo modelo sintético: {base_name}")
                return False
                
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
                
            # 3. Calculate structural connectivity & subassemblies (NetworkX) / PyG Graphs
            if parts_count > 2000:
                print(f"[Pipeline Warning] El modelo {set_id} tiene {parts_count} piezas (> 2000). Omitiendo análisis de conectividad, subensamblajes y generación de grafos PyG.")
                subassemblies_count = 0
                is_fully_connected = 0
                subassemblies_list = []
                
                # Generate basic dataset sequences without PyG graph
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
                print(f"[Pipeline] Archivos básicos procesados guardados exitosamente para {set_id}")
            else:
                print("[Pipeline] Analizando grafo de conectividad y subensamblajes...")
                G = nx.Graph()
                for idx in range(parts_count):
                    G.add_node(idx)
                    
                for i in range(parts_count):
                    for j in range(i + 1, parts_count):
                        if check_connection_optimized(parts[i], parts[j]):
                            G.add_edge(i, j)
                            
                components = list(nx.connected_components(G))
                
                # Post-processing fusion: merge small subassemblies (<= 5 parts) close to the main component
                if components:
                    largest_idx = max(range(len(components)), key=lambda idx: len(components[idx]))
                    main_comp = set(components[largest_idx])
                    
                    fused_components = [main_comp]
                    other_comps = [components[i] for i in range(len(components)) if i != largest_idx]
                    
                    for comp in other_comps:
                        if len(comp) <= 5:
                            # Verify no organic minifig components (e.g. torso, legs, head)
                            has_organic = False
                            for p_idx in comp:
                                part = parts[p_idx]
                                if is_minifig_part(part):
                                    name = get_part_name_from_db(part.part_id)
                                    if any(x in name.lower() or x in part.part_id.lower() for x in ["torso", "leg", "hips"]):
                                        has_organic = True
                                        break
                                        
                            if not has_organic:
                                # Check proximity to the main component using 8.0 LDU margin
                                is_close = False
                                for p_idx in comp:
                                    part_s = parts[p_idx]
                                    for m_idx in main_comp:
                                        part_m = parts[m_idx]
                                        if obbs_touching(part_s, part_m, margin=8.0):
                                            is_close = True
                                            break
                                    if is_close:
                                        break
                                        
                                if is_close:
                                    main_comp.update(comp)
                                    continue
                        
                        fused_components.append(comp)
                    components = fused_components
                    
                subassemblies_count = len(components)
                is_fully_connected = 1 if subassemblies_count == 1 else 0
                
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
                    
                if subassemblies_list:
                    max_parts = max(sub["parts_count"] for sub in subassemblies_list)
                    has_marked_main = False
                    for sub in subassemblies_list:
                        if sub["parts_count"] == max_parts and not has_marked_main:
                            sub["is_main"] = 1
                            has_marked_main = True
                        else:
                            sub["is_main"] = 0
                    
                # 4. Generate dataset sequences
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
                print(f"[Pipeline] Archivos procesados guardados exitosamente para {set_id}")
                
            # Parse LDraw comments/headers for name/theme
            theme = "Unknown"
            name = set_id
            try:
                with open(working_file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
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
            
        # 5. Populate local SQLite database
        print(f"[Pipeline] Registrando metadatos en la base de datos local: {db_path}")
        with db_write_lock:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
                
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
            
            # Delete old subassemblies and inventory for this set to avoid stale records
            cursor.execute("DELETE FROM subassemblies WHERE set_id = ?", (set_id,))
            cursor.execute("DELETE FROM parts_inventory WHERE set_id = ?", (set_id,))
            
            # Insert subassemblies
            for sub in subassemblies_list:
                cursor.execute("""
                INSERT OR REPLACE INTO subassemblies (set_id, subassembly_id, parts_count, parts_list, is_grounded, is_main)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    set_id,
                    sub["subassembly_id"],
                    sub["parts_count"],
                    sub["parts_list"],
                    sub["is_grounded"],
                    sub.get("is_main", 0)
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

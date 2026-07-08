import os
import re
import sqlite3
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.mpd_parser import flatten_mpd

# Regex patterns to extract standard LDraw headers
THEME_PATTERN = re.compile(r'^0\s+!THEME\s+(.+)$', re.IGNORECASE)
NAME_PATTERN = re.compile(r'^0\s+!NAME\s+(.+)$', re.IGNORECASE)
AUTHOR_PATTERN = re.compile(r'^0\s+!AUTHOR\s+(.+)$', re.IGNORECASE)

def parse_file_metadata(filepath: str) -> dict:
    """
    Parses LDraw/MPD file, flattening it and extracting header metadata.
    Designed to run in parallel worker processes.
    """
    filename = os.path.basename(filepath)
    set_id = os.path.splitext(filename)[0]
    
    # Read headers from the first 50 lines to find metadata fast
    name = set_id
    theme = "Unknown"
    author = "Unknown"
    
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                
                theme_match = THEME_PATTERN.match(line)
                if theme_match:
                    theme = theme_match.group(1)
                    continue
                    
                name_match = NAME_PATTERN.match(line)
                if name_match:
                    name = name_match.group(1)
                    continue
                    
                auth_match = AUTHOR_PATTERN.match(line)
                if auth_match:
                    author = auth_match.group(1)
                    continue
                    
        # Flatten the model to get actual parts list and count
        parts = flatten_mpd(filepath)
        parts_count = len(parts)
        
        # Build inventory counts
        inventory = {}
        for p in parts:
            key = (p.part_id.lower(), p.color)
            inventory[key] = inventory.get(key, 0) + 1
            
        # Format inventory list for transfer
        inv_list = [{"part_id": k[0], "color": k[1], "quantity": v} for k, v in inventory.items()]
        

            
        # Find standardized LDraw counterpart
        normalized_path = None
        std_candidates = [
            os.path.join("data/standardized", f"{set_id}.ldr"),
            os.path.join("data/standardized", f"omr_{set_id}.ldr"),
            os.path.join("data/standardized", f"{set_id}-1.ldr"),
        ]
        for c in std_candidates:
            if os.path.exists(c):
                normalized_path = c
                break
                
        return {
            "set_id": set_id,
            "name": name,
            "theme": theme,
            "author": author,
            "file_path": filepath,
            "normalized_file_path": normalized_path,
            "parts_count": parts_count,
            "inventory": inv_list,
            "success": True
        }
    except Exception as e:
        return {
            "set_id": set_id,
            "success": False,
            "error": str(e)
        }

def process_local_cache(raw_dir: str = "data/omr_raw", db_path: str = "data/catalog/models_catalog.db"):
    if not os.path.exists(raw_dir):
        print(f"Directorio {raw_dir} no existe.")
        return
        
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(('.mpd', '.ldr'))]
    print(f"Encontrados {len(files)} archivos locales en {raw_dir}. Iniciando indexación...")
    
    # Calculate CPU workers (12 CPU cores, reserve 4 to maintain Mac UI responsiveness)
    max_workers = max(1, os.cpu_count() - 4)
    print(f"Ejecutando con {max_workers} procesos concurrentes...")
    
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(parse_file_metadata, f): f for f in files}
        
        for future in futures:
            res = future.result()
            if res["success"]:
                records.append(res)
            else:
                print(f"Error parsing {res['set_id']}: {res.get('error')}")
                
    # Insert records into SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Insertando {len(records)} registros en la base de datos...")
    
    for r in records:
        try:
            # Determine source
            source = "BrickLink" if "bricklink" in r["file_path"].lower() else "OMR"
            
            # Insert set
            cursor.execute("""
            INSERT OR REPLACE INTO sets (
                set_id, name, theme, year, description, source, file_path, 
                normalized_file_path, parts_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["set_id"],
                r["name"],
                r["theme"],
                None, # Year
                f"Modeled by {r['author']}",
                source,
                r["file_path"],
                r["normalized_file_path"],
                r["parts_count"]
            ))
            
            # Insert inventory
            for inv in r["inventory"]:
                cursor.execute("""
                INSERT OR REPLACE INTO parts_inventory (set_id, part_id, color, quantity)
                VALUES (?, ?, ?, ?)
                """, (r["set_id"], inv["part_id"], inv["color"], inv["quantity"]))
        except Exception as e:
            print(f"Error insertando {r['set_id']}: {e}")
            
    conn.commit()
    conn.close()
    print("¡Indexación local del catálogo completada exitosamente!")

if __name__ == "__main__":
    process_local_cache()

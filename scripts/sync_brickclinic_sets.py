import os
import sys
import json
import sqlite3
import urllib.request
import urllib.error
import time
import psutil

DB_PATH = "data/catalog/models_catalog.db"
BRICKCLINIC_URL = os.getenv("BRICKCLINIC_URL", "http://127.0.0.1:54341/rest/v1/sets")

def get_service_role_key() -> str:
    key = os.getenv("BRICKCLINIC_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not key and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("BRICKCLINIC_SERVICE_ROLE_KEY=") or line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
                    key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        break
    return key

def sync_brickclinic_sets(batch_size: int = 1000):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Pre-load theme mappings for fast in-memory joins
    cursor.execute("SELECT id, name FROM rb_themes;")
    theme_map = {row[0]: row[1] for row in cursor.fetchall()}
    
    print(f"[Brickclinic Sync] Conectando a {BRICKCLINIC_URL}...")
    service_role_key = get_service_role_key()
    
    offset = 0
    total_synced = 0
    total_parts_mapped = 0
    start_time = time.time()
    
    while True:
        # Check memory safety buffer (Apple Silicon 48GB M4)
        mem = psutil.virtual_memory()
        if mem.percent > 85.0:
            print(f"[Aviso de Memoria] Uso de RAM al {mem.percent}%. Pausando 1s...")
            time.sleep(1.0)
            
        url = f"{BRICKCLINIC_URL}?year=gte.2004&num_parts=gt.10&order=set_num.asc"
        headers = {
            "Range": f"{offset}-{offset + batch_size - 1}",
            "Prefer": "count=exact"
        }
        if service_role_key:
            headers["apikey"] = service_role_key
            headers["Authorization"] = f"Bearer {service_role_key}"

        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content_range = resp.headers.get("content-range", "")
                
                if not data:
                    break
                    
                total_records = int(content_range.split("/")[-1]) if "/" in content_range else "?"
                
                # Prepare batch inserts for sets
                sets_to_insert = []
                set_nums = []
                
                for item in data:
                    s_num = item.get("set_num")
                    if not s_num:
                        continue
                    set_nums.append(s_num)
                    name = item.get("name", "")
                    year = item.get("year", 0)
                    theme_id = item.get("theme_id")
                    num_parts = item.get("num_parts", 0)
                    img_url = item.get("img_url", "")
                    theme_name = theme_map.get(theme_id, f"Theme {theme_id}")
                    
                    sets_to_insert.append((
                        s_num,
                        name,
                        theme_name,
                        year,
                        f"LEGO {name} ({s_num}) - Sync from Brickclinic local DB",
                        "brickclinic_local",
                        num_parts,
                        f"https://rebrickable.com/sets/{s_num}/",
                        img_url,
                        1
                    ))
                    
                # Bulk insert into 'sets'
                cursor.executemany("""
                    INSERT INTO sets (
                        set_id, name, theme, year, description, source, parts_count,
                        source_url, image_url, is_fully_connected
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(set_id) DO UPDATE SET
                        name = excluded.name,
                        theme = excluded.theme,
                        year = excluded.year,
                        parts_count = excluded.parts_count,
                        image_url = excluded.image_url,
                        source = excluded.source
                """, sets_to_insert)
                
                # Link parts_inventory from local rb_inventories / rb_inventory_parts
                # for sets that don't have parts_inventory populated yet
                placeholders = ",".join(["?"] * len(set_nums))
                cursor.execute(f"""
                    INSERT OR IGNORE INTO parts_inventory (set_id, part_id, color, quantity)
                    SELECT inv.set_num, rip.part_num, rip.color_id, rip.quantity
                    FROM rb_inventories inv
                    JOIN rb_inventory_parts rip ON inv.id = rip.inventory_id
                    WHERE inv.set_num IN ({placeholders})
                """, set_nums)
                
                parts_inserted = cursor.rowcount
                total_parts_mapped += max(0, parts_inserted)
                
                conn.commit()
                
                batch_count = len(data)
                total_synced += batch_count
                offset += batch_count
                
                elapsed = time.time() - start_time
                rate = total_synced / max(0.1, elapsed)
                print(f"[{total_synced}/{total_records}] sets sincronizados ({rate:.1f} sets/s)...")
                
                if batch_count < batch_size:
                    break
                    
        except Exception as e:
            print(f"[Error en lote {offset}] {e}")
            break
            
    conn.close()
    elapsed = time.time() - start_time
    print(f"\n[Completado] {total_synced} sets sincronizados en models_catalog.db en {elapsed:.2f}s.")
    print(f"Total de registros de piezas vinculados: {total_parts_mapped}.")

if __name__ == "__main__":
    sync_brickclinic_sets()

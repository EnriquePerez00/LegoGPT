import os
import sys
import json
import sqlite3
import urllib.request
import urllib.error
import time

DB_PATH = "data/catalog/models_catalog.db"

def get_rebrickable_api_key():
    key = os.getenv("REBRICKABLE_API_KEY", "")
    if not key and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("REBRICKABLE_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    break
    return key

def sync_parent_set(set_id: str, db_conn: sqlite3.Connection):
    cursor = db_conn.cursor()
    
    # 1. Check if set exists in rb_sets
    cursor.execute("""
        SELECT s.set_num, s.name, s.year, s.theme_id, s.num_parts, s.img_url, t.name
        FROM rb_sets s
        LEFT JOIN rb_themes t ON s.theme_id = t.id
        WHERE s.set_num = ?
    """, (set_id,))
    row = cursor.fetchone()
    
    if not row:
        print(f"[Error] Set {set_id} no se encuentra en rb_sets.")
        return False
        
    set_num, name, year, theme_id, num_parts, img_url, theme_name = row
    
    # 2. Insert or replace in 'sets' table
    cursor.execute("""
        INSERT INTO sets (
            set_id, name, theme, year, description, source, parts_count,
            source_url, image_url, is_fully_connected
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(set_id) DO UPDATE SET
            name = excluded.name,
            theme = excluded.theme,
            year = excluded.year,
            parts_count = excluded.parts_count,
            image_url = excluded.image_url
    """, (
        set_num,
        name,
        theme_name or f"Theme {theme_id}",
        year,
        f"Official LEGO set {name} ({set_num})",
        "rebrickable_official",
        num_parts,
        f"https://rebrickable.com/sets/{set_num}/",
        img_url,
        1
    ))
    
    # 3. Populate parts_inventory from rb_inventory_parts
    cursor.execute("SELECT id FROM rb_inventories WHERE set_num = ?", (set_id,))
    inv_rows = cursor.fetchall()
    total_parts_inserted = 0
    
    for inv in inv_rows:
        inv_id = inv[0]
        cursor.execute("""
            SELECT part_num, color_id, quantity
            FROM rb_inventory_parts
            WHERE inventory_id = ?
        """, (inv_id,))
        parts = cursor.fetchall()
        
        for part_num, color_id, qty in parts:
            cursor.execute("""
                INSERT OR REPLACE INTO parts_inventory (set_id, part_id, color, quantity)
                VALUES (?, ?, ?, ?)
            """, (set_id, part_num, color_id, qty))
            total_parts_inserted += qty
            
    db_conn.commit()
    print(f"[Éxito] Set padre {set_id} ({name}) sincronizado en BD con {total_parts_inserted} piezas en inventario.")
    return True

def fetch_rebrickable_alternates(set_id: str, api_key: str):
    url = f"https://rebrickable.com/api/v3/lego/sets/{set_id}/alternates/?page_size=100"
    req = urllib.request.Request(url, headers={
        "Authorization": f"key {api_key}",
        "Accept": "application/json",
        "User-Agent": "LegoGPT/1.0"
    })
    
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", [])
    except Exception as e:
        print(f"[Error API Rebrickable] {e}")
        return []

def sync_alternates_to_db(set_id: str, alternates: list, db_conn: sqlite3.Connection):
    cursor = db_conn.cursor()
    inserted = 0
    
    for alt in alternates:
        moc_id = alt.get("set_num")
        name = alt.get("name")
        num_parts = alt.get("num_parts", 0)
        moc_url = alt.get("moc_url")
        designer_name = alt.get("designer_name")
        
        cursor.execute("""
            INSERT INTO alternate_builds (
                moc_id, parent_set_id, name, source, creator_username,
                num_parts, source_url, download_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(moc_id) DO UPDATE SET
                name = excluded.name,
                num_parts = excluded.num_parts,
                creator_username = excluded.creator_username,
                source_url = excluded.source_url
        """, (
            moc_id,
            set_id,
            name,
            "rebrickable_alt",
            designer_name,
            num_parts,
            moc_url
        ))
        inserted += 1
        
    db_conn.commit()
    print(f"[Éxito] {inserted} modelos alternativos registrados en 'alternate_builds' para el set {set_id}.")

def main():
    set_id = sys.argv[1] if len(sys.argv) > 1 else "75105-1"
    api_key = get_rebrickable_api_key()
    
    if not api_key:
        print("[Error] REBRICKABLE_API_KEY no encontrada.")
        sys.exit(1)
        
    conn = sqlite3.connect(DB_PATH)
    
    print(f"--- 1. Sincronizando Set Padre {set_id} ---")
    sync_parent_set(set_id, conn)
    
    print(f"\n--- 2. Obteniendo Alternate Builds para {set_id} vía API ---")
    alts = fetch_rebrickable_alternates(set_id, api_key)
    print(f"Total encontrados: {len(alts)}")
    
    print(f"\n--- 3. Guardando Alternate Builds en BD ---")
    sync_alternates_to_db(set_id, alts, conn)
    
    conn.close()

if __name__ == "__main__":
    main()

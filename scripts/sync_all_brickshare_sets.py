import os
import re
import sqlite3

SEED_PATH = "/Users/I764690/Code_personal/brickshare.org/supabase/seed.sql"
DB_PATH = "data/catalog/models_catalog.db"

def extract_sets_from_brickshare_seed():
    if not os.path.exists(SEED_PATH):
        raise FileNotFoundError(f"seed.sql not found at {SEED_PATH}")
        
    sets_data = []
    
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "INSERT INTO public.sets" in line:
                # Regex to extract set_ref and year_released
                # Pattern: set_ref is usually something like '10295-1', '42138-1'
                # year_released is an integer
                ref_match = re.search(r"'([0-9]+-[0-9]+)'", line)
                year_match = re.search(r",\s*(19\d\d|20\d\d)\s*,", line)
                name_match = re.search(r"VALUES\s*\('[^']+',\s*'([^']+)'", line)
                
                if ref_match:
                    set_ref = ref_match.group(1)
                    year = int(year_match.group(1)) if year_match else None
                    name = name_match.group(1) if name_match else ""
                    sets_data.append({
                        "set_ref": set_ref,
                        "year": year,
                        "name": name
                    })
                    
    return sets_data

def sync_brickshare_sets_to_catalog():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    sets = extract_sets_from_brickshare_seed()
    print(f"Extraídos {len(sets)} sets desde brickshare.org/supabase/seed.sql")
    
    # Filter year >= 2004
    filtered_sets = [s for s in sets if s["year"] is not None and s["year"] >= 2004]
    print(f"Sets con year >= 2004: {len(filtered_sets)}")
    
    synced_count = 0
    missing_in_rb = []
    
    for item in filtered_sets:
        set_ref = item["set_ref"]
        # Look up canonical info in rb_sets
        cursor.execute("""
            SELECT s.set_num, s.name, s.year, s.theme_id, s.num_parts, s.img_url, t.name
            FROM rb_sets s
            LEFT JOIN rb_themes t ON s.theme_id = t.id
            WHERE s.set_num = ?
        """, (set_ref,))
        row = cursor.fetchone()
        
        if row:
            s_num, s_name, s_year, s_theme_id, s_parts, s_img, s_theme_name = row
            cursor.execute("""
                INSERT INTO sets (
                    set_id, name, theme, year, description, source, parts_count,
                    source_url, image_url, is_fully_connected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(set_id) DO UPDATE SET
                    name = excluded.name,
                    theme = excluded.theme,
                    year = excluded.year,
                    parts_count = excluded.parts_count,
                    image_url = excluded.image_url
            """, (
                s_num,
                s_name,
                s_theme_name or f"Theme {s_theme_id}",
                s_year,
                f"LEGO {s_name} ({s_num}) - Sync from brickshare.org",
                "brickshare_official",
                s_parts,
                f"https://rebrickable.com/sets/{s_num}/",
                s_img
            ))
            
            # Sync parts inventory
            cursor.execute("SELECT id FROM rb_inventories WHERE set_num = ?", (s_num,))
            invs = cursor.fetchall()
            for inv in invs:
                inv_id = inv[0]
                cursor.execute("""
                    SELECT part_num, color_id, quantity
                    FROM rb_inventory_parts
                    WHERE inventory_id = ?
                """, (inv_id,))
                parts = cursor.fetchall()
                for p_num, c_id, qty in parts:
                    cursor.execute("""
                        INSERT OR REPLACE INTO parts_inventory (set_id, part_id, color, quantity)
                        VALUES (?, ?, ?, ?)
                    """, (s_num, p_num, c_id, qty))
                    
            synced_count += 1
        else:
            missing_in_rb.append(set_ref)
            
    conn.commit()
    conn.close()
    
    print(f"Sincronizados con éxito en models_catalog.db: {synced_count} sets.")
    if missing_in_rb:
        print(f"Sets no encontrados en rb_sets: {missing_in_rb}")

if __name__ == "__main__":
    sync_brickshare_sets_to_catalog()

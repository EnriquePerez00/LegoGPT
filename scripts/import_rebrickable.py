import os
import csv
import gzip
import sqlite3
import urllib.request
import tempfile
import ssl
from datetime import datetime

# Configuration
DB_PATH = "data/catalog/models_catalog.db"
BASE_URL = "https://rebrickable.com/media/downloads/"

TABLE_SCHEMAS = {
    "rb_colors": {
        "create": """
            CREATE TABLE rb_colors (
                id INTEGER PRIMARY KEY,
                name TEXT,
                rgb TEXT,
                is_trans INTEGER
            )
        """,
        "insert": "INSERT INTO rb_colors (id, name, rgb, is_trans) VALUES (?, ?, ?, ?)",
        "map_row": lambda r: (int(r[0]), r[1], r[2], 1 if r[3] == 't' else 0),
        "file": "colors.csv.gz"
    },
    "rb_themes": {
        "create": """
            CREATE TABLE rb_themes (
                id INTEGER PRIMARY KEY,
                name TEXT,
                parent_id INTEGER
            )
        """,
        "insert": "INSERT INTO rb_themes (id, name, parent_id) VALUES (?, ?, ?)",
        "map_row": lambda r: (int(r[0]), r[1], int(r[2]) if r[2] else None),
        "file": "themes.csv.gz"
    },
    "rb_part_categories": {
        "create": """
            CREATE TABLE rb_part_categories (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """,
        "insert": "INSERT INTO rb_part_categories (id, name) VALUES (?, ?)",
        "map_row": lambda r: (int(r[0]), r[1]),
        "file": "part_categories.csv.gz"
    },
    "rb_parts": {
        "create": """
            CREATE TABLE rb_parts (
                part_num TEXT PRIMARY KEY,
                name TEXT,
                part_cat_id INTEGER,
                part_material TEXT
            )
        """,
        "insert": "INSERT INTO rb_parts (part_num, name, part_cat_id, part_material) VALUES (?, ?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2]), r[3]),
        "file": "parts.csv.gz"
    },
    "rb_sets": {
        "create": """
            CREATE TABLE rb_sets (
                set_num TEXT PRIMARY KEY,
                name TEXT,
                year INTEGER,
                theme_id INTEGER,
                num_parts INTEGER,
                img_url TEXT
            )
        """,
        "insert": "INSERT INTO rb_sets (set_num, name, year, theme_id, num_parts, img_url) VALUES (?, ?, ?, ?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2]) if r[2] else None, int(r[3]) if r[3] else None, int(r[4]) if r[4] else 0, r[5]),
        "file": "sets.csv.gz"
    },
    "rb_minifigs": {
        "create": """
            CREATE TABLE rb_minifigs (
                fig_num TEXT PRIMARY KEY,
                name TEXT,
                num_parts INTEGER,
                img_url TEXT
            )
        """,
        "insert": "INSERT INTO rb_minifigs (fig_num, name, num_parts, img_url) VALUES (?, ?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2]) if r[2] else 0, r[3]),
        "file": "minifigs.csv.gz"
    },
    "rb_inventories": {
        "create": """
            CREATE TABLE rb_inventories (
                id INTEGER PRIMARY KEY,
                version INTEGER,
                set_num TEXT
            )
        """,
        "insert": "INSERT INTO rb_inventories (id, version, set_num) VALUES (?, ?, ?)",
        "map_row": lambda r: (int(r[0]), int(r[1]), r[2]),
        "file": "inventories.csv.gz"
    },
    "rb_inventory_parts": {
        "create": """
            CREATE TABLE rb_inventory_parts (
                inventory_id INTEGER,
                part_num TEXT,
                color_id INTEGER,
                quantity INTEGER,
                is_spare INTEGER,
                img_url TEXT
            )
        """,
        "insert": "INSERT INTO rb_inventory_parts (inventory_id, part_num, color_id, quantity, is_spare, img_url) VALUES (?, ?, ?, ?, ?, ?)",
        "map_row": lambda r: (int(r[0]), r[1], int(r[2]), int(r[3]), 1 if r[4] == 't' else 0, r[5]),
        "file": "inventory_parts.csv.gz"
    },
    "rb_inventory_minifigs": {
        "create": """
            CREATE TABLE rb_inventory_minifigs (
                inventory_id INTEGER,
                fig_num TEXT,
                quantity INTEGER
            )
        """,
        "insert": "INSERT INTO rb_inventory_minifigs (inventory_id, fig_num, quantity) VALUES (?, ?, ?)",
        "map_row": lambda r: (int(r[0]), r[1], int(r[2])),
        "file": "inventory_minifigs.csv.gz"
    },
    "rb_elements": {
        "create": """
            CREATE TABLE rb_elements (
                element_id TEXT PRIMARY KEY,
                part_num TEXT,
                color_id INTEGER
            )
        """,
        "insert": "INSERT INTO rb_elements (element_id, part_num, color_id) VALUES (?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2])),
        "file": "elements.csv.gz"
    }
}

def download_file(filename, dest_path):
    url = f"{BASE_URL}{filename}"
    print(f"Descargando {url}...")
    
    # Use headers to mimic browser request in case rebrickable blocks basic urllib user-agents
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    )
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=context) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())

def import_table_from_gz(conn, table_name, schema_info, gz_path):
    print(f"Importando datos a la tabla {table_name} desde {os.path.basename(gz_path)}...")
    cursor = conn.cursor()
    
    # Recreate the table
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(schema_info["create"])
    
    # Read the gzipped CSV file
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        
        batch = []
        batch_size = 50000
        count = 0
        
        for row in reader:
            try:
                mapped = schema_info["map_row"](row)
                batch.append(mapped)
                count += 1
            except (ValueError, IndexError) as e:
                # Skip malformed/incomplete records, log occasionally
                if count % 10000 == 0:
                    print(f"Advertencia al mapear fila en {table_name}: {row} -> {e}")
                continue
                
            if len(batch) >= batch_size:
                cursor.executemany(schema_info["insert"], batch)
                conn.commit()
                batch = []
                print(f"  Ingestados {count} registros...")
                
        if batch:
            cursor.executemany(schema_info["insert"], batch)
            conn.commit()
            
    print(f"Completado: Tabla {table_name} poblada con {count} registros.")

def create_indexes(conn):
    print("Creando índices para optimizar búsquedas y JOINs...")
    cursor = conn.cursor()
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_rb_parts_cat ON rb_parts(part_cat_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_sets_theme ON rb_sets(theme_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inv_parts_id ON rb_inventory_parts(inventory_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inv_parts_num ON rb_inventory_parts(part_num)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inv_minifigs_id ON rb_inventory_minifigs(inventory_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_elements_part ON rb_elements(part_num)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inventories_set ON rb_inventories(set_num)"
    ]
    
    for idx in indexes:
        cursor.execute(idx)
    conn.commit()
    print("Índices creados exitosamente.")

def main():
    start_time = datetime.now()
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Increase SQLite speed
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for table_name, schema_info in TABLE_SCHEMAS.items():
                filename = schema_info["file"]
                local_path = os.path.join(tmpdir, filename)
                
                try:
                    download_file(filename, local_path)
                    import_table_from_gz(conn, table_name, schema_info, local_path)
                except Exception as e:
                    print(f"Error al procesar la tabla {table_name}: {e}")
                    # Continue with next table
            
            # Create indexes
            create_indexes(conn)
            
    finally:
        conn.close()
        
    duration = datetime.now() - start_time
    print(f"Proceso finalizado en: {duration}")

if __name__ == "__main__":
    main()

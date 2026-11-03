import os
import csv
import gzip
import sqlite3
import urllib.request
import tempfile
import ssl
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
DB_PATH = "data/catalog/models_catalog.db"
BASE_URLS = [
    "https://cdn.rebrickable.com/media/downloads/",
    "https://rebrickable.com/media/downloads/"
]

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
        "insert": "INSERT OR REPLACE INTO rb_colors (id, name, rgb, is_trans) VALUES (?, ?, ?, ?)",
        "map_row": lambda r: (int(r[0]), r[1], r[2], 1 if str(r[3]).lower() in ('t', 'true', '1') else 0),
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
        "insert": "INSERT OR REPLACE INTO rb_themes (id, name, parent_id) VALUES (?, ?, ?)",
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
        "insert": "INSERT OR REPLACE INTO rb_part_categories (id, name) VALUES (?, ?)",
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
        "insert": "INSERT OR REPLACE INTO rb_parts (part_num, name, part_cat_id, part_material) VALUES (?, ?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2]) if r[2] else None, r[3] if len(r) > 3 else None),
        "file": "parts.csv.gz"
    },
    "rb_part_relationships": {
        "create": """
            CREATE TABLE rb_part_relationships (
                rel_type TEXT,
                child_part_num TEXT,
                parent_part_num TEXT
            )
        """,
        "insert": "INSERT INTO rb_part_relationships (rel_type, child_part_num, parent_part_num) VALUES (?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], r[2]),
        "file": "part_relationships.csv.gz"
    },
    "rb_elements": {
        "create": """
            CREATE TABLE rb_elements (
                element_id TEXT PRIMARY KEY,
                part_num TEXT,
                color_id INTEGER,
                design_id TEXT
            )
        """,
        "insert": "INSERT OR REPLACE INTO rb_elements (element_id, part_num, color_id, design_id) VALUES (?, ?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2]) if r[2] else None, r[3] if len(r) > 3 else None),
        "file": "elements.csv.gz"
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
        "insert": "INSERT OR REPLACE INTO rb_sets (set_num, name, year, theme_id, num_parts, img_url) VALUES (?, ?, ?, ?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2]) if r[2] else None, int(r[3]) if r[3] else None, int(r[4]) if r[4] else 0, r[5] if len(r) > 5 else None),
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
        "insert": "INSERT OR REPLACE INTO rb_minifigs (fig_num, name, num_parts, img_url) VALUES (?, ?, ?, ?)",
        "map_row": lambda r: (r[0], r[1], int(r[2]) if r[2] else 0, r[3] if len(r) > 3 else None),
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
        "insert": "INSERT OR REPLACE INTO rb_inventories (id, version, set_num) VALUES (?, ?, ?)",
        "map_row": lambda r: (int(r[0]), int(r[1]) if r[1] else 1, r[2]),
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
        "map_row": lambda r: (int(r[0]), r[1], int(r[2]), int(r[3]), 1 if str(r[4]).lower() in ('t', 'true', '1') else 0, r[5] if len(r) > 5 else None),
        "file": "inventory_parts.csv.gz"
    },
    "rb_inventory_sets": {
        "create": """
            CREATE TABLE rb_inventory_sets (
                inventory_id INTEGER,
                set_num TEXT,
                quantity INTEGER
            )
        """,
        "insert": "INSERT INTO rb_inventory_sets (inventory_id, set_num, quantity) VALUES (?, ?, ?)",
        "map_row": lambda r: (int(r[0]), r[1], int(r[2])),
        "file": "inventory_sets.csv.gz"
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
    }
}


def check_memory_safety(min_avail_gb: float = 4.0) -> int:
    """Checks available system RAM on macOS and returns safe batch size."""
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)
    if avail_gb < min_avail_gb:
        return 15000
    return 75000


def download_file(filename: str, dest_path: str) -> str:
    """Downloads a gzipped CSV from Rebrickable CDN with fallback URLs."""
    context = ssl._create_unverified_context()
    last_err = None
    for base_url in BASE_URLS:
        url = f"{base_url}{filename}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                }
            )
            with urllib.request.urlopen(req, context=context, timeout=60) as response, open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
            print(f"  [OK] Descargado {filename} ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")
            return dest_path
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No se pudo descargar {filename}: {last_err}")


def import_table_from_gz(conn: sqlite3.Connection, table_name: str, schema_info: dict, gz_path: str):
    print(f"Importando datos a la tabla {table_name} desde {os.path.basename(gz_path)}...")
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(schema_info["create"])

    batch_size = check_memory_safety()
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header

        batch = []
        count = 0

        for row in reader:
            try:
                mapped = schema_info["map_row"](row)
                batch.append(mapped)
                count += 1
            except (ValueError, IndexError) as e:
                if count % 25000 == 0:
                    print(f"Advertencia al mapear fila en {table_name}: {row} -> {e}")
                continue

            if len(batch) >= batch_size:
                cursor.executemany(schema_info["insert"], batch)
                conn.commit()
                batch = []
                batch_size = check_memory_safety()

        if batch:
            cursor.executemany(schema_info["insert"], batch)
            conn.commit()

    print(f"Completado: Tabla {table_name} poblada con {count:,} registros.")


def create_indexes(conn: sqlite3.Connection):
    print("Creando índices para optimizar búsquedas y JOINs...")
    cursor = conn.cursor()

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_rb_parts_cat ON rb_parts(part_cat_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_part_rel_child ON rb_part_relationships(child_part_num)",
        "CREATE INDEX IF NOT EXISTS idx_rb_part_rel_parent ON rb_part_relationships(parent_part_num)",
        "CREATE INDEX IF NOT EXISTS idx_rb_sets_theme ON rb_sets(theme_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_sets_filter ON rb_sets(year, num_parts)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inv_parts_id ON rb_inventory_parts(inventory_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inv_parts_num ON rb_inventory_parts(part_num)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inv_sets_id ON rb_inventory_sets(inventory_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inv_minifigs_id ON rb_inventory_minifigs(inventory_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_elements_part ON rb_elements(part_num)",
        "CREATE INDEX IF NOT EXISTS idx_rb_elements_design ON rb_elements(design_id)",
        "CREATE INDEX IF NOT EXISTS idx_rb_inventories_set ON rb_inventories(set_num)"
    ]

    for idx in indexes:
        cursor.execute(idx)
    conn.commit()
    print("Índices creados exitosamente.")


def sync_target_sets_to_catalog(conn: sqlite3.Connection, min_parts: int = 11, max_parts: int = 149, min_year: int = 2004):
    """
    Populates `sets`, `parts_inventory`, `figures`, and `set_images` for all sets
    in `rb_sets` (and existing `sets`) meeting:
      num_parts > 10 AND num_parts < 150 AND year >= 2004
    """
    print(f"\nSincronizando tablas de catálogo (sets, parts_inventory, figures, set_images) "
          f"para sets con piezas entre {min_parts} y {max_parts} y año >= {min_year}...")
    cursor = conn.cursor()

    # 1. Insert/update qualifying sets from rb_sets into sets
    cursor.execute("""
        INSERT INTO sets (set_id, name, theme, year, description, source, parts_count, image_url)
        SELECT
            s.set_num,
            s.name,
            COALESCE(t.name, 'Unknown'),
            s.year,
            'Official LEGO Set imported from Rebrickable (' || s.year || ')',
            'rebrickable_official',
            s.num_parts,
            s.img_url
        FROM rb_sets s
        LEFT JOIN rb_themes t ON s.theme_id = t.id
        WHERE s.num_parts >= ? AND s.num_parts <= ? AND s.year >= ?
        ON CONFLICT(set_id) DO UPDATE SET
            name = COALESCE(sets.name, excluded.name),
            theme = CASE WHEN sets.theme IS NULL OR sets.theme = 'Unknown' THEN excluded.theme ELSE sets.theme END,
            year = COALESCE(sets.year, excluded.year),
            parts_count = COALESCE(sets.parts_count, excluded.parts_count),
            image_url = COALESCE(sets.image_url, excluded.image_url)
    """, (min_parts, max_parts, min_year))
    conn.commit()

    # Count qualifying sets
    cursor.execute("""
        SELECT COUNT(*) FROM rb_sets
        WHERE num_parts >= ? AND num_parts <= ? AND year >= ?
    """, (min_parts, max_parts, min_year))
    qualifying_count = cursor.fetchone()[0]
    print(f"  -> {qualifying_count:,} sets cumplen (num_parts > 10 y < 150, year >= {min_year}).")

    # 2. Populate set_images for qualifying sets
    cursor.execute("""
        INSERT OR IGNORE INTO set_images (set_id, image_url, source)
        SELECT set_num, img_url, 'rebrickable'
        FROM rb_sets
        WHERE num_parts >= ? AND num_parts <= ? AND year >= ?
          AND img_url IS NOT NULL AND img_url != ''
    """, (min_parts, max_parts, min_year))
    conn.commit()

    # 3. Create temp table with latest inventory_id per qualifying set_num
    cursor.execute("DROP TABLE IF EXISTS temp_latest_set_inv")
    cursor.execute("""
        CREATE TEMP TABLE temp_latest_set_inv AS
        SELECT i.set_num, MAX(i.id) AS inventory_id
        FROM rb_inventories i
        JOIN rb_sets s ON i.set_num = s.set_num
        WHERE s.num_parts >= ? AND s.num_parts <= ? AND s.year >= ?
        GROUP BY i.set_num
    """, (min_parts, max_parts, min_year))
    cursor.execute("CREATE INDEX idx_temp_latest_inv_id ON temp_latest_set_inv(inventory_id)")
    cursor.execute("CREATE INDEX idx_temp_latest_set_num ON temp_latest_set_inv(set_num)")

    # 4. Populate parts_inventory (aggregating non-spare quantities per set_id, part_num, color_id)
    cursor.execute("""
        INSERT OR REPLACE INTO parts_inventory (set_id, part_id, color, quantity)
        SELECT
            tl.set_num,
            ip.part_num,
            ip.color_id,
            SUM(ip.quantity)
        FROM temp_latest_set_inv tl
        JOIN rb_inventory_parts ip ON tl.inventory_id = ip.inventory_id
        WHERE ip.is_spare = 0
        GROUP BY tl.set_num, ip.part_num, ip.color_id
    """)
    conn.commit()

    cursor.execute("""
        SELECT COUNT(*) FROM parts_inventory pi
        JOIN temp_latest_set_inv tl ON pi.set_id = tl.set_num
    """)
    parts_rows = cursor.fetchone()[0]
    print(f"  -> {parts_rows:,} registros en 'parts_inventory' sincronizados para los sets filtrados.")

    # 5. Populate figures table from rb_inventory_minifigs + rb_minifigs
    cursor.execute("""
        INSERT OR REPLACE INTO figures (set_id, fig_id, name, quantity)
        SELECT
            tl.set_num,
            im.fig_num,
            COALESCE(m.name, im.fig_num),
            SUM(im.quantity)
        FROM temp_latest_set_inv tl
        JOIN rb_inventory_minifigs im ON tl.inventory_id = im.inventory_id
        LEFT JOIN rb_minifigs m ON im.fig_num = m.fig_num
        GROUP BY tl.set_num, im.fig_num
    """)
    conn.commit()

    cursor.execute("""
        SELECT COUNT(*) FROM figures f
        JOIN temp_latest_set_inv tl ON f.set_id = tl.set_num
    """)
    figs_rows = cursor.fetchone()[0]
    print(f"  -> {figs_rows:,} registros en 'figures' sincronizados para los sets filtrados.")


def main():
    start_time = datetime.now()
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Calculate worker threads with safety margin (12 cores -> reserve 4 cores)
    max_workers = max(1, (os.cpu_count() or 8) - 4)
    print(f"Descargando {len(TABLE_SCHEMAS)} tablas de Rebrickable en paralelo ({max_workers} hilos)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded_files = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_table = {
                executor.submit(download_file, info["file"], os.path.join(tmpdir, info["file"])): (t_name, info)
                for t_name, info in TABLE_SCHEMAS.items()
            }
            for future in as_completed(future_to_table):
                t_name, info = future_to_table[future]
                try:
                    gz_path = future.result()
                    downloaded_files[t_name] = (info, gz_path)
                except Exception as e:
                    print(f"Error descargando {t_name}: {e}")

        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA cache_size = -500000")  # ~500MB cache

            for table_name in TABLE_SCHEMAS.keys():
                if table_name in downloaded_files:
                    schema_info, gz_path = downloaded_files[table_name]
                    try:
                        import_table_from_gz(conn, table_name, schema_info, gz_path)
                    except Exception as e:
                        print(f"Error al procesar la tabla {table_name}: {e}")

            create_indexes(conn)
            sync_target_sets_to_catalog(conn, min_parts=11, max_parts=149, min_year=2004)
        finally:
            conn.close()

    duration = datetime.now() - start_time
    print(f"\nProceso de importación y sincronización finalizado en: {duration}")


if __name__ == "__main__":
    main()

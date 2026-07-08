import os
import sqlite3

def initialize_database(db_path: str = "data/catalog/models_catalog.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create sets table with rich metadata and taxonomy columns
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sets (
        set_id TEXT PRIMARY KEY,
        name TEXT,
        theme TEXT,
        year INTEGER,
        description TEXT,
        source TEXT,
        file_path TEXT,
        normalized_file_path TEXT,
        parts_count INTEGER,
        subassemblies_count INTEGER,
        is_fully_connected INTEGER,
        tags TEXT,
        likes_count INTEGER,
        downloads_count INTEGER,
        views_count INTEGER,
        creator_username TEXT,
        source_url TEXT,
        image_url TEXT,
        level_1_entorno TEXT,
        level_2_proposito TEXT,
        level_3_clase TEXT,
        level_4_escala TEXT,
        level_4_motorizacion TEXT,
        level_4_licencia TEXT,
        confidence_score REAL,
        reasoning_notes TEXT,
        needs_human_review INTEGER,
        review_table_payload TEXT
    )
    """)
    
    # Run migrations for existing databases
    cursor.execute("PRAGMA table_info(sets)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    
    migrations = [
        ("source_url", "TEXT"),
        ("image_url", "TEXT"),
        ("subassemblies_count", "INTEGER"),
        ("is_fully_connected", "INTEGER"),
        ("level_1_entorno", "TEXT"),
        ("level_2_proposito", "TEXT"),
        ("level_3_clase", "TEXT"),
        ("level_4_escala", "TEXT"),
        ("level_4_motorizacion", "TEXT"),
        ("level_4_licencia", "TEXT"),
        ("confidence_score", "REAL"),
        ("reasoning_notes", "TEXT"),
        ("needs_human_review", "INTEGER"),
        ("review_table_payload", "TEXT")
    ]
    
    for col_name, col_type in migrations:
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE sets ADD COLUMN {col_name} {col_type}")
            print(f"Migración: Columna '{col_name}' añadida a la tabla 'sets'.")
            
    # 2. Create parts_inventory table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parts_inventory (
        set_id TEXT,
        part_id TEXT,
        color INTEGER,
        quantity INTEGER,
        PRIMARY KEY (set_id, part_id, color),
        FOREIGN KEY (set_id) REFERENCES sets(set_id)
    )
    """)
    
    # 3. Create figures table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS figures (
        set_id TEXT,
        fig_id TEXT,
        name TEXT,
        quantity INTEGER,
        PRIMARY KEY (set_id, fig_id),
        FOREIGN KEY (set_id) REFERENCES sets(set_id)
    )
    """)
    
    # 4. Create bricklink_scraping_queue table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bricklink_scraping_queue (
        model_id TEXT PRIMARY KEY,
        status TEXT DEFAULT 'pending',
        attempts INTEGER DEFAULT 0,
        last_attempt_time TIMESTAMP,
        error_msg TEXT
    )
    """)
    
    # 5. Create subassemblies table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subassemblies (
        set_id TEXT,
        subassembly_id INTEGER,
        parts_count INTEGER,
        parts_list TEXT,
        is_grounded INTEGER,
        PRIMARY KEY (set_id, subassembly_id),
        FOREIGN KEY (set_id) REFERENCES sets(set_id)
    )
    """)
    
    # 6. Create set_images table for holding multiple images per set/MOC
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS set_images (
        set_id TEXT,
        image_url TEXT,
        source TEXT,
        PRIMARY KEY (set_id, image_url)
    )
    """)
    
    # Create indexes for high query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_id ON parts_inventory(part_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sets_theme ON sets(theme)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sets_source ON sets(source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON bricklink_scraping_queue(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_set_images_id ON set_images(set_id)")
    
    conn.commit()
    conn.close()
    print(f"Base de datos del catálogo inicializada correctamente (incluyendo metadatos enriquecidos) en: {db_path}")

if __name__ == "__main__":
    initialize_database()


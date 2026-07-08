import os
import sqlite3

def initialize_database(db_path: str = "data/catalog/models_catalog.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create sets table with rich metadata columns
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
        tags TEXT,
        likes_count INTEGER,
        downloads_count INTEGER,
        views_count INTEGER,
        creator_username TEXT,
        source_url TEXT,
        image_url TEXT
    )
    """)
    
    # Run migrations for existing databases
    cursor.execute("PRAGMA table_info(sets)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    if "source_url" not in existing_columns:
        cursor.execute("ALTER TABLE sets ADD COLUMN source_url TEXT")
        print("Migración: Columna 'source_url' añadida a la tabla 'sets'.")
    if "image_url" not in existing_columns:
        cursor.execute("ALTER TABLE sets ADD COLUMN image_url TEXT")
        print("Migración: Columna 'image_url' añadida a la tabla 'sets'.")
    
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
    
    # Create indexes for high query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_id ON parts_inventory(part_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sets_theme ON sets(theme)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sets_source ON sets(source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON bricklink_scraping_queue(status)")
    
    conn.commit()
    conn.close()
    print(f"Base de datos del catálogo inicializada correctamente (incluyendo metadatos enriquecidos) en: {db_path}")

if __name__ == "__main__":
    initialize_database()

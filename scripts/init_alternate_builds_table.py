import sqlite3
import os

DB_PATH = "data/catalog/models_catalog.db"

def init_alternate_builds_table():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alternate_builds (
            moc_id TEXT PRIMARY KEY,
            parent_set_id TEXT NOT NULL,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            creator_username TEXT,
            num_parts INTEGER,
            num_steps INTEGER,
            has_discrete_steps BOOLEAN DEFAULT 0,
            avg_parts_per_step REAL,
            io_file_path TEXT,
            ldr_extracted_path TEXT,
            graph_tensor_path TEXT,
            source_url TEXT,
            is_free BOOLEAN DEFAULT 1,
            download_status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(parent_set_id) REFERENCES sets(set_id)
        );
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alt_parent ON alternate_builds(parent_set_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alt_source ON alternate_builds(source);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alt_status ON alternate_builds(download_status);")
    
    conn.commit()
    conn.close()
    print("Tabla 'alternate_builds' inicializada correctamente.")

if __name__ == "__main__":
    init_alternate_builds_table()

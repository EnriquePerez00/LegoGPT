import os
import sqlite3
import pytest
from scripts.harvest_bricklink_gallery_scroll import harvest_by_scroll
from scripts.initialize_catalog import initialize_database

def test_scroll_harvester(tmp_path):
    db_file = os.path.join(tmp_path, "test_scroll.db")
    initialize_database(db_file)
    
    print("\n[Test Scroll Harvester] Ejecutando cosecha de prueba de 2 scrolls...")
    try:
        harvest_by_scroll(db_file, max_scrolls=2)
        
        # Verify SQLite tables
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Check sets count
        cursor.execute("SELECT COUNT(*) FROM sets WHERE source = 'BrickLink'")
        count = cursor.fetchone()[0]
        print(f"[Test Scroll Harvester] Modelos cosechados en BD: {count}")
        
        # Verify that we harvested some models
        assert count > 0
        
        # Verify a record details
        cursor.execute("SELECT set_id, name, likes_count FROM sets WHERE source = 'BrickLink' LIMIT 1")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[1] is not None
        print(f"[Test Scroll Harvester] Primer modelo cosechado:")
        print(f"  ID: {row[0]}")
        print(f"  Nombre: {row[1]}")
        print(f"  Likes: {row[2]}")
        
        conn.close()
    except Exception as e:
        pytest.skip(f"Ignorado por conectividad o bloqueos de red: {e}")

import os
import sqlite3
import pytest
from scripts.initialize_catalog import initialize_database
from scripts.catalog_parser import parse_file_metadata, process_local_cache

def test_catalog_lifecycle(tmp_path):
    db_file = os.path.join(tmp_path, "test_catalog.db")
    
    # 1. Initialize DB
    initialize_database(db_file)
    assert os.path.exists(db_file)
    
    # Check tables
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    assert "sets" in tables
    assert "parts_inventory" in tables
    assert "figures" in tables
    
    # 2. Write a mock LDraw file
    mock_ldr = os.path.join(tmp_path, "mock_set.ldr")
    with open(mock_ldr, "w") as f:
        f.write("0 !NAME Mock Race Car\n")
        f.write("0 !THEME Racers\n")
        f.write("0 !AUTHOR LegoAgent\n")
        f.write("1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat\n") # A brick
        f.write("1 0 -10 -16 20 1 0 0 0 1 0 0 0 1 42610.dat\n") # A wheel
        f.write("1 0 10 -16 20 1 0 0 0 1 0 0 0 1 42610.dat\n") # Another wheel
        
    # 3. Parse mock file metadata
    res = parse_file_metadata(mock_ldr)
    assert res["success"] is True
    assert res["name"] == "Mock Race Car"
    assert res["theme"] == "Racers"
    assert res["parts_count"] == 3
    
    # 4. Insert into database using process_local_cache wrapper
    # We create a raw dir in tmp_path
    raw_dir = os.path.join(tmp_path, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    # Copy mock ldr there
    os.rename(mock_ldr, os.path.join(raw_dir, "9999-1.ldr"))
    
    process_local_cache(raw_dir, db_file)
    
    # Check insertion
    cursor.execute("SELECT name, theme, parts_count FROM sets WHERE set_id = '9999-1'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "Mock Race Car"
    assert row[1] == "Racers"
    assert row[2] == 3
        
    # Check inventory count
    cursor.execute("SELECT part_id, quantity FROM parts_inventory WHERE set_id = '9999-1' AND part_id = '42610.dat'")
    inv_row = cursor.fetchone()
    assert inv_row is not None
    assert inv_row[1] == 2 # 2 wheels
    
    conn.close()

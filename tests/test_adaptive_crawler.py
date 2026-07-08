import os
import sqlite3
import pytest
from scripts.adaptive_bricklink_crawler import (
    AdaptiveCadenceController,
    get_next_pending_model,
    update_model_status
)
from scripts.initialize_catalog import initialize_database

def test_adaptive_cadence_success_scaling():
    # Start with initial delay = 60s
    controller = AdaptiveCadenceController(initial_delay=60.0, min_delay=30.0, success_threshold=3)
    
    assert controller.record_success() == 60.0
    assert controller.success_streak == 1
    assert controller.record_success() == 60.0
    assert controller.success_streak == 2
    assert controller.record_success() == 50.0 # Streak reached
    
    controller.record_success()
    controller.record_success()
    assert controller.record_success() == 40.0
    
    controller.record_success()
    controller.record_success()
    assert controller.record_success() == 30.0 # Floor reached

def test_adaptive_cadence_block_scaling():
    controller = AdaptiveCadenceController(initial_delay=60.0, max_delay=3600.0)
    controller.record_success()
    
    new_delay, cooldown = controller.record_block()
    assert controller.success_streak == 0
    assert new_delay == 120.0
    assert cooldown == 900.0

def test_database_queue_state_transitions(tmp_path):
    db_file = os.path.join(tmp_path, "test_catalog.db")
    initialize_database(db_file)
    
    # Insert mock pending model
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO bricklink_scraping_queue (model_id, status)
    VALUES ('BL_9999', 'pending')
    """)
    conn.commit()
    
    # 1. Fetch next pending
    next_id = get_next_pending_model(db_file)
    assert next_id == 'BL_9999'
    
    # 2. Update to completed
    update_model_status(db_file, 'BL_9999', 'completed')
    
    # 3. Queue should now be empty (returns None)
    assert get_next_pending_model(db_file) is None
    
    # Verify DB record
    cursor.execute("SELECT status, attempts FROM bricklink_scraping_queue WHERE model_id = 'BL_9999'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == 'completed'
    assert row[1] == 0
    
    # Test failed status transitions
    cursor.execute("UPDATE bricklink_scraping_queue SET status = 'pending' WHERE model_id = 'BL_9999'")
    conn.commit()
    
    update_model_status(db_file, 'BL_9999', 'failed', error_msg="Timeout")
    cursor.execute("SELECT status, attempts, error_msg FROM bricklink_scraping_queue WHERE model_id = 'BL_9999'")
    row2 = cursor.fetchone()
    assert row2[0] == 'failed'
    assert row2[1] == 1
    assert row2[2] == 'Timeout'
    
    conn.close()

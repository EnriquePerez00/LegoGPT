import sqlite3
import os

local_db_path = "data/catalog/models_catalog.db"
dump_db_path = "/Users/I764690/.gemini/antigravity/brain/c51f4324-6269-4f12-9b70-7ba37caea9bd/scratch/temp_dump.db"

def get_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]

def merge_data():
    if not os.path.exists(dump_db_path):
        print(f"Error: Dump database not found at {dump_db_path}")
        return
        
    print("Iniciando migración e integración incremental de datos...")
    
    local_conn = sqlite3.connect(local_db_path)
    local_cursor = local_conn.cursor()
    
    dump_conn = sqlite3.connect(dump_db_path)
    dump_cursor = dump_conn.cursor()
    
    # 1. Merge SETS table
    print("\n--- Integrando tabla 'sets' ---")
    local_sets_cols = get_columns(local_cursor, "sets")
    dump_sets_cols = get_columns(dump_cursor, "sets")
    
    # Columns we can transfer (intersection)
    common_cols = list(set(local_sets_cols).intersection(dump_sets_cols))
    print(f"Columnas comunes en 'sets': {common_cols}")
    
    dump_cursor.execute("SELECT * FROM sets")
    dump_sets = dump_cursor.fetchall()
    
    sets_inserted = 0
    sets_updated = 0
    
    for row in dump_sets:
        # Create a dict for the dump row
        row_dict = dict(zip(dump_sets_cols, row))
        set_id = row_dict["set_id"]
        
        # Check if exists in local
        local_cursor.execute("SELECT 1 FROM sets WHERE set_id = ?", (set_id,))
        exists = local_cursor.fetchone()
        
        if not exists:
            # Insert new record using common columns
            cols_str = ", ".join(common_cols)
            placeholders = ", ".join(["?"] * len(common_cols))
            vals = [row_dict[col] for col in common_cols]
            
            local_cursor.execute(f"INSERT INTO sets ({cols_str}) VALUES ({placeholders})", vals)
            sets_inserted += 1
        else:
            # Update columns that are new or changed: subassemblies_count, is_fully_connected
            update_cols = []
            update_vals = []
            for col in ["subassemblies_count", "is_fully_connected"]:
                if col in row_dict and row_dict[col] is not None:
                    # Check if local is NULL
                    local_cursor.execute(f"SELECT {col} FROM sets WHERE set_id = ?", (set_id,))
                    local_val = local_cursor.fetchone()[0]
                    if local_val is None:
                        update_cols.append(f"{col} = ?")
                        update_vals.append(row_dict[col])
                        
            if update_cols:
                update_vals.append(set_id)
                cols_str = ", ".join(update_cols)
                local_cursor.execute(f"UPDATE sets SET {cols_str} WHERE set_id = ?", update_vals)
                sets_updated += 1
                
    print(f"Tabla 'sets': {sets_inserted} insertados, {sets_updated} actualizados.")

    # 2. Merge PARTS_INVENTORY table
    print("\n--- Integrando tabla 'parts_inventory' ---")
    dump_cursor.execute("SELECT set_id, part_id, color, quantity FROM parts_inventory")
    parts_inserted = 0
    
    for set_id, part_id, color, quantity in dump_cursor.fetchall():
        local_cursor.execute(
            "SELECT quantity FROM parts_inventory WHERE set_id = ? AND part_id = ? AND color = ?",
            (set_id, part_id, color)
        )
        exists = local_cursor.fetchone()
        if not exists:
            local_cursor.execute(
                "INSERT INTO parts_inventory (set_id, part_id, color, quantity) VALUES (?, ?, ?, ?)",
                (set_id, part_id, color, quantity)
            )
            parts_inserted += 1
            
    print(f"Tabla 'parts_inventory': {parts_inserted} nuevos registros de inventario insertados.")

    # 3. Merge SUBASSEMBLIES table
    print("\n--- Integrando tabla 'subassemblies' ---")
    dump_cursor.execute("SELECT set_id, subassembly_id, parts_count, parts_list, is_grounded FROM subassemblies")
    sub_inserted = 0
    
    for set_id, sub_id, parts_count, parts_list, is_grounded in dump_cursor.fetchall():
        local_cursor.execute(
            "SELECT 1 FROM subassemblies WHERE set_id = ? AND subassembly_id = ?",
            (set_id, sub_id)
        )
        exists = local_cursor.fetchone()
        if not exists:
            local_cursor.execute(
                "INSERT INTO subassemblies (set_id, subassembly_id, parts_count, parts_list, is_grounded) VALUES (?, ?, ?, ?, ?)",
                (set_id, sub_id, parts_count, parts_list, is_grounded)
            )
            sub_inserted += 1
            
    print(f"Tabla 'subassemblies': {sub_inserted} subensamblajes insertados.")

    # 4. Merge BRICKLINK_SCRAPING_QUEUE table
    print("\n--- Integrando tabla 'bricklink_scraping_queue' ---")
    dump_cursor.execute("SELECT model_id, status, attempts, last_attempt_time, error_msg FROM bricklink_scraping_queue")
    queue_inserted = 0
    queue_updated = 0
    
    for model_id, status, attempts, last_attempt, error_msg in dump_cursor.fetchall():
        local_cursor.execute(
            "SELECT status, attempts FROM bricklink_scraping_queue WHERE model_id = ?",
            (model_id,)
        )
        row = local_cursor.fetchone()
        if not row:
            local_cursor.execute(
                "INSERT INTO bricklink_scraping_queue (model_id, status, attempts, last_attempt_time, error_msg) VALUES (?, ?, ?, ?, ?)",
                (model_id, status, attempts, last_attempt, error_msg)
            )
            queue_inserted += 1
        else:
            local_status, local_attempts = row
            # Update if the dump has a completed/failed status and the local is pending, or if the dump has more attempts
            should_update = False
            if status in ('completed', 'failed') and local_status == 'pending':
                should_update = True
            elif attempts > local_attempts:
                should_update = True
                
            if should_update:
                local_cursor.execute(
                    "UPDATE bricklink_scraping_queue SET status = ?, attempts = ?, last_attempt_time = ?, error_msg = ? WHERE model_id = ?",
                    (status, attempts, last_attempt, error_msg, model_id)
                )
                queue_updated += 1
                
    print(f"Tabla 'bricklink_scraping_queue': {queue_inserted} insertados, {queue_updated} actualizados.")

    # Commit changes
    local_conn.commit()
    local_conn.close()
    dump_conn.close()
    
    print("\nSincronización de datos completada con éxito.")

if __name__ == "__main__":
    merge_data()

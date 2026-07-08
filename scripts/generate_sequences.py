import os
import sqlite3
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from generative.llm_pipeline.standardizer import standardize_and_order_ldr

def process_single_set(args):
    """Worker function to process a single set."""
    set_id, input_path, output_path = args
    if not os.path.exists(input_path):
        return set_id, None, False, "Input file does not exist"
        
    try:
        # Run with our new physical disassembly planner
        success = standardize_and_order_ldr(input_path, output_path)
        if success:
            return set_id, output_path, True, None
        else:
            return set_id, None, False, "Standardizer returned False"
    except Exception as e:
        return set_id, None, False, str(e)

def main():
    db_path = "data/catalog/models_catalog.db"
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} does not exist.")
        return
        
    # 1. Fetch pending sets from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT set_id, file_path 
        FROM sets 
        WHERE normalized_file_path IS NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    
    total_pending = len(rows)
    print(f"Found {total_pending} sets pending physical sequence planning.")
    if total_pending == 0:
        print("Nothing to process.")
        return
        
    # Prepare task arguments
    tasks = []
    for set_id, file_path in rows:
        # Standardized output path
        output_dir = "data/standardized"
        output_path = os.path.join(output_dir, f"{set_id}.ldr")
        tasks.append((set_id, file_path, output_path))
        
    # 2. Run parallel execution
    # Use exactly 50% of the M4 CPU cores (6 concurrent processes)
    max_workers = 6
    print(f"Executing batch sequencing with {max_workers} concurrent processes...")
    
    success_count = 0
    fail_count = 0
    updates = []
    
    # We will write DB updates in batches of 50 to maximize throughput
    def commit_batch(batch_updates):
        db_conn = sqlite3.connect(db_path)
        db_cursor = db_conn.cursor()
        db_cursor.executemany("""
            UPDATE sets 
            SET normalized_file_path = ? 
            WHERE set_id = ?
        """, batch_updates)
        db_conn.commit()
        db_conn.close()
        
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_set, task): task for task in tasks}
        
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            set_id, out_path, success, err_msg = future.result()
            
            if success:
                success_count += 1
                updates.append((out_path, set_id))
            else:
                fail_count += 1
                print(f"[{idx+1}/{total_pending}] Failed set {set_id}: {err_msg}")
                
            # Perform batch database commits
            if len(updates) >= 50:
                commit_batch(updates)
                print(f"Progress: Committed {success_count} successful sequences so far...")
                updates = []
                
    # Commit remaining updates
    if updates:
        commit_batch(updates)
        
    print("\n" + "="*50)
    print("Batch sequence generation completed!")
    print(f"Successfully processed: {success_count} sets")
    print(f"Failed: {fail_count} sets")
    print("="*50)

if __name__ == "__main__":
    main()

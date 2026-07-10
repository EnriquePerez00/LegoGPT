import os
import sys
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification_pipeline import VehicleClassifierAgent, BricklinkGalleryExtractor, save_classification_to_db

DB_PATH = "data/catalog/models_catalog.db"
db_lock = threading.Lock()

def classify_single_set(agent, extractor, set_id):
    print(f"Classifying '{set_id}'...")
    try:
        # Extract metadata
        with db_lock:
            metadata = extractor.extract(set_id)
            
        if not metadata:
            print(f"  [-] Metadata extraction failed for '{set_id}'")
            return set_id, False
            
        # VLM inference
        result = agent.classify_design(metadata)
        
        # Save to DB
        with db_lock:
            save_classification_to_db(result)
            
        if hasattr(result.taxonomy_proposal, 'Level_1_Entorno'):
            print(f"  [+] Success: '{set_id}' classified as {result.taxonomy_proposal.Level_1_Entorno} / {result.taxonomy_proposal.Level_3_Clase}")
        else:
            print(f"  [+] Success: '{set_id}' classified as Animal: {result.taxonomy_proposal.Level_1_Habitat} / {result.taxonomy_proposal.Level_3_Especie}")
        return set_id, True
    except Exception as e:
        print(f"  [-] Error classifying '{set_id}': {e}")
        return set_id, False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=DB_PATH, help="Path to database")
    parser.add_argument("--model", type=str, default="qwen2.5vl:latest", help="Ollama model to use for classification")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    args = parser.parse_args()
    
    print(f"=== Batch Parallel VLM Classification Started ===")
    print(f"Model: {args.model}")
    print(f"Database: {args.db_path}")
    
    conn = sqlite3.connect(args.db_path, timeout=30.0)
    c = conn.cursor()
    
    # Query all BrickLink sets that are not human verified
    rows = c.execute("""
        SELECT set_id 
        FROM sets 
        WHERE source = 'BrickLink' AND (classification_status IS NULL OR classification_status != 'human_verified')
    """).fetchall()
    conn.close()
    
    pending_sets = [r[0] for r in rows]
    total_pending = len(pending_sets)
    print(f"Found {total_pending} non-human-verified BrickLink sets to classify.")
    
    if total_pending == 0:
        print("No pending sets to classify.")
        return
        
    # Initialize VLM agent and metadata extractor
    vlm_agent = VehicleClassifierAgent(model_name=args.model)
    bl_extractor = BricklinkGalleryExtractor()
    
    # Execute batch in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(classify_single_set, vlm_agent, bl_extractor, sid): sid for sid in pending_sets}
        for future in as_completed(futures):
            set_id, success = future.result()
            if success:
                success_count += 1
                
    print(f"\n=== Batch Classification Finished. Successfully classified {success_count}/{total_pending} sets ===")

if __name__ == "__main__":
    main()

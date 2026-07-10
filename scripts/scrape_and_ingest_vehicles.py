import os
import sys
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from scripts.download_bricklink_model import download_bricklink_model
from src.ingestion_pipeline import process_and_register_downloaded_model

DB_PATH = "data/catalog/models_catalog.db"
OUTPUT_DIR = "data/bricklink_raw"

def is_already_processed(model_id: str) -> bool:
    # 1. Check if the file already exists in raw dataset
    file_path = os.path.join(OUTPUT_DIR, f"{model_id}.io")
    if os.path.exists(file_path):
        print(f"[Skip check] El archivo {file_path} ya existe en raw dataset.")
        return True
        
    # 2. Check if the model is registered in the DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check sets table
    cursor.execute("SELECT set_id FROM sets WHERE set_id = ?", (model_id,))
    in_sets = cursor.fetchone() is not None
    
    # Check queue table for completed ones
    cursor.execute("SELECT model_id FROM bricklink_scraping_queue WHERE model_id = ? AND status = 'completed'", (model_id,))
    in_queue = cursor.fetchone() is not None
    
    conn.close()
    
    if in_sets or in_queue:
        print(f"[Skip check] El modelo {model_id} ya existe en la base de datos.")
        return True
        
    return False

def ingest_worker(model_id: str, save_path: str, image_url: str):
    print(f"[Ingest Thread] Iniciando ingesta en segundo plano para el modelo {model_id}...")
    source_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
    
    success = process_and_register_downloaded_model(
        file_path=save_path,
        source="BrickLink",
        source_url=source_url,
        image_url=image_url,
        db_path=DB_PATH
    )
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = ""
    try:
        from datetime import datetime
        now_str = datetime.now().isoformat()
    except Exception:
        pass
        
    if success:
        print(f"[Ingest Thread] Ingesta COMPLETADA con éxito para el modelo {model_id}.")
        cursor.execute("""
        INSERT OR REPLACE INTO bricklink_scraping_queue (model_id, status, last_attempt_time)
        VALUES (?, 'completed', ?)
        """, (model_id, now_str))
    else:
        print(f"[Ingest Thread] Ingesta FALLIDA para el modelo {model_id}.")
        cursor.execute("""
        INSERT OR REPLACE INTO bricklink_scraping_queue (model_id, status, last_attempt_time, error_msg)
        VALUES (?, 'failed', ?, 'Ingestion failed')
        """, (model_id, now_str))
        
    conn.commit()
    conn.close()

def main():
    targets_file = "data/bricklink_targets.json"
    if not os.path.exists(targets_file):
        print(f"Error: No se encontró {targets_file}. Ejecuta primero search_bricklink_gallery.py.")
        sys.exit(1)
        
    with open(targets_file, "r") as f:
        model_ids = json.load(f)
        
    print(f"Leídos {len(model_ids)} IDs objetivo de {targets_file}")
    
    # Filter targets
    targets_to_scrape = [mid for mid in model_ids if not is_already_processed(mid)]
    print(f"Modelos nuevos a descargar y procesar: {targets_to_scrape}")
    
    if not targets_to_scrape:
        print("Todos los modelos ya se encuentran en el dataset o en la base de datos.")
        return
        
    # We use a ThreadPoolExecutor to run ingestion pipelines in parallel
    # while the main thread downloads models sequentially to avoid bot blocks.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for idx, model_id in enumerate(targets_to_scrape):
            print(f"\n[{idx+1}/{len(targets_to_scrape)}] Iniciando descarga del modelo {model_id}...")
            
            # Record status in DB queue
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO bricklink_scraping_queue (model_id, status, attempts)
            VALUES (?, 'downloading', COALESCE((SELECT attempts FROM bricklink_scraping_queue WHERE model_id = ?), 0) + 1)
            """, (model_id, model_id))
            conn.commit()
            conn.close()
            
            success, image_url, save_path = download_bricklink_model(model_id, OUTPUT_DIR)
            
            if success and save_path:
                print(f"[Main Thread] Descarga completada. Lanzando ingesta en paralelo...")
                # Submit ingestion to executor (runs in background thread)
                executor.submit(ingest_worker, model_id, save_path, image_url)
            else:
                print(f"[Main Thread] Error en la descarga del modelo {model_id} (puede que no sea descargable o haya Cloudflare).")
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO bricklink_scraping_queue (model_id, status, error_msg)
                VALUES (?, 'failed', 'Download failed')
                """, (model_id,))
                conn.commit()
                conn.close()
                
            # Random delay between downloads to simulate human behavior
            import time
            import random
            delay = random.uniform(5.0, 10.0)
            print(f"[Main Thread] Esperando {delay:.2f} segundos antes del siguiente modelo...")
            time.sleep(delay)

    print("\nTodos los hilos de ingesta y descargas han finalizado.")

if __name__ == "__main__":
    main()

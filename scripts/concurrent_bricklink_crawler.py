import time
import os
import sys
import argparse
import sqlite3
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scripts.download_bricklink_model import download_bricklink_model
from src.ingestion_pipeline import process_and_register_downloaded_model

# Lock to ensure thread-safe operations on the SQLite database
db_lock = threading.Lock()

# Global events for managing WAF cooldown
cooldown_active = threading.Event()
active_workers_count = 3  # Start with 3 concurrent workers

class GlobalCadenceController:
    def __init__(self, initial_delay=3.0, min_delay=1.0):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.success_streak = 0
        self.success_threshold = 5
        self.lock = threading.Lock()
        
    def record_success(self):
        with self.lock:
            self.success_streak += 1
            if self.success_streak >= self.success_threshold:
                self.current_delay = max(self.min_delay, self.current_delay - 0.5)
                self.success_streak = 0
                print(f"[Cadence] Racha de éxitos lograda. Reduciendo delay entre hilos a {self.current_delay:.1f}s")
            return self.current_delay

    def record_block(self):
        with self.lock:
            self.success_streak = 0
            self.current_delay = min(60.0, self.current_delay * 2.0)
            return self.current_delay

cadence_controller = GlobalCadenceController()

def get_next_pending_model_safe(db_path: str) -> str:
    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT model_id FROM bricklink_scraping_queue 
        WHERE status = 'pending' 
        LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            model_id = row[0]
            # Immediately mark as downloading to prevent other threads from picking it up
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("""
            UPDATE bricklink_scraping_queue 
            SET status = 'downloading', last_attempt_time = ?, attempts = attempts + 1
            WHERE model_id = ?
            """, (now_str, model_id))
            conn.commit()
            conn.close()
            return model_id
        conn.close()
        return None

def update_model_status_safe(db_path: str, model_id: str, status: str, error_msg: str = None):
    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # If it's a specific conversion error due to password protection, mark as not_possible
        if error_msg and "encrypted" in error_msg.lower():
            status = 'not_possible'
            error_msg = 'Encrypted - not possible'
            
        cursor.execute("""
        UPDATE bricklink_scraping_queue 
        SET status = ?, last_attempt_time = ?, error_msg = ?
        WHERE model_id = ?
        """, (status, now_str, error_msg, model_id))
        conn.commit()
        conn.close()

def worker_task(worker_id: int, db_path: str, output_dir: str, stop_event: threading.Event):
    global active_workers_count
    
    print(f"[Worker-{worker_id}] Iniciado.")
    
    while not stop_event.is_set():
        # If a WAF block is active, wait
        if cooldown_active.is_set():
            print(f"[Worker-{worker_id}] Cooldown activo. Esperando...")
            time.sleep(5)
            continue
            
        model_id = get_next_pending_model_safe(db_path)
        if not model_id:
            print(f"[Worker-{worker_id}] No hay más modelos pendientes. Finalizando.")
            break
            
        print(f"[Worker-{worker_id}] Procesando modelo ID {model_id}...")
        
        # Simulate staggered starts
        time.sleep(cadence_controller.current_delay * worker_id)
        
        try:
            success, image_url, save_path = download_bricklink_model(model_id, output_dir, fast=True)
        except Exception as e:
            print(f"[Worker-{worker_id} Error] Excepción en descarga: {e}")
            success, image_url, save_path = False, None, None
            
        if success:
            if save_path:
                print(f"[Worker-{worker_id}] Descarga exitosa. Iniciando ingesta para {model_id}...")
                try:
                    pipeline_success = process_and_register_downloaded_model(
                        file_path=save_path,
                        source="BrickLink",
                        source_url=f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}",
                        image_url=image_url,
                        db_path=db_path
                    )
                    if pipeline_success:
                        update_model_status_safe(db_path, model_id, 'completed')
                        print(f"[Worker-{worker_id}] Ingesta completada con éxito para {model_id}.")
                    else:
                        update_model_status_safe(db_path, model_id, 'failed', error_msg="Ingestion pipeline failed")
                except Exception as ingest_err:
                    print(f"[Worker-{worker_id} Error] Fallo ingesta {model_id}: {ingest_err}")
                    update_model_status_safe(db_path, model_id, 'failed', error_msg=str(ingest_err))
            else:
                # Display only model
                update_model_status_safe(db_path, model_id, 'completed', error_msg="Display only")
                print(f"[Worker-{worker_id}] Modelo {model_id} es 'Display Only'. Registrado metadato.")
                
            cadence_controller.record_success()
        else:
            # Check if it was a redirection or block
            print(f"[Worker-{worker_id} WAF Detect] Fallo de descarga detectado para {model_id}.")
            update_model_status_safe(db_path, model_id, 'failed', error_msg="Download failed or blocked")
            
            # Activate global WAF cooldown and reduce concurrency dynamically
            if not cooldown_active.is_set():
                cooldown_active.set()
                cadence_controller.record_block()
                
                # Reduce concurrent workers count to prevent future WAF triggers (minimum 1 worker)
                if active_workers_count > 1:
                    active_workers_count -= 1
                    print(f"[WAF Alert] Reduciendo concurrencia a {active_workers_count} trabajadores.")
                    
                cooldown_seconds = 180.0  # 3 minutes cooldown for fast recovery
                print(f"[WAF Alert] Iniciando cooldown global de {cooldown_seconds}s...")
                
                # Thread to release the cooldown after time
                def release_cooldown():
                    time.sleep(cooldown_seconds)
                    cooldown_active.clear()
                    print("[WAF Alert] Cooldown finalizado. Reanudando trabajadores...")
                
                threading.Thread(target=release_cooldown, daemon=True).start()
                
        # Inter-loop delay
        time.sleep(cadence_controller.current_delay)

def run_concurrent_crawler(db_path: str, output_dir: str, limit_count: int, num_workers: int):
    global active_workers_count
    active_workers_count = num_workers
    
    stop_event = threading.Event()
    threads = []
    
    print(f"\n==================================================")
    print(f"INICIANDO CRAWLER CONCURRENTE EN PARALELO")
    print(f"Base de datos: {db_path}")
    print(f"Hilos trabajadores: {num_workers}")
    print(f"==================================================")
    
    for i in range(num_workers):
        t = threading.Thread(target=worker_task, args=(i+1, db_path, output_dir, stop_event), daemon=True)
        threads.append(t)
        t.start()
        time.sleep(2)  # Staggered starts
        
    try:
        # Keep main thread alive monitoring progress
        while True:
            # Check if any thread is alive
            alive = any(t.is_alive() for t in threads)
            if not alive:
                print("Todos los trabajadores han finalizado.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[Detener] Cancelando trabajadores...")
        stop_event.set()
        for t in threads:
            t.join()
        print("Todos los trabajadores detenidos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="data/catalog/models_catalog.db")
    parser.add_argument("--output_dir", type=str, default="data/bricklink_raw")
    parser.add_argument("--limit", type=int, default=3500)
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    args = parser.parse_args()
    
    run_concurrent_crawler(args.db_path, args.output_dir, args.limit, args.workers)

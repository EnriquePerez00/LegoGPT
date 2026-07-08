import time
import os
import sys
import argparse
import sqlite3
from datetime import datetime
from scripts.download_bricklink_model import download_bricklink_model

class AdaptiveCadenceController:
    def __init__(self, initial_delay=3.0, min_delay=1.0, max_delay=60.0, success_threshold=3):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.success_threshold = success_threshold
        self.success_streak = 0
        
    def record_success(self) -> float:
        self.success_streak += 1
        if self.success_streak >= self.success_threshold:
            self.current_delay = max(self.min_delay, self.current_delay - 1.0)
            self.success_streak = 0
            print(f"[Adaptive Cadence] ¡Racha de éxitos lograda! Reduciendo delay a {self.current_delay:.1f}s")
        return self.current_delay
        
    def record_block(self) -> tuple[float, float]:
        self.success_streak = 0
        self.current_delay = min(self.max_delay, self.current_delay * 2.0)
        cooldown = 15.0 # Short cooldown for fast testing
        print(f"[Adaptive WAF Alert] Bloqueo o timeout detectado. Incrementando delay a {self.current_delay:.1f}s.")
        print(f"[Adaptive WAF Alert] Iniciando cooldown de {cooldown:.1f} segundos...")
        return self.current_delay, cooldown

def get_next_pending_model(db_path: str) -> str:
    """Fetches the next pending model ID from the SQLite queue."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT model_id FROM bricklink_scraping_queue 
    WHERE status = 'pending' 
    LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def update_model_status(db_path: str, model_id: str, status: str, error_msg: str = None):
    """Updates the status and last attempt timestamp of a model in the queue."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    
    if status == 'completed':
        cursor.execute("""
        UPDATE bricklink_scraping_queue 
        SET status = 'completed', last_attempt_time = ?, error_msg = NULL
        WHERE model_id = ?
        """, (now_str, model_id))
    else:
        cursor.execute("""
        UPDATE bricklink_scraping_queue 
        SET status = ?, last_attempt_time = ?, attempts = attempts + 1, error_msg = ?
        WHERE model_id = ?
        """, (status, now_str, error_msg, model_id))
        
    conn.commit()
    conn.close()

def run_adaptive_db_queue(db_path: str = "data/catalog/models_catalog.db", output_dir: str = "data/bricklink_raw", limit_count: int = 15):
    controller = AdaptiveCadenceController()
    
    print(f"\nIniciando cola adaptativa desde la Base de Datos ({db_path}). Límite de sesión: {limit_count} descargas...")
    
    downloaded_this_session = 0
    
    while downloaded_this_session < limit_count:
        model_id = get_next_pending_model(db_path)
        
        if not model_id:
            print("[Cola Completada] No hay más modelos con estado 'pending' en la base de datos.")
            break
            
        print(f"\n--------------------------------------------------")
        print(f"Descargando [{downloaded_this_session + 1}/{limit_count}] de la base de datos: ID {model_id}")
        
        # Mark as downloading
        update_model_status(db_path, model_id, 'downloading')
        
        success, image_url, save_path = download_bricklink_model(model_id, output_dir, fast=True)
        
        if success:
            from src.ingestion_pipeline import process_and_register_downloaded_model
            source_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
            
            print(f"[Crawler] Iniciando pipeline de ingesta automatizada para {model_id}...")
            pipeline_success = process_and_register_downloaded_model(
                file_path=save_path,
                source="BrickLink",
                source_url=source_url,
                image_url=image_url,
                db_path=db_path
            )
            
            if pipeline_success:
                update_model_status(db_path, model_id, 'completed')
            else:
                update_model_status(db_path, model_id, 'failed', error_msg="Ingestion pipeline failed")
                
            downloaded_this_session += 1
            next_delay = controller.record_success()
            
            if downloaded_this_session < limit_count:
                print(f"[Bucle] Esperando {next_delay:.1f}s...")
                time.sleep(next_delay)
        else:
            update_model_status(db_path, model_id, 'failed', error_msg="Download failed or blocked")
            next_delay, cooldown = controller.record_block()
            
            if downloaded_this_session < limit_count:
                time.sleep(cooldown)
                print(f"[Bucle] Reanudando tras cooldown. Esperando delay de {next_delay:.1f}s...")
                time.sleep(next_delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="data/catalog/models_catalog.db", help="Path to SQLite catalog database")
    parser.add_argument("--output_dir", type=str, default="data/bricklink_raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=15, help="Max models to download in this session")
    args = parser.parse_args()
    
    run_adaptive_db_queue(args.db_path, args.output_dir, args.limit)

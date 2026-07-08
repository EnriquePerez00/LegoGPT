import os
import json
import sqlite3
import random
import time
from scripts.search_bricklink_gallery import search_gallery_for_targets
from scripts.download_bricklink_model import human_delay

THEMES_TO_HARVEST = [
    "car", "motorcycle", "truck", "train", "airplane",
    "helicopter", "ship", "star wars", "castle", "space",
    "technic", "creature", "robot", "city", "rover"
]

def harvest_and_register_ids(db_path: str = "data/catalog/models_catalog.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_added = 0
    print(f"Iniciando recolección masiva de IDs para {len(THEMES_TO_HARVEST)} temáticas...")
    
    for theme in THEMES_TO_HARVEST:
        print(f"\nBuscando en galería para temática: '{theme}'...")
        # Scrape top 20 models per theme
        ids = search_gallery_for_targets(theme, limit=20)
        
        added_for_theme = 0
        for model_id in ids:
            try:
                # INSERT OR IGNORE avoids duplicates
                cursor.execute("""
                INSERT OR IGNORE INTO bricklink_scraping_queue (model_id, status, attempts)
                VALUES (?, 'pending', 0)
                """, (model_id,))
                
                # Check if it was actually inserted
                if cursor.rowcount > 0:
                    added_for_theme += 1
            except Exception as e:
                print(f"Error registrando ID {model_id}: {e}")
                
        conn.commit()
        print(f"Registrados {added_for_theme} nuevos IDs para '{theme}'.")
        total_added += added_for_theme
        
        # Human delay between theme queries
        human_delay(6.0, 12.0)
        
    conn.close()
    print(f"\n[Catálogo Cola] ¡Cosecha completada! Se han añadido {total_added} nuevos modelos únicos a la cola persistente.")

if __name__ == "__main__":
    harvest_and_register_ids()

import os
import sys
import sqlite3
import re
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.omr_downloader import build_sets_index, _sets_index, search_set_metadata
from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

def resolve_omr_images():
    print("--- Resolviendo imágenes y referencias para OMR ---")
    
    # Ensure sets index is built
    build_sets_index()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all OMR sets missing image_url
    cursor.execute("""
    SELECT set_id FROM sets 
    WHERE source = 'OMR' AND (image_url IS NULL OR image_url = '')
    """)
    rows = cursor.fetchall()
    print(f"Encontrados {len(rows)} modelos OMR sin imagen en la BD.")
    
    if not rows:
        conn.close()
        return
        
    def fetch_omr_image(set_id):
        # Clean set_id to look up in OMR index (e.g., "8303-1" -> "8303")
        clean_id = set_id.split("-")[0]
        meta = search_set_metadata(clean_id)
        if meta:
            return set_id, meta.get("image_url"), meta.get("source_url")
        return set_id, None, None

    updated_count = 0
    # Process OMR image metadata in parallel using ThreadPoolExecutor
    print("Obteniendo detalles de modelos OMR desde la web...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_omr_image, r[0]): r for r in rows}
        
        batch = []
        for future in as_completed(futures):
            set_id, img_url, src_url = future.result()
            if img_url:
                batch.append((img_url, src_url, set_id))
                updated_count += 1
                
            if len(batch) >= 100:
                cursor.executemany("UPDATE sets SET image_url = ?, source_url = ? WHERE set_id = ?", batch)
                conn.commit()
                batch = []
                print(f"Progreso OMR: {updated_count} modelos actualizados...")
                
        if batch:
            cursor.executemany("UPDATE sets SET image_url = ?, source_url = ? WHERE set_id = ?", batch)
            conn.commit()
            
    conn.close()
    print(f"Finalizado OMR: {updated_count} imágenes y referencias actualizadas.")

def resolve_bricklink_images(limit=10):
    print("--- Resolviendo imágenes para BrickLink (usando Playwright) ---")
    
    from playwright.sync_api import sync_playwright
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT set_id FROM sets 
    WHERE source = 'BrickLink' AND (image_url IS NULL OR image_url = '')
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    print(f"Procesando {len(rows)} modelos BrickLink sin imagen en este lote...")
    
    if not rows:
        conn.close()
        return
        
    model_ids = [r[0] for r in rows]
    ua = random.choice(USER_AGENTS)
    
    updated_count = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        for model_id in model_ids:
            url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
            print(f"Navegando a {url}...")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                # Wait a bit for components
                page.wait_for_timeout(2000)
                
                # Check for meta tag og:image
                og_image_el = page.locator('meta[property="og:image"]')
                if og_image_el.count() > 0:
                    image_url = og_image_el.get_attribute("content")
                    if image_url:
                        cursor.execute("UPDATE sets SET image_url = ?, source_url = ? WHERE set_id = ?", (image_url, url, model_id))
                        conn.commit()
                        updated_count += 1
                        print(f"  [+] Actualizada imagen para BrickLink ID {model_id} -> {image_url}")
                        continue
                        
                # Fallback to search inside page DOM
                img_el = page.locator(".moc-card__image-content, img[src*='file.bricklink.info']").first
                if img_el.count() > 0:
                    image_url = img_el.get_attribute("src")
                    if image_url:
                        if not image_url.startswith("http"):
                            image_url = "https:" + image_url
                        cursor.execute("UPDATE sets SET image_url = ?, source_url = ? WHERE set_id = ?", (image_url, url, model_id))
                        conn.commit()
                        updated_count += 1
                        print(f"  [+] Actualizada imagen (DOM Fallback) para BrickLink ID {model_id} -> {image_url}")
            except Exception as e:
                print(f"Error procesando {model_id}: {e}")
                
            # Short polite delay
            time_delay = random.uniform(2.0, 4.0)
            page.wait_for_timeout(int(time_delay * 1000))
            
        browser.close()
        
    conn.close()
    print(f"Finalizado BrickLink: {updated_count} imágenes actualizadas.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="all", choices=["omr", "bricklink", "all"])
    parser.add_argument("--limit-bl", type=int, default=10, help="Limit of BrickLink designs to resolve")
    args = parser.parse_args()
    
    if args.source in ["omr", "all"]:
        resolve_omr_images()
    if args.source in ["bricklink", "all"]:
        resolve_bricklink_images(args.limit_bl)

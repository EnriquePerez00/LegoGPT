import os
import sys
import sqlite3
import random
import time
import argparse
import json
from playwright.sync_api import sync_playwright
from scripts.download_bricklink_model import USER_AGENTS, human_delay

def harvest_metadata_pages(db_path: str = "data/catalog/models_catalog.db", max_pages: int = 10, items_per_page: int = 50):
    url = "https://www.bricklink.com/v3/studio/gallery.page"
    print(f"[Harvester] Inicializando navegador Playwright para WAF bypass...")
    
    ua = random.choice(USER_AGENTS)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")
        
        try:
            print("[Harvester] Cargando la galería de BrickLink...")
            page.goto(url, wait_until="networkidle", timeout=40000)
            human_delay(5.0, 10.0)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            print(f"[Harvester] Bucle de recolección iniciado (Límite: {max_pages} páginas)...")
            
            for page_num in range(1, max_pages + 1):
                print(f"\n---> Cosechando Página {page_num}/{max_pages}...")
                
                # Fetch query to internal AJAX search endpoint in browser context
                ajax_url = f"/ajax/clone/studio/gallery/search.ajax?q=*&downloadable=true&page={page_num}&itemsPerPage={items_per_page}&sort=recently_liked"
                
                js_fetch = f"""
                fetch('{ajax_url}')
                    .then(response => {{
                        if (!response.ok) throw new Error('HTTP status ' + response.status);
                        return response.json();
                    }})
                """
                
                try:
                    res_json = page.evaluate(js_fetch)
                except Exception as eval_err:
                    print(f"[Harvester Error] Falló evaluate de fetch: {eval_err}")
                    # If blocked or challenged, sleep longer and retry
                    human_delay(15.0, 30.0)
                    continue
                
                # Extract results list
                # Inspect structure: BrickLink usually stores it in 'list' or 'results' key
                creations = res_json.get("list", []) if isinstance(res_json, dict) else []
                
                if not creations:
                    # Alternative key check
                    if isinstance(res_json, dict) and "data" in res_json:
                        creations = res_json["data"].get("list", [])
                        
                if not creations or len(creations) == 0:
                    print("[Harvester] No se recibieron más modelos. Fin del catálogo alcanzado.")
                    break
                    
                print(f"[Harvester] Recibidos {len(creations)} modelos en JSON.")
                
                added = 0
                for item in creations:
                    model_id = str(item.get("idModel") or item.get("id") or "")
                    if not model_id:
                        continue
                        
                    name = item.get("name") or item.get("title") or f"Model {model_id}"
                    creator = item.get("creatorUsername") or item.get("creator", {}).get("username") or "Unknown"
                    parts_count = int(item.get("partsCount") or item.get("pieces") or 0)
                    likes = int(item.get("likeCount") or item.get("likes") or 0)
                    views = int(item.get("viewCount") or item.get("views") or 0)
                    downloads = int(item.get("downloadCount") or item.get("downloads") or 0)
                    
                    # Tags extraction
                    tag_list = item.get("tags") or []
                    tags_str = ",".join(tag_list) if isinstance(tag_list, list) else str(tag_list)
                    description = item.get("description") or f"Studio creation by {creator}"
                    
                    try:
                        # 1. Register in sets table
                        cursor.execute("""
                        INSERT OR REPLACE INTO sets (set_id, name, theme, description, source, file_path, parts_count, tags, likes_count, downloads_count, views_count, creator_username)
                        VALUES (?, ?, ?, ?, 'BrickLink', ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            model_id,
                            name,
                            "Studio MOC",
                            description,
                            f"data/bricklink_raw/{model_id}.io",
                            parts_count,
                            tags_str,
                            likes,
                            downloads,
                            views,
                            creator
                        ))
                        
                        # 2. Add to scraping queue
                        cursor.execute("""
                        INSERT OR IGNORE INTO bricklink_scraping_queue (model_id, status, attempts)
                        VALUES (?, 'pending', 0)
                        """, (model_id,))
                        
                        added += 1
                    except Exception as db_err:
                        print(f"Error base de datos para ID {model_id}: {db_err}")
                
                conn.commit()
                print(f"[Harvester] Guardados {added} registros en la base de datos de forma segura.")
                
                # Polite delay between page requests to avoid rate limits
                human_delay(1.5, 3.5)
                
            conn.close()
            print("\n[Harvester] Sesión de cosecha finalizada exitosamente.")
            
        except Exception as e:
            print(f"[ERROR] Error general del Harvester: {e}")
            
        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="data/catalog/models_catalog.db")
    parser.add_argument("--pages", type=int, default=10, help="Number of pages to harvest (50 items/page)")
    parser.add_argument("--limit", type=int, default=50, help="Items per page")
    args = parser.parse_args()
    
    harvest_metadata_pages(args.db_path, args.pages, args.limit)

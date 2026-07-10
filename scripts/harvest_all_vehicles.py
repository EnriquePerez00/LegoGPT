import os
import sys
import json
import sqlite3
import random
import time
from playwright.sync_api import sync_playwright
from scripts.download_bricklink_model import USER_AGENTS, human_delay

DB_PATH = "data/catalog/models_catalog.db"

def harvest_all_vehicles(db_path: str = DB_PATH, max_items: int = 10057):
    url = "https://www.bricklink.com/v3/studio/gallery.page?cat=1&show=downloadable"
    print(f"[Vehicle Harvester] Iniciando Playwright para cosechar la categoría de Vehículos...")
    
    ua = random.choice(USER_AGENTS)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        auth_state = "scratch/auth_state.json"
        storage = auth_state if os.path.exists(auth_state) else None
        
        context_args = {"storage_state": storage}
        if not storage:
            context_args["user_agent"] = ua
            
        context = browser.new_context(**context_args)
        page = context.new_page()
        page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")
        
        try:
            print("[Vehicle Harvester] Cargando la galería de Vehículos en BrickLink...")
            page.goto(url, wait_until="networkidle", timeout=50000)
            human_delay(5.0, 8.0)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Ensure tables exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS bricklink_scraping_queue (
                model_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                last_attempt_time TEXT,
                error_msg TEXT
            )
            """)
            conn.commit()
            
            total_collected = 0
            no_new_batch_count = 0
            
            print("[Vehicle Harvester] Comenzando bucle de extracción de metadatos e imágenes...")
            
            while total_collected < max_items:
                # Scroll to ensure elements are rendered and Load More is visible
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                human_delay(1.5, 2.5)
                
                # Extract all cards currently in the DOM
                cards = page.evaluate("""
                () => {
                    const data = [];
                    const items = document.querySelectorAll('.moc-card__internal');
                    items.forEach(item => {
                        const idModel = item.getAttribute('data-ts-id');
                        if (!idModel) return;
                        
                        // Extract only downloadable ones
                        const dlEl = item.querySelector('.moc-card__downloadable');
                        if (!dlEl) return;
                        
                        const nameEl = item.querySelector('.moc-card__name');
                        const creatorEl = item.querySelector('.moc-card__designer-name');
                        const likesEl = item.querySelector('.moc-card__likes');
                        const imgEl = item.querySelector('img');
                        
                        const name = nameEl ? nameEl.innerText.trim() : "";
                        const creator = creatorEl ? creatorEl.innerText.trim() : "";
                        const imgUrl = imgEl ? imgEl.src : "";
                        
                        let likes = 0;
                        if (likesEl) {
                            const likesText = likesEl.innerText.replace(/[^0-9]/g, '');
                            likes = parseInt(likesText) || 0;
                        }
                        
                        data.push({
                            idModel: idModel,
                            name: name,
                            creator: creator,
                            likes: likes,
                            imageUrl: imgUrl
                        });
                    });
                    return data;
                }
                """)
                
                # Save batch to DB
                new_in_batch = 0
                for c in cards:
                    model_id = c["idModel"]
                    name = c["name"]
                    creator = c["creator"]
                    likes = c["likes"]
                    img_url = c["imageUrl"]
                    
                    try:
                        # Insert set record with source_url and image_url
                        source_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
                        cursor.execute("""
                        INSERT OR REPLACE INTO sets (
                            set_id, name, theme, source, file_path, 
                            parts_count, likes_count, source_url, image_url, creator_username
                        )
                        VALUES (?, ?, 'Studio MOC', 'BrickLink', ?, 0, ?, ?, ?, ?)
                        """, (
                            model_id,
                            name,
                            f"data/bricklink_raw/{model_id}.io",
                            likes,
                            source_url,
                            img_url,
                            creator
                        ))
                        
                        # Add to queue as pending if not already processed
                        cursor.execute("""
                        INSERT OR IGNORE INTO bricklink_scraping_queue (model_id, status, attempts)
                        VALUES (?, 'pending', 0)
                        """, (model_id,))
                        
                        if cursor.rowcount > 0:
                            new_in_batch += 1
                    except Exception as e:
                        print(f"Error guardando ID {model_id}: {e}")
                        
                conn.commit()
                
                if new_in_batch > 0:
                    total_collected += new_in_batch
                    print(f"[Cosechador] Encontrados {len(cards)} en DOM | Nuevos guardados: {new_in_batch} | Total acumulado: {total_collected}/{max_items}")
                    no_new_batch_count = 0
                else:
                    no_new_batch_count += 1
                    print(f"[Cosechador] Sin nuevos modelos en esta carga (Intento {no_new_batch_count} - continuo...)")
                
                # OPTIMIZACIÓN DE MEMORIA: Borrar los nodos del DOM ya procesados para que el navegador no se congele
                try:
                    page.evaluate("""
                    () => {
                        const cards = document.querySelectorAll('.moc-card');
                        cards.forEach(card => card.remove());
                    }
                    """)
                except Exception as e_dom:
                    print(f"[Warning] Error al limpiar DOM: {e_dom}")
                
                # Intentar hacer click en 'Load more' o scroll, tolerando errores de red/comunicación
                try:
                    load_more_btn = page.locator('button[data-ts-name="studio-gallery-feed__load-more"]').first
                    if load_more_btn.count() > 0:
                        load_more_btn.click(timeout=10000)
                        human_delay(4.0, 8.0) # Esperar a que cargue la red y renderice
                    else:
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        human_delay(3.0, 6.0)
                except Exception as e_net:
                    print(f"[Cosechador Error] Error temporal de red/interacción: {e_net}. Reintentando en el próximo ciclo...")
                    human_delay(10.0, 20.0) # Pausa larga de recuperación ante error de red
                    
            conn.close()
            print(f"\n[Vehicle Harvester] ¡Cosecha masiva finalizada! Total de vehículos registrados: {total_collected}")
            
        except Exception as e:
            print(f"[Vehicle Harvester Error Crítico] {e}")
            
        try:
            browser.close()
        except Exception:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=DB_PATH)
    parser.add_argument("--max", type=int, default=10057, help="Maximum items to harvest")
    args = parser.parse_args()
    
    harvest_all_vehicles(args.db_path, args.max)

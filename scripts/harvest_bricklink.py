import os
import sys
import json
import sqlite3
import random
import time
from playwright.sync_api import sync_playwright
from scripts.download_bricklink_model import USER_AGENTS, human_delay

DB_PATH = "data/catalog/models_catalog.db"

CATEGORY_NAMES = {
    0: "All",
    1: "Vehicle",
    2: "Space",
    3: "Building",
    4: "Mecha",
    5: "Character",
    6: "Seasonal",
    7: "Life",
    8: "Misc",
    9: "Animal",
    10: "Popular"
}

def harvest_category(db_path: str, cat_id: int, max_items: int):
    cat_name = CATEGORY_NAMES.get(cat_id, "Desconocido")
    url = f"https://www.bricklink.com/v3/studio/gallery.page?cat={cat_id}&show=downloadable"
    print(f"\n==================================================")
    print(f"[Cosechador] Iniciando cosecha de categoría: {cat_name} (ID: {cat_id})")
    print(f"URL: {url}")
    print(f"==================================================")
    
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
            page.goto(url, wait_until="networkidle", timeout=60000)
            human_delay(4.0, 7.0)
            
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
            
            while total_collected < max_items:
                # Scroll to render elements
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
                        source_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
                        cursor.execute("""
                        INSERT OR REPLACE INTO sets (
                            set_id, name, theme, source, file_path, 
                            parts_count, likes_count, source_url, image_url, creator_username
                        )
                        VALUES (?, ?, ?, 'BrickLink', ?, 0, ?, ?, ?, ?)
                        """, (
                            model_id,
                            name,
                            cat_name,
                            f"data/bricklink_raw/{model_id}.io",
                            likes,
                            source_url,
                            img_url,
                            creator
                        ))
                        
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
                    print(f"[Cosechador] Encontrados {len(cards)} en DOM | Nuevos guardados: {new_in_batch} | Total acum. {cat_name}: {total_collected}/{max_items}")
                    no_new_batch_count = 0
                else:
                    no_new_batch_count += 1
                    print(f"[Cosechador] Sin nuevos modelos en esta carga (Intento {no_new_batch_count})")
                    if no_new_batch_count >= 10:
                        print("[Cosechador] Fin de catálogo detectado tras 10 cargas sin novedades.")
                        break
                
                # OPTIMIZACIÓN DE MEMORIA
                try:
                    page.evaluate("""
                    () => {
                        const cards = document.querySelectorAll('.moc-card');
                        cards.forEach(card => card.remove());
                    }
                    """)
                except Exception as e_dom:
                    print(f"[Warning] Error al limpiar DOM: {e_dom}")
                
                # Click Load More or Scroll
                try:
                    load_more_btn = page.locator('button[data-ts-name="studio-gallery-feed__load-more"]').first
                    if load_more_btn.count() > 0:
                        load_more_btn.click(timeout=10000)
                        human_delay(3.0, 6.0)
                    else:
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        human_delay(2.0, 4.0)
                except Exception as e_net:
                    print(f"[Cosechador Warning] Error de red/interacción: {e_net}. Recuperando...")
                    human_delay(8.0, 15.0)
                    
            conn.close()
            print(f"[Éxito] Finalizada cosecha para {cat_name}. Registrados {total_collected} modelos.")
            
        except Exception as e:
            print(f"[ERROR Crítico en {cat_name}] {e}")
        finally:
            browser.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=DB_PATH)
    parser.add_argument("--categories", type=str, default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated category IDs to harvest")
    parser.add_argument("--max_per_category", type=int, default=2000, help="Maximum items to harvest per category")
    args = parser.parse_args()
    
    cat_ids = [int(x.strip()) for x in args.categories.split(",") if x.strip().isdigit()]
    
    print(f"=== INICIANDO COSECHADOR DE BRICKLINK GALLERY ===")
    print(f"Base de Datos: {args.db_path}")
    print(f"Categorías seleccionadas: {cat_ids} ({[CATEGORY_NAMES.get(cid, '?') for cid in cat_ids]})")
    print(f"Límite máximo por categoría: {args.max_per_category}")
    
    for cid in cat_ids:
        harvest_category(args.db_path, cid, args.max_per_category)

if __name__ == "__main__":
    main()

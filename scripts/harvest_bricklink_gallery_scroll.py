import os
import sys
import sqlite3
import random
import time
import argparse
from playwright.sync_api import sync_playwright
from scripts.download_bricklink_model import USER_AGENTS, human_delay

def harvest_by_scroll(db_path: str = "data/catalog/models_catalog.db", max_scrolls: int = 100):
    url = "https://www.bricklink.com/v3/studio/gallery.page?q=*&show=downloadable"
    print(f"[Scroll Harvester] Iniciando Playwright para cosecha masiva en: {url}")
    
    ua = random.choice(USER_AGENTS)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        # Bypass webdriver detection
        page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")
        
        try:
            print("[Scroll Harvester] Cargando galería...")
            page.goto(url, wait_until="load", timeout=40000)
            human_delay(5.0, 8.0)
            
            # Click on "Popular" tab to unlock infinite scroll of 37k items
            print("[Scroll Harvester] Seleccionando pestaña 'Popular'...")
            popular_tab = page.locator('button:has-text("Popular"), div:has-text("Popular"), a:has-text("Popular")').first
            if popular_tab.count() > 0:
                popular_tab.click()
                human_delay(3.0, 5.0)
            
            # Select "Show: Downloadable" if not selected by URL
            # The URL parameter show=downloadable usually works, but let's make sure
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            unique_ids = set()
            no_new_count = 0
            
            print("[Scroll Harvester] Comenzando bucle de scrolling...")
            
            for scroll_idx in range(1, max_scrolls + 1):
                # Scroll to bottom to ensure "Load more" button is visible
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                human_delay(1.5, 2.5)
                
                # Click the "Load more creations" button
                load_more_btn = page.locator('button[data-ts-name="studio-gallery-feed__load-more"]')
                if load_more_btn.count() > 0:
                    load_more_btn.click()
                    human_delay(3.5, 5.0)  # Wait for new cards to render
                else:
                    # Fallback scroll
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    human_delay(2.0, 3.5)
                
                # Extract all cards currently in the DOM
                cards = page.evaluate("""
                () => {
                    const data = [];
                    const items = document.querySelectorAll('.moc-card__internal');
                    items.forEach(item => {
                        const idModel = item.getAttribute('data-ts-id');
                        if (!idModel) return;
                        
                        const dlEl = item.querySelector('.moc-card__downloadable');
                        if (!dlEl) return; // Skip non-downloadable creations
                        
                        const nameEl = item.querySelector('.moc-card__name');
                        const creatorEl = item.querySelector('.moc-card__designer-name');
                        const likesEl = item.querySelector('.moc-card__likes');
                        
                        const name = nameEl ? nameEl.innerText.trim() : "";
                        const creator = creatorEl ? creatorEl.innerText.trim() : "";
                        
                        let likes = 0;
                        if (likesEl) {
                            const likesText = likesEl.innerText.replace(/[^0-9]/g, '');
                            likes = parseInt(likesText) || 0;
                        }
                        
                        data.push({
                            idModel: idModel,
                            name: name,
                            creator: creator,
                            likes: likes
                        });
                    });
                    return data;
                }
                """)
                
                new_added = 0
                for c in cards:
                    model_id = c["idModel"]
                    if model_id not in unique_ids:
                        unique_ids.add(model_id)
                        new_added += 1
                        
                        # Store in database
                        try:
                            # 1. Register in sets
                            cursor.execute("""
                            INSERT OR REPLACE INTO sets (set_id, name, theme, description, source, file_path, parts_count, likes_count)
                            VALUES (?, ?, 'Studio MOC', 'No description yet', 'BrickLink', ?, 0, ?)
                            """, (
                                model_id,
                                c["name"],
                                f"data/bricklink_raw/{model_id}.io",
                                c["likes"]
                            ))
                            
                            # 2. Add to download queue
                            cursor.execute("""
                            INSERT OR IGNORE INTO bricklink_scraping_queue (model_id, status, attempts)
                            VALUES (?, 'pending', 0)
                            """, (model_id,))
                        except Exception as db_err:
                            print(f"Error guardando ID {model_id}: {db_err}")
                
                conn.commit()
                
                if new_added > 0:
                    print(f"Scroll {scroll_idx}/{max_scrolls} | Encontrados: {len(cards)} | Nuevos agregados: {new_added} | Total acumulado: {len(unique_ids)}")
                    no_new_count = 0
                else:
                    no_new_count += 1
                    print(f"Scroll {scroll_idx}/{max_scrolls} | Sin nuevos modelos (Intento {no_new_count}/10)")
                    
                if no_new_count >= 10:
                    print("[Scroll Harvester] Fin del scroll o fin de página detectado.")
                    break
                    
            conn.close()
            print(f"[Scroll Harvester] Proceso finalizado. Total de modelos cosechados en esta ejecución: {len(unique_ids)}")
            
        except Exception as e:
            print(f"[Scroll Harvester Error] {e}")
            
        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="data/catalog/models_catalog.db")
    parser.add_argument("--scrolls", type=int, default=100)
    args = parser.parse_args()
    
    harvest_by_scroll(args.db_path, args.scrolls)

import re
import json
import os
import sqlite3
import random
import time
from playwright.sync_api import sync_playwright
from scripts.download_bricklink_model import USER_AGENTS, human_delay

DB_PATH = "data/catalog/models_catalog.db"
OUTPUT_DIR = "data/bricklink_raw"

def is_already_processed(model_id: str) -> bool:
    # Check file
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{model_id}.io")):
        return True
    
    # Check DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT set_id FROM sets WHERE set_id = ?", (model_id,))
    in_sets = cursor.fetchone() is not None
    cursor.execute("SELECT model_id FROM bricklink_scraping_queue WHERE model_id = ? AND status = 'completed'", (model_id,))
    in_queue = cursor.fetchone() is not None
    conn.close()
    
    return in_sets or in_queue

def search_vehicles_category(limit: int = 20) -> list[str]:
    # URL specified by user with show=downloadable
    url = "https://www.bricklink.com/v3/studio/gallery.page?cat=1&show=downloadable"
    print(f"[Buscador Categoría Vehículos] Navegando a: {url}")
    
    ua = random.choice(USER_AGENTS)
    model_ids = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")
        
        try:
            page.goto(url, wait_until="networkidle", timeout=40000)
            human_delay(5.0, 8.0)
            
            # Ensure "Downloadable" is selected in the DOM if needed (although show=downloadable parameter is in URL)
            # Let's scroll down to load creations
            print("[Buscador Categoría Vehículos] Haciendo scroll para cargar más diseños...")
            for _ in range(3):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                human_delay(2.0, 4.0)
                
            # Extract cards matching downloadable criteria
            cards_data = page.evaluate("""
            () => {
                const ids = [];
                const items = document.querySelectorAll('.moc-card__internal');
                items.forEach(item => {
                    const idModel = item.getAttribute('data-ts-id');
                    if (!idModel) return;
                    
                    // Verify it has the downloadable indicator icon/text
                    const dlEl = item.querySelector('.moc-card__downloadable');
                    if (dlEl) {
                        ids.push(idModel);
                    }
                });
                return ids;
            }
            """)
            
            unique_ids = list(set(cards_data))
            print(f"[Buscador Categoría Vehículos] Encontrados {len(unique_ids)} IDs de modelos descargables en la página.")
            
            # Filter out already processed/downloaded models
            new_ids = [mid for mid in unique_ids if not is_already_processed(mid)]
            print(f"[Buscador Categoría Vehículos] {len(new_ids)} de ellos son totalmente nuevos (no descargados).")
            
            model_ids = new_ids[:limit]
            
        except Exception as e:
            print(f"[ERROR] Error navegando por la categoría de vehículos: {e}")
            
        browser.close()
        
    return model_ids

def main():
    os.makedirs("data", exist_ok=True)
    new_vehicle_ids = search_vehicles_category(limit=15)
    
    with open("data/bricklink_targets.json", "w") as f:
        json.dump(new_vehicle_ids, f, indent=2)
        
    print(f"\n[Buscador Categoría Vehículos] Guardados {len(new_vehicle_ids)} nuevos IDs de vehículos a descargar en data/bricklink_targets.json")
    print(f"IDs a descargar: {new_vehicle_ids}")

if __name__ == "__main__":
    main()

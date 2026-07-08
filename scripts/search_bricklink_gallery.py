import re
import json
import os
import random
import time
from playwright.sync_api import sync_playwright
from scripts.download_bricklink_model import USER_AGENTS, human_delay

def search_gallery_for_targets(query: str = "car", limit: int = 15) -> list[str]:
    url = f"https://www.bricklink.com/v3/studio/gallery.page?q={query}"
    print(f"[Buscador Gallery] Navegando a: {url}")
    
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
            page.goto(url, wait_until="networkidle", timeout=30000)
            human_delay(4.0, 8.0)
            
            # Scroll down to load more cards
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            human_delay(2.0, 4.0)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            human_delay(3.0, 6.0)
            
            # Find all links matching design.page?idModel=XXXX
            html = page.content()
            matches = re.findall(r'idModel=([0-9]+)', html)
            
            # Deduplicate
            unique_ids = list(set(matches))
            print(f"[Buscador Gallery] Encontrados {len(unique_ids)} IDs de modelos únicos para la búsqueda '{query}'.")
            
            # Limit results
            model_ids = unique_ids[:limit]
            
        except Exception as e:
            print(f"[ERROR] Error al buscar en la galería: {e}")
            
        browser.close()
        
    return model_ids

def main():
    # Search for both cars and bikes to expand our 4-wheeled and 2-wheeled dataset
    targets = []
    for q in ["car", "motorcycle"]:
        found = search_gallery_for_targets(q, limit=10)
        targets.extend(found)
        human_delay(5.0, 10.0)
        
    targets = list(set(targets))
    
    os.makedirs("data", exist_ok=True)
    with open("data/bricklink_targets.json", "w") as f:
        json.dump(targets, f, indent=2)
        
    print(f"\n[Catálogo Ingestión] Guardados {len(targets)} IDs de modelos objetivo en data/bricklink_targets.json")
    print(f"IDs a descargar: {targets}")

if __name__ == "__main__":
    main()

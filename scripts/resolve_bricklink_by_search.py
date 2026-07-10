import os
import sys
import sqlite3
import random
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

def resolve_by_search():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Select BrickLink sets
    cursor.execute("SELECT set_id, name, source_url FROM sets WHERE source = 'BrickLink'")
    rows = cursor.fetchall()
    
    targets = []
    for set_id, name, source_url in rows:
        # If set_id is not numeric and we don't have a numeric source_url resolved yet
        if not set_id.isdigit():
            is_resolved = False
            if source_url and "idModel=" in source_url:
                model_id_part = source_url.split("idModel=")[-1].split("&")[0]
                if model_id_part.isdigit():
                    is_resolved = True
            if not is_resolved:
                targets.append((set_id, name))
                
    if not targets:
        print("Todos los sets de BrickLink ya tienen IDs numéricos resueltos.")
        conn.close()
        return
        
    print(f"Iniciando búsqueda para resolver {len(targets)} modelos restantes...")
    
    ua = random.choice(USER_AGENTS)
    resolved_count = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        print("Cargando galería de BrickLink...")
        page.goto("https://www.bricklink.com/v3/studio/gallery.page", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(3000)
        
        for set_id, name in targets:
            query_name = name if name else set_id
            print(f"\nBuscando '{query_name}'...")
            
            try:
                # Clear search box first
                page.fill('#searchBox', '')
                page.wait_for_timeout(500)
                # Fill search box
                page.fill('#searchBox', query_name)
                page.wait_for_timeout(1000)
                # Click search cta button
                page.click('button[data-ts-name="studio-gallery__search-cta"]')
                page.wait_for_timeout(5000) # Wait for Ajax
                
                # Extract first matching model ID
                links = page.evaluate("""
                    () => {
                        const urls = [];
                        document.querySelectorAll('a').forEach(a => {
                            if (a.href && a.href.includes('idModel=')) {
                                urls.push(a.href);
                            }
                        });
                        return urls;
                    }
                """)
                
                if links:
                    first_link = links[0]
                    model_id = first_link.split("idModel=")[-1].split("&")[0]
                    if model_id.isdigit():
                        new_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
                        cursor.execute("""
                        UPDATE sets 
                        SET source_url = ?, image_url = NULL 
                        WHERE set_id = ?
                        """, (new_url, set_id))
                        conn.commit()
                        resolved_count += 1
                        print(f"  [+] Resuelto: '{set_id}' -> Model ID: {model_id}")
                        print(f"      Nueva URL: {new_url}")
                    else:
                        print(f"  [-] Link no numérico encontrado para '{set_id}': {first_link}")
                else:
                    print(f"  [-] No se encontraron resultados de búsqueda para '{set_id}'.")
                    
            except Exception as e:
                print(f"  [-] Error buscando '{set_id}': {e}")
                
            page.wait_for_timeout(random.randint(1500, 3000))
            
        browser.close()
        
    conn.close()
    print(f"\n=== Mapeo finalizado. Se resolvieron {resolved_count}/{len(targets)} modelos. ===")

if __name__ == "__main__":
    resolve_by_search()

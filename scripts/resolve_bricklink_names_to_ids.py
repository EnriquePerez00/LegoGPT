import os
import sys
import sqlite3
import random
import urllib.parse
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

def resolve_names():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Fetch all BrickLink sets
    cursor.execute("SELECT set_id, name, source_url FROM sets WHERE source = 'BrickLink'")
    rows = cursor.fetchall()
    
    # Filter for non-numeric set_ids
    targets = []
    for r in rows:
        set_id = r[0]
        if not set_id.isdigit():
            targets.append(r)
            
    if not targets:
        print("No se encontraron sets de BrickLink con IDs textuales para resolver.")
        conn.close()
        return
        
    print(f"Detectados {len(targets)} modelos con IDs textuales que necesitan resolución a IDs numéricos.")
    
    ua = random.choice(USER_AGENTS)
    resolved_count = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        # Go to gallery page to establish session/cookies
        print("Navegando a la galería de BrickLink para establecer la sesión...")
        page.goto("https://www.bricklink.com/v3/studio/gallery.page", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)
        
        for set_id, name, current_url in targets:
            print(f"\nResolviendo ID numérico para: '{set_id}'...")
            
            # Use name or set_id for query
            query = name if name else set_id
            encoded_query = urllib.parse.quote(query)
            ajax_url = f"/ajax/clone/studio/gallery/search.ajax?q={encoded_query}&downloadable=true"
            
            js_fetch = f"""
            fetch('{ajax_url}')
                .then(response => {{
                    if (!response.ok) throw new Error('HTTP status ' + response.status);
                    return response.json();
                }})
            """
            
            try:
                res_json = page.evaluate(js_fetch)
                results_list = res_json.get("result", {}).get("list", [])
                
                if results_list:
                    # Look for first exact or close match
                    matched_model = None
                    for model in results_list:
                        m_name = model.get("name", "").lower()
                        if query.lower() in m_name or m_name in query.lower():
                            matched_model = model
                            break
                    
                    if not matched_model:
                        # Fallback to first result
                        matched_model = results_list[0]
                        
                    model_id = matched_model.get("id") or matched_model.get("idModel")
                    matched_name = matched_model.get("name")
                    
                    if model_id:
                        new_source_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
                        cursor.execute("""
                        UPDATE sets 
                        SET source_url = ?, image_url = NULL 
                        WHERE set_id = ?
                        """, (new_source_url, set_id))
                        conn.commit()
                        resolved_count += 1
                        print(f"  [+] Resuelto: '{set_id}' -> Model ID: {model_id} ({matched_name})")
                        print(f"      Nueva URL: {new_source_url}")
                    else:
                        print(f"  [-] No se encontró model ID en el objeto resultado para '{set_id}'.")
                else:
                    print(f"  [-] La búsqueda AJAX no devolvió resultados para '{set_id}'.")
                    
            except Exception as e:
                print(f"  [-] Error buscando '{set_id}': {e}")
                
            # Random delay
            page.wait_for_timeout(random.randint(1500, 3000))
            
        browser.close()
    
    conn.close()
    print(f"\n=== Proceso completado. Se resolvieron {resolved_count}/{len(targets)} IDs textuales de BrickLink. ===")

if __name__ == "__main__":
    resolve_names()

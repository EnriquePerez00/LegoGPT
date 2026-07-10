import os
import sys
import sqlite3
import random
import urllib.parse
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

REMAINING_NAMES = [
    'Obsidian monarch', 'Hommage', "Kai's hot rod", 'Transforming Blue Car - Accessories',
    'Subaru Baja grey different sides', 'HCR2 Rally Car', 'Red convertible_Copy', 'hot rod 2',
    'A jazzy scene', '2020 Dodge Challenger Hellcat Redeye', 'Beach fiat 500', 'micro cart 3',
    'Circuit racer 3'
]

def resolve_remaining():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    ua = random.choice(USER_AGENTS)
    resolved_count = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        # Open Google home once to establish context
        print("Abriendo Google...")
        page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
        
        # Check if there is a cookie consent dialog and click it if possible
        try:
            reject_btn = page.locator('button:has-text("Rechazar todo"), button:has-text("Reject all"), button:has-text("Antes de ir a Google")')
            if reject_btn.count() > 0:
                reject_btn.first.click()
                print("Consensuado/Rechazado cookies de Google.")
                page.wait_for_timeout(1000)
        except Exception:
            pass
            
        for name in REMAINING_NAMES:
            print(f"\nBuscando en Google: '{name}'...")
            
            # Format search query
            query = f'"{name}" bricklink studio design'
            q_encoded = urllib.parse.quote(query)
            google_url = f"https://www.google.com/search?q={q_encoded}"
            
            try:
                page.goto(google_url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(3000)
                
                # Extract all links containing bricklink.com and design.page
                links = page.evaluate("""
                    () => {
                        const urls = [];
                        document.querySelectorAll('a').forEach(a => {
                            const href = a.href || '';
                            if (href.includes('bricklink.com') && (href.includes('design.page') || href.includes('idModel'))) {
                                urls.push(href);
                            }
                        });
                        return urls;
                    }
                """)
                
                matched_model_id = None
                for l in links:
                    decoded = urllib.parse.unquote(l)
                    # Look for idModel=XXXXXX inside decoded URL
                    if "idModel=" in decoded:
                        model_id = decoded.split("idModel=")[-1].split("&")[0]
                        if model_id.isdigit():
                            matched_model_id = model_id
                            break
                    elif "idModel%3D" in decoded:
                        model_id = decoded.split("idModel%3D")[-1].split("&")[0]
                        if model_id.isdigit():
                            matched_model_id = model_id
                            break
                            
                if matched_model_id:
                    new_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={matched_model_id}"
                    cursor.execute("""
                    UPDATE sets 
                    SET source_url = ?, image_url = NULL 
                    WHERE set_id = ?
                    """, (new_url, name))
                    conn.commit()
                    resolved_count += 1
                    print(f"  [+] ¡ENCONTRADO! '{name}' -> Model ID: {matched_model_id}")
                    print(f"      URL: {new_url}")
                else:
                    print(f"  [-] No se pudo extraer model ID de las URLs encontradas: {links[:3]}")
                    
            except Exception as e:
                print(f"  [-] Error buscando '{name}': {e}")
                
            page.wait_for_timeout(random.randint(2000, 4000))
            
        browser.close()
        
    conn.close()
    print(f"\n=== Finalizado. Se resolvieron {resolved_count}/{len(REMAINING_NAMES)} modelos restantes. ===")

if __name__ == "__main__":
    resolve_remaining()

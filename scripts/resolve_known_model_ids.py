import os
import sys
import sqlite3
import random
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

KNOWN_IDS = [
    723203, 804483, 808425, 808584, 808927, 809074, 810166, 810438, 811112,
    811354, 811374, 812346, 812710, 813002, 813418, 813802, 814165, 814787,
    814812, 815247, 815734, 816441, 816479, 816599, 817451, 817460, 817623,
    817683, 818687, 819570, 819819, 819889, 819890, 820180, 820460, 820501,
    822060, 822304, 823122, 823263, 823397, 823402, 823435, 823646, 823866,
    824063, 824247, 824394, 824473, 824822, 825691, 825862, 826108, 826320,
    826420, 826484, 826935, 827055, 827158, 827351, 828418, 828534, 828541,
    828624, 829013, 829756, 829769, 830149, 830225, 830564, 830943, 831213,
    831226, 831540, 831804, 831840, 832121, 832182, 832221, 832510, 832624,
    832877, 832959, 833941, 834069, 834284, 836591, 838077
]

def resolve_known():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch non-numeric sets
    cursor.execute("SELECT set_id, name FROM sets WHERE source = 'BrickLink'")
    rows = cursor.fetchall()
    
    targets = {}
    for r in rows:
        set_id = r[0]
        if not set_id.isdigit():
            # Standardize name for matching
            clean_key = set_id.lower().replace(" ", "").replace("-", "").replace("_", "").strip()
            targets[clean_key] = set_id
            
    if not targets:
        print("No hay sets textuales pendientes de resolver.")
        conn.close()
        return
        
    print(f"Buscando correspondencia para {len(targets)} sets textuales usando {len(KNOWN_IDS)} IDs conocidos...")
    
    ua = random.choice(USER_AGENTS)
    resolved = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        for model_id in KNOWN_IDS:
            url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
            print(f"Verificando ID {model_id}...")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                title = page.title()
                
                # Title format: "Compact Patrol from BrickLink Studio [BrickLink]" or similar
                clean_title = title.split("from BrickLink Studio")[0].strip()
                if not clean_title or "Page Not Found" in title or "Design Gallery" in title:
                    continue
                    
                match_key = clean_title.lower().replace(" ", "").replace("-", "").replace("_", "").strip()
                
                # Check for direct match or substring match
                matched_set_id = None
                for t_key, original_id in targets.items():
                    if t_key == match_key or t_key in match_key or match_key in t_key:
                        matched_set_id = original_id
                        break
                        
                if matched_set_id:
                    cursor.execute("""
                    UPDATE sets 
                    SET source_url = ?, image_url = NULL 
                    WHERE set_id = ?
                    """, (url, matched_set_id))
                    conn.commit()
                    resolved += 1
                    print(f"  [+] ¡ASOCIACIÓN ENCONTRADA! '{matched_set_id}' -> Model ID {model_id} ('{clean_title}')")
                    # Remove from targets to avoid duplicate matches
                    del targets[matched_set_id.lower().replace(" ", "").replace("-", "").replace("_", "").strip()]
                    
            except Exception as e:
                print(f"  [-] Error verificando {model_id}: {e}")
                
            if not targets:
                print("Todos los sets textuales han sido resueltos.")
                break
                
            page.wait_for_timeout(random.randint(1000, 2000))
            
        browser.close()
        
    conn.close()
    print(f"\n=== Mapeo finalizado. {resolved} sets textuales fueron asociados a sus IDs numéricos correspondientes. ===")

if __name__ == "__main__":
    resolve_known()

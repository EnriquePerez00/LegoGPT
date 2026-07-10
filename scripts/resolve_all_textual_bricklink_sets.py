import os
import sys
import sqlite3
import random
import time
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

def clean_string(s):
    if not s:
        return ""
    return "".join(c.lower() for c in s if c.isalnum())

def resolve_all_strict(limit=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Select BrickLink sets that have textual set_ids and are not resolved yet
    cursor.execute("""
        SELECT set_id, name FROM sets 
        WHERE source = 'BrickLink' AND (source_url IS NULL OR source_url NOT LIKE '%idModel=%')
    """)
    rows = cursor.fetchall()
    
    targets = []
    for set_id, name in rows:
        if not set_id.isdigit():
            targets.append((set_id, name))
            
    if not targets:
        print("No textual sets left to resolve.")
        conn.close()
        return
        
    print(f"Found {len(targets)} textual sets. Resolving strictly...")
    if limit:
        targets = targets[:limit]
        print(f"Limiting to first {limit} sets for this run.")
        
    ua = random.choice(USER_AGENTS)
    resolved_count = 0
    skipped_count = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        # Load gallery page once to set cookies/session
        print("Loading BrickLink Gallery...")
        page.goto("https://www.bricklink.com/v3/studio/gallery.page", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)
        
        for idx, (set_id, name) in enumerate(targets):
            query = name if name else set_id
            print(f"\n[{idx+1}/{len(targets)}] Processing '{set_id}' (Query: '{query}')...")
            
            try:
                # Dismiss any blocking overlays if present
                try:
                    overlay_btn = page.locator('.custom-confirm__btn, .custom-confirm__btn--primary, button:has-text("Accept"), button:has-text("OK")').first
                    if overlay_btn.is_visible():
                        overlay_btn.click()
                        print("  [~] Dismissed blocking overlay.")
                except Exception:
                    pass
                    
                # 1. Clear and fill search box
                page.fill('input#searchBox', '')
                page.wait_for_timeout(300)
                page.fill('input#searchBox', query)
                page.wait_for_timeout(500)
                page.press('input#searchBox', 'Enter')
                page.wait_for_timeout(4000) # Wait for AJAX render
                
                # 2. Extract results
                cards = page.evaluate("""
                () => {
                    const data = [];
                    document.querySelectorAll('.moc-card__internal').forEach(item => {
                        const idModel = item.getAttribute('data-ts-id');
                        const nameEl = item.querySelector('.moc-card__name');
                        if (idModel && nameEl) {
                            data.push({
                                id: idModel,
                                name: nameEl.innerText.trim()
                            });
                        }
                    });
                    return data;
                }
                """)
                
                # 3. Perform strict string matching
                matched_id = None
                target_clean = clean_string(query)
                
                # Check for exact matches first
                for card in cards:
                    card_clean = clean_string(card["name"])
                    if target_clean == card_clean:
                        matched_id = card["id"]
                        print(f"  [+] Exact match found: '{card['name']}' -> ID {matched_id}")
                        break
                        
                # Fallback to substring match only if we are very confident
                if not matched_id and cards:
                    for card in cards:
                        card_clean = clean_string(card["name"])
                        if target_clean in card_clean or card_clean in target_clean:
                            matched_id = card["id"]
                            print(f"  [~] Substring match found: '{card['name']}' -> ID {matched_id}")
                            break
                            
                if matched_id:
                    # 4. Navigate directly to design page to get canonical image
                    design_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={matched_id}"
                    
                    # Open in new page to avoid losing search state, or just navigate back
                    det_page = context.new_page()
                    det_page.goto(design_url, wait_until="domcontentloaded", timeout=20000)
                    det_page.wait_for_timeout(2000)
                    
                    og_image = det_page.locator('meta[property="og:image"]').get_attribute("content")
                    det_page.close()
                    
                    if og_image:
                        print(f"  [+] Canonical Image found: {og_image}")
                        cursor.execute("""
                        UPDATE sets 
                        SET source_url = ?, image_url = ? 
                        WHERE set_id = ?
                        """, (design_url, og_image, set_id))
                        conn.commit()
                        resolved_count += 1
                    else:
                        print(f"  [-] Failed to find og:image for model ID {matched_id}")
                        cursor.execute("""
                        UPDATE sets 
                        SET source_url = ? 
                        WHERE set_id = ?
                        """, (design_url, set_id))
                        conn.commit()
                else:
                    print(f"  [-] No strict match found in {len(cards)} search results.")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"  [-] Error processing '{set_id}': {e}")
                
            # Politely delay between searches (3-5 seconds)
            page.wait_for_timeout(random.randint(3000, 5000))
            
            # Periodically rest for 25 seconds every 10 searches to let IP cool down
            if (idx + 1) % 10 == 0:
                print("  [~] Batch finished. Cooling down for 25 seconds...")
                page.wait_for_timeout(25000)
            
        browser.close()
        
    conn.close()
    print(f"\n=== Resolution Complete: {resolved_count} sets resolved, {skipped_count} skipped. ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max sets to resolve in this run")
    args = parser.parse_args()
    
    resolve_all_strict(args.limit)


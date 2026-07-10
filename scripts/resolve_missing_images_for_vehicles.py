import os
import sys
import sqlite3
import random
import time
import urllib.parse
from playwright.sync_api import sync_playwright

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

def main():
    print("=== Resolviendo Imágenes para Vehículos Clasificados ===")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all classified vehicles (level_1_entorno is not null and not 'Otros')
    rows = c.execute("""
        SELECT set_id, name, source_url, image_url 
        FROM sets 
        WHERE level_1_entorno IS NOT NULL AND level_1_entorno != 'Otros' AND source = 'BrickLink'
    """).fetchall()
    conn.close()
    
    if not rows:
        print("No classified vehicles found in the database.")
        return
        
    print(f"Found {len(rows)} classified BrickLink vehicles to check.")
    
    ua = random.choice(USER_AGENTS)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        # Load gallery to establish session
        try:
            page.goto("https://www.bricklink.com/v3/studio/gallery.page", wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(2000)
        except Exception:
            pass
            
        for idx, (set_id, name, src_url, img_url) in enumerate(rows):
            # If src_url is fallback or None, let's try to resolve it first
            model_id = set_id
            if not src_url or "gallery.page?q=" in src_url or not src_url.startswith("http"):
                print(f"\n[{idx+1}/{len(rows)}] Resolving URL for '{set_id}'...")
                try:
                    page.fill('#searchBox', '')
                    page.wait_for_timeout(500)
                    page.fill('#searchBox', set_id if not set_id.isdigit() else name)
                    page.wait_for_timeout(1000)
                    page.click('button[data-ts-name="studio-gallery__search-cta"]')
                    page.wait_for_timeout(4000)
                    
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
                            src_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
                            conn = sqlite3.connect(DB_PATH)
                            cur = conn.cursor()
                            cur.execute("UPDATE sets SET source_url = ? WHERE set_id = ?", (src_url, set_id))
                            conn.commit()
                            conn.close()
                            print(f"  [+] URL resolved: {src_url}")
                except Exception as e:
                    print(f"  [-] Failed to resolve URL for '{set_id}': {e}")
                    
            if not src_url or "idModel=" not in src_url:
                if set_id.isdigit():
                    model_id = set_id
                    src_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
                else:
                    print(f"[{idx+1}/{len(rows)}] Skipping '{set_id}', no valid design page URL.")
                    continue
            else:
                model_id = src_url.split("idModel=")[-1].split("&")[0]
                
            print(f"\n[{idx+1}/{len(rows)}] Scraping images for '{set_id}' (Model: {model_id})...")
            try:
                page.goto(src_url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(3000)
                
                # Extract specific model images and main image
                found_urls = page.evaluate("""
                    () => {
                        const urls = [];
                        const og = document.querySelector('meta[property="og:image"]');
                        if (og && og.content) {
                            urls.push(og.content);
                        }
                        const imgs = document.querySelectorAll('img');
                        imgs.forEach(img => {
                            const src = img.src;
                            if (src && src.includes('file.bricklink.info')) {
                                let parent = img.parentElement;
                                let isRec = false;
                                while (parent) {
                                    if (parent.classList.contains('moc-card') || 
                                        parent.classList.contains('studio-gallery-feed') || 
                                        parent.classList.contains('studio-model__recommend') ||
                                        parent.classList.contains('gallery-card__thumbnails-item') ||
                                        parent.classList.contains('gallery-card__thumbnails') ||
                                        parent.id === 'related-creations') {
                                        isRec = true;
                                        break;
                                    }
                                    parent = parent.parentElement;
                                }
                                if (!isRec) {
                                    urls.push(src);
                                }
                            }
                        });
                        return Array.from(new Set(urls));
                    }
                """)
                
                found_urls = [url if url.startswith('http') else 'https:' + url for url in found_urls]
                
                if found_urls:
                    main_img = found_urls[0]
                    print(f"  [+] Main image: {main_img}")
                    print(f"  [+] Carousel images found: {len(found_urls)}")
                    
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    
                    # Update main image
                    cur.execute("UPDATE sets SET image_url = ? WHERE set_id = ?", (main_img, set_id))
                    
                    # Save all to set_images
                    for url in found_urls:
                        cur.execute("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, 'BrickLink')", (set_id, url))
                        
                    conn.commit()
                    conn.close()
                else:
                    print("  [-] No specific model images found.")
            except Exception as e:
                print(f"  [-] Error scraping images: {e}")
                
            page.wait_for_timeout(random.randint(1500, 3000))
            
        browser.close()
        
    print("\n=== Image Resolution Finished ===")

if __name__ == "__main__":
    main()

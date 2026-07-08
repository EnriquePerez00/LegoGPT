import os
import sys
import sqlite3
import re
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.omr_downloader import build_sets_index, search_set_metadata
from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

def harvest_omr_all_images():
    print("--- Cosechando TODAS las imágenes de OMR ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get OMR sets from sets table
    cursor.execute("SELECT set_id, source_url, image_url FROM sets WHERE source = 'OMR'")
    rows = cursor.fetchall()
    print(f"Procesando {len(rows)} modelos OMR en la base de datos...")
    
    # Pre-populate main images in set_images table if they exist
    main_images = []
    for r in rows:
        set_id, src_url, img_url = r
        if img_url:
            main_images.append((set_id, img_url, 'OMR'))
            
    if main_images:
        cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", main_images)
        conn.commit()
        print(f"  [+] Pobladas {len(main_images)} imágenes principales en set_images.")
        
    # Scrape the source pages for extra images (like alternate renders)
    def scrape_omr_page(set_id, source_url):
        if not source_url:
            return set_id, []
        try:
            res = requests.get(source_url, timeout=10)
            if res.status_code == 200:
                # Find all images in ldraw.org media/omr_models
                urls = re.findall(r'src="(https://library\.ldraw\.org/media/omr_models/[^"]+)"', res.text)
                hrefs = re.findall(r'href="(https://library\.ldraw\.org/media/omr_models/[^"]+)"', res.text)
                return set_id, list(set(urls + hrefs))
        except Exception as e:
            print(f"Error scraping OMR {set_id}: {e}")
        return set_id, []

    total_extra = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(scrape_omr_page, r[0], r[1]): r for r in rows if r[1]}
        
        batch = []
        for future in as_completed(futures):
            set_id, img_urls = future.result()
            for url in img_urls:
                batch.append((set_id, url, 'OMR'))
                total_extra += 1
                
            if len(batch) >= 100:
                cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", batch)
                conn.commit()
                batch = []
                
        if batch:
            cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", batch)
            conn.commit()
            
    conn.close()
    print(f"Finalizado OMR: {total_extra} referencias de imágenes guardadas en total.")

def harvest_rebrickable_all_images(limit=50):
    print("--- Cosechando TODAS las imágenes de Rebrickable ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get Rebrickable sets
    cursor.execute("SELECT set_num, img_url FROM rb_sets LIMIT ?", (limit,))
    rows = cursor.fetchall()
    print(f"Procesando {len(rows)} sets oficiales de Rebrickable...")
    
    # Pre-populate main images
    main_images = []
    for r in rows:
        set_num, img_url = r
        if img_url:
            main_images.append((set_num, img_url, 'Official'))
            
    if main_images:
        cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", main_images)
        conn.commit()
        print(f"  [+] Pobladas {len(main_images)} imágenes principales oficiales.")

    # Scrape rebrickable set page for additional images
    def scrape_rebrickable_page(set_id):
        # Format set ID for Rebrickable URL: e.g. "75105-1" -> "75105-1"
        url = f"https://rebrickable.com/sets/{set_id}/"
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                # Look for images in the HTML matching rebrickable's media/sets/ or similar paths
                urls = re.findall(r'src="(https://cdn\.rebrickable\.com/media/sets/[^"]+)"', res.text)
                hrefs = re.findall(r'href="(https://cdn\.rebrickable\.com/media/sets/[^"]+)"', res.text)
                # Alternate MOC images or parts images could also be extracted
                mocs_urls = re.findall(r'src="(https://cdn\.rebrickable\.com/media/mocs/[^"]+)"', res.text)
                return set_id, list(set(urls + hrefs + mocs_urls))
        except Exception as e:
            print(f"Error scraping Rebrickable {set_id}: {e}")
        return set_id, []

    total_extra = 0
    # Rebrickable can rate-limit easily, so we use fewer threads and a small delay
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scrape_rebrickable_page, r[0]): r for r in rows}
        
        batch = []
        for future in as_completed(futures):
            set_id, img_urls = future.result()
            for url in img_urls:
                batch.append((set_id, url, 'Official'))
                total_extra += 1
            time.sleep(random.uniform(0.5, 1.5))
            
            if len(batch) >= 50:
                cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", batch)
                conn.commit()
                batch = []
                
        if batch:
            cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", batch)
            conn.commit()
            
    conn.close()
    print(f"Finalizado Rebrickable: {total_extra} referencias de imágenes guardadas en total.")

def harvest_bricklink_all_images(limit=10):
    print("--- Cosechando TODAS las imágenes de BrickLink (usando Playwright) ---")
    from playwright.sync_api import sync_playwright
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT set_id, image_url FROM sets WHERE source = 'BrickLink' AND set_id != '563116' LIMIT ?", (limit,))
    rows = cursor.fetchall()
    
    # Pre-populate main images
    main_images = []
    for r in rows:
        set_id, img_url = r
        if img_url:
            main_images.append((set_id, img_url, 'BrickLink'))
            
    if main_images:
        cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", main_images)
        conn.commit()
        print(f"  [+] Pobladas {len(main_images)} imágenes principales de BrickLink.")

    # Scrape slideshow/gallery images using Playwright
    ua = random.choice(USER_AGENTS)
    total_extra = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        for r in rows:
            model_id = r[0]
            url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
            print(f"Scraping BrickLink gallery for ID {model_id}...")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(3000)
                
                # Extract only specific model images using DOM filtering
                found_urls = page.evaluate("""
                    () => {
                        const urls = [];
                        
                        // 1. Get OpenGraph preview image
                        const og = document.querySelector('meta[property="og:image"]');
                        if (og && og.content) {
                            urls.push(og.content);
                        }
                        
                        // 2. Get specific model images and thumbs (exclude recommendations)
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
                
                # Convert relative URLs to absolute if any
                found_urls = [url if url.startswith('http') else 'https:' + url for url in found_urls]
                
                if found_urls:
                    print(f"  [+] Encontradas {len(found_urls)} imágenes específicas para {model_id}.")
                    batch = [(model_id, url, 'BrickLink') for url in found_urls]
                    cursor.executemany("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, ?)", batch)
                    conn.commit()
                    total_extra += len(found_urls)
                    
            except Exception as e:
                print(f"Error scraping BrickLink MOC {model_id}: {e}")
                
            page.wait_for_timeout(random.randint(2000, 4000))
            
        browser.close()
        
    conn.close()
    print(f"Finalizado BrickLink: {total_extra} imágenes guardadas.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="all", choices=["omr", "rebrickable", "bricklink", "all"])
    parser.add_argument("--limit-rb", type=int, default=30, help="Limit of Rebrickable sets to scrape")
    parser.add_argument("--limit-bl", type=int, default=5, help="Limit of BrickLink MOCs to scrape")
    args = parser.parse_args()
    
    if args.source in ["omr", "all"]:
        harvest_omr_all_images()
    if args.source in ["rebrickable", "all"]:
        harvest_rebrickable_all_images(args.limit_rb)
    if args.source in ["bricklink", "all"]:
        harvest_bricklink_all_images(args.limit_bl)

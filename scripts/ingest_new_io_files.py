import os
import sys
import glob
import time
import json
import sqlite3
import urllib.parse
import random
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion_pipeline import process_and_register_downloaded_model
from scripts.download_bricklink_model import USER_AGENTS
from playwright.sync_api import sync_playwright

# Import VLM classifier components
from src.classification_pipeline import VehicleClassifierAgent, BricklinkGalleryExtractor, save_classification_to_db

DB_PATH = "data/catalog/models_catalog.db"
PROGRESS_PATH = "data/ingestion_progress.json"

db_lock = threading.Lock()
progress_lock = threading.Lock()
playwright_lock = threading.Lock()

def search_url_for_set(page, query_id):
    try:
        page.fill('#searchBox', '')
        page.wait_for_timeout(300)
        page.fill('#searchBox', query_id)
        page.wait_for_timeout(800)
        page.click('button[data-ts-name="studio-gallery__search-cta"]')
        try:
            page.wait_for_selector('a[href*="idModel="]', timeout=4000)
        except Exception:
            page.wait_for_timeout(1000)
        
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
                return f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
    except Exception as e:
        print(f"  [-] Search failed for '{query_id}': {e}")
    return None

def main():
    print("=== Parallel Ingestion & VLM Classification Process Started ===")
    
    # 1. Identify missing files
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        db_sets = set(row[0] for row in c.execute("SELECT set_id FROM sets").fetchall())
        
        # Fetch sets that have fallback URLs to resolve
        fallback_rows = c.execute("SELECT set_id FROM sets WHERE source = 'BrickLink' AND (source_url LIKE '%gallery.page?q=%' OR source_url IS NULL)").fetchall()
        fallbacks_to_fix = [r[0] for r in fallback_rows]
        conn.close()
    
    io_files = glob.glob("data/bricklink_raw/*.io")
    to_process = []
    for f in io_files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base not in db_sets:
            to_process.append((base, f, os.path.getsize(f)))
            
    # Sort by file size in ascending order (smallest first)
    to_process.sort(key=lambda x: x[2])
    to_process = [(x[0], x[1]) for x in to_process]
            
    total_to_process = len(to_process)
    print(f"Found {len(io_files)} total .io files. {total_to_process} need ingestion. {len(fallbacks_to_fix)} existing sets need URL fix.")
    
    # Load or initialize progress
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                progress = json.load(f)
        except Exception:
            progress = {
                "start_time": datetime.now().isoformat(),
                "total_to_process": total_to_process,
                "processed_count": 0,
                "completed": False,
                "items": []
            }
    else:
        progress = {
            "start_time": datetime.now().isoformat(),
            "total_to_process": total_to_process,
            "processed_count": 0,
            "completed": False,
            "items": []
        }
        
    progress["completed"] = False
    
    # Start Playwright Browser
    with sync_playwright() as p:
        print("Launching browser for URL resolution...")
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        
        # 2. Fix existing fallbacks in parallel (workers = 3)
        if fallbacks_to_fix:
            print(f"\nFixing {len(fallbacks_to_fix)} fallback URLs in parallel...")
            
            def fix_single_fallback(set_id):
                ua = random.choice(USER_AGENTS)
                with playwright_lock:
                    context = browser.new_context(user_agent=ua)
                    page = context.new_page()
                try:
                    page.goto("https://www.bricklink.com/v3/studio/gallery.page", wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(2000)
                except Exception:
                    pass
                
                print(f"Fixing URL for '{set_id}'...")
                resolved_url = search_url_for_set(page, set_id)
                with playwright_lock:
                    context.close()
                
                if resolved_url:
                    with db_lock:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("UPDATE sets SET source_url = ? WHERE set_id = ?", (resolved_url, set_id))
                        conn.commit()
                        conn.close()
                    print(f"  [+] URL fixed to: {resolved_url}")
                    
                    with progress_lock:
                        # Also update in the progress JSON if present
                        for item in progress.get("items", []):
                            if item["set_id"] == set_id:
                                item["url"] = resolved_url
                        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
                            json.dump(progress, f, indent=2)
                else:
                    print(f"  [-] Could not resolve correct URL for '{set_id}'")
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(fix_single_fallback, sid) for sid in fallbacks_to_fix]
                for future in as_completed(futures):
                    pass
                    
        # 3. Process remaining missing files in parallel (workers = 3)
        if to_process:
            print(f"\nIngesting and classifying {total_to_process} sets in parallel...")
            
            # Pre-initialize agents per thread/context
            vlm_agent = VehicleClassifierAgent()
            bl_extractor = BricklinkGalleryExtractor()
            
            def ingest_and_classify_single(set_id, file_path):
                print(f"Starting Ingestion for '{set_id}'...")
                start_time = time.time()
                
                # Ingest (safe to do concurrently, write done inside process_and_register_downloaded_model is protected by database engine)
                # But wait, to be safe against concurrent writes inside process_and_register_downloaded_model, we execute it under db_lock
                with db_lock:
                    success = process_and_register_downloaded_model(file_path, "BrickLink", db_path=DB_PATH)
                
                elapsed = time.time() - start_time
                subassemblies_count = 0
                parts_count = 0
                source_url = None
                
                if success:
                    # Read metadata from DB
                    with db_lock:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        row = c.execute("SELECT parts_count, subassemblies_count FROM sets WHERE set_id = ?", (set_id,)).fetchone()
                        if row:
                            parts_count, subassemblies_count = row
                        conn.close()
                    
                    # Resolve URL using UI search (requires isolated page)
                    ua = random.choice(USER_AGENTS)
                    with playwright_lock:
                        context = browser.new_context(user_agent=ua)
                        page = context.new_page()
                    try:
                        page.goto("https://www.bricklink.com/v3/studio/gallery.page", wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(2000)
                        source_url = search_url_for_set(page, set_id)
                    except Exception as e:
                        print(f"  [-] Search context error: {e}")
                    finally:
                        with playwright_lock:
                            context.close()
                        
                    if source_url:
                        with db_lock:
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute("UPDATE sets SET source_url = ? WHERE set_id = ?", (source_url, set_id))
                            conn.commit()
                            conn.close()
                        print(f"  [+] URL resolved: {source_url}")
                    else:
                        source_url = f"https://www.bricklink.com/v3/studio/gallery.page?q={urllib.parse.quote(set_id)}"
                    
                    # Automatically run VLM classification (Ollama call is thread-safe)
                    print(f"  [VLM] Running classification for '{set_id}'...")
                    try:
                        with db_lock:
                            metadata = bl_extractor.extract(set_id)
                        if metadata:
                            classification_result = vlm_agent.classify_design(metadata)
                            with db_lock:
                                save_classification_to_db(classification_result)
                            print(f"  [VLM] Success: {classification_result.taxonomy_proposal.Level_1_Entorno} / {classification_result.taxonomy_proposal.Level_3_Clase}")
                        else:
                            print("  [VLM Error] Could not extract metadata.")
                    except Exception as ve:
                        print(f"  [VLM Error] Classification failed: {ve}")
                else:
                    source_url = "N/A"
                    
                # Append item to progress
                with progress_lock:
                    item_exists = False
                    for item in progress.get("items", []):
                        if item["set_id"] == set_id:
                            item["status"] = "success" if success else "failed"
                            item["parts_count"] = parts_count
                            item["subassemblies_count"] = subassemblies_count
                            item["elapsed_seconds"] = round(elapsed, 2)
                            item["url"] = source_url
                            item_exists = True
                            break
                            
                    if not item_exists:
                        item = {
                            "set_id": set_id,
                            "status": "success" if success else "failed",
                            "parts_count": parts_count,
                            "subassemblies_count": subassemblies_count,
                            "elapsed_seconds": round(elapsed, 2),
                            "url": source_url
                        }
                        progress["items"].append(item)
                        progress["processed_count"] += 1
                    
                    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
                        json.dump(progress, f, indent=2)
                        
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(ingest_and_classify_single, sid, fpath) for sid, fpath in to_process]
                for future in as_completed(futures):
                    pass
                    
        browser.close()
        
    with progress_lock:
        progress["completed"] = True
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
            
    print("\n=== Parallel Ingestion Process Completed ===")

if __name__ == "__main__":
    main()

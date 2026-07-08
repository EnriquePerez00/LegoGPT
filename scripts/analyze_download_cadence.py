import os
import sys
import time
import json
import sqlite3
import random
import argparse
from datetime import datetime
from playwright.sync_api import sync_playwright
from scripts.download_bricklink_model import USER_AGENTS, human_delay

def extract_metadata(page) -> dict:
    """Extracts description, tags, downloads, views, and creator from the detail page DOM."""
    metadata = {}
    try:
        # Fallback selectors for description
        desc_el = page.locator('.design-detail__description, .moc-detail__description, .moc-desc, p.description').first
        metadata['description'] = desc_el.inner_text().strip() if desc_el.count() > 0 else "No description"
    except:
        metadata['description'] = "No description"
        
    try:
        # Extract tags
        tags = page.evaluate("""
        () => {
            const tagEls = document.querySelectorAll('.design-detail__tag, .moc-detail__tag, a[href*="tags="]');
            return Array.from(tagEls).map(el => el.innerText.trim()).filter(t => t.length > 0);
        }
        """)
        metadata['tags'] = ",".join(tags)
    except:
        metadata['tags'] = ""
        
    try:
        # Extract downloads/views counts
        stats = page.evaluate("""
        () => {
            const stats = { downloads: 0, views: 0 };
            const text = document.body.innerText;
            const dlMatch = text.match(/([0-9,]+)\\s*Downloads/i);
            const viewMatch = text.match(/([0-9,]+)\\s*Views/i);
            if (dlMatch) stats.downloads = parseInt(dlMatch[1].replace(/,/g, '')) || 0;
            if (viewMatch) stats.views = parseInt(viewMatch[1].replace(/,/g, '')) || 0;
            return stats;
        }
        """)
        metadata['downloads_count'] = stats['downloads']
        metadata['views_count'] = stats['views']
    except:
        metadata['downloads_count'] = 0
        metadata['views_count'] = 0
        
    return metadata

def execute_write_query(db_path: str, query: str, params: tuple = ()):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def run_cadence_experiment(db_path: str = "data/catalog/models_catalog.db", batch_size: int = 40):
    os.makedirs("scratch", exist_ok=True)
    os.makedirs("data/bricklink_raw", exist_ok=True)
    
    # Fetch pending models and close connection immediately
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT model_id FROM bricklink_scraping_queue 
    WHERE status = 'pending' 
    LIMIT ?
    """, (batch_size,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("[Profiler] No hay modelos en estado 'pending' en la cola.")
        return
        
    model_ids = [r[0] for r in rows]
    print(f"[Profiler] Iniciando prueba con lote de {len(model_ids)} modelos...")
    
    history = []
    current_delay = 8.0  # Starting delay
    consecutive_successes = 0
    
    ua = random.choice(USER_AGENTS)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = browser.new_context(user_agent=ua, storage_state="scratch/auth_state.json")
        page = context.new_page()
        page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")
        
        for idx, model_id in enumerate(model_ids):
            print(f"\n--- [{idx+1}/{len(model_ids)}] Procesando ID: {model_id} | Delay actual: {current_delay:.1f}s ---")
            
            # Mark as downloading using short-lived connection
            execute_write_query(
                db_path,
                "UPDATE bricklink_scraping_queue SET status = 'downloading', attempts = attempts + 1, last_attempt_time = ? WHERE model_id = ?",
                (datetime.now().isoformat(), model_id)
            )
            
            url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
            start_time = time.time()
            
            try:
                # Load design details page
                page.goto(url, wait_until="load", timeout=30000)
                human_delay(2.0, 3.5)
                
                # Check for Cloudflare challenge or Access Denied
                page_title = page.title()
                if "Access Denied" in page_title or "Cloudflare" in page_title or "Attention Required" in page_title:
                    raise Exception(f"WAF Blocked: Page title is '{page_title}'")
                    
                # 1. Parse Metadata
                metadata = extract_metadata(page)
                
                # 2. Trigger Download
                output_path = f"data/bricklink_raw/{model_id}.io"
                
                # Click the Download button
                download_btn = page.locator('span:has-text("Download"), a:has-text("Download"), button:has-text("Download")').first
                if download_btn.count() == 0:
                    raise Exception("Boton de descarga no encontrado en la pagina.")
                    
                with page.expect_download(timeout=20000) as download_info:
                    download_btn.click()
                    
                download = download_info.value
                download.save_as(output_path)
                
                # Success!
                elapsed = time.time() - start_time
                print(f"[SUCCESS] Guardado en: {output_path} (Tomó {elapsed:.1f}s)")
                
                # Save metadata to sets database using short-lived connection
                execute_write_query(
                    db_path,
                    "UPDATE sets SET description = ?, tags = ?, downloads_count = ?, views_count = ? WHERE set_id = ?",
                    (metadata['description'], metadata['tags'], metadata['downloads_count'], metadata['views_count'], model_id)
                )
                
                # Update queue status to completed
                execute_write_query(
                    db_path,
                    "UPDATE bricklink_scraping_queue SET status = 'completed', error_msg = NULL WHERE model_id = ?",
                    (model_id,)
                )
                
                # Record metrics
                history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model_id": model_id,
                    "status": "success",
                    "delay_used": current_delay,
                    "elapsed_seconds": elapsed,
                    "likes": metadata.get('likes_count', 0)
                })
                
                # Adjust cadence dynamically
                consecutive_successes += 1
                if consecutive_successes >= 5:
                    old_delay = current_delay
                    current_delay = max(2.0, current_delay - 1.0)
                    consecutive_successes = 0
                    print(f"[Cadencia] Reduciendo delay: {old_delay:.1f}s -> {current_delay:.1f}s (5 exitos seguidos)")
                    
                # Wait using the current delay before next loop
                time.sleep(current_delay + random.uniform(0.5, 1.5))
                
            except Exception as err:
                elapsed = time.time() - start_time
                err_msg = str(err)
                print(f"[FAILURE] ID {model_id} falló: {err_msg}")
                
                # Mark as failed in queue using short-lived connection
                execute_write_query(
                    db_path,
                    "UPDATE bricklink_scraping_queue SET status = 'failed', error_msg = ? WHERE model_id = ?",
                    (err_msg, model_id)
                )
                
                # Record metrics
                history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model_id": model_id,
                    "status": "failed",
                    "delay_used": current_delay,
                    "elapsed_seconds": elapsed,
                    "error": err_msg
                })
                
                # Reset successes and penalize delay
                consecutive_successes = 0
                old_delay = current_delay
                current_delay = min(30.0, current_delay * 2.0)
                print(f"[Cadencia] ¡Bloqueo o Fallo! Penalizando delay: {old_delay:.1f}s -> {current_delay:.1f}s")
                
                # If WAF block is detected, trigger cooling period
                if "WAF" in err_msg or "Access Denied" in err_msg:
                    print("[Enfriamiento] Activando pausa de seguridad de 3 minutos...")
                    time.sleep(180)
                else:
                    time.sleep(5)
                    
        browser.close()
    
    # Save history to analysis JSON file
    analysis_path = "scratch/cadence_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "history": history,
            "final_delay": current_delay
        }, f, indent=2)
        
    print(f"\n[Profiler] Analisis de cadencia guardado con exito en: {analysis_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="data/catalog/models_catalog.db")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    
    run_cadence_experiment(args.db_path, args.limit)

import os
import sys
import sqlite3
import random
import time
import re
import urllib.parse
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_bricklink_model import USER_AGENTS, simulate_scroll, human_delay, long_inactivity_pause

DB_PATH = "data/catalog/models_catalog.db"
OUTPUT_DIR = "data/bricklink_raw"

def clean_string(s):
    if not s:
        return ""
    return "".join(c.lower() for c in s if c.isalnum())

def get_search_queries(name):
    # 1. Base cleaning
    q = re.sub(r'\(.*?\)', '', name) # Remove parenthesis content
    q = re.sub(r'_(Copy|v\d+|\d+)', '', q, flags=re.IGNORECASE) # Remove _Copy, _2, _v2, etc.
    q = re.sub(r'\b(19|20)\d{2}\b', '', q) # Remove years
    q = q.replace('-', ' ').replace('_', ' ')
    q = re.sub(r'\s+', ' ', q).strip()
    
    queries = [q]
    
    words = q.split()
    if len(words) > 2:
        q3 = " ".join(words[:3])
        if q3 not in queries:
            queries.append(q3)
        q2 = " ".join(words[:2])
        if q2 not in queries:
            queries.append(q2)
            
    seen = set()
    result = []
    for query in queries:
        cleaned_q = query.strip()
        if cleaned_q and cleaned_q.lower() not in seen:
            seen.add(cleaned_q.lower())
            result.append(cleaned_q)
    return result

class AdaptiveDelay:
    def __init__(self, min_delay=2.0, max_delay=30.0, initial_delay=3.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = initial_delay
        self.success_streak = 0

    def get_delay(self):
        return random.uniform(self.current_delay, self.current_delay + 1.5)

    def report_success(self):
        self.success_streak += 1
        if self.success_streak >= 5:
            self.current_delay = max(self.min_delay, self.current_delay - 0.5)
            self.success_streak = 0
            print(f"  [~] Ritmo Adaptativo: 5 éxitos seguidos. Acelerando. Retraso base = {self.current_delay:.2f}s")

    def report_failure(self, reason="timeout/block"):
        self.success_streak = 0
        self.current_delay = min(self.max_delay, self.current_delay * 2.0)
        print(f"  [!] Ritmo Adaptativo: Fallo ({reason}). Ralentizando. Retraso base = {self.current_delay:.2f}s")

def search_google_for_model_id(page, name):
    print(f"  [Google] Buscando ID para: '{name}'...")
    query = f'"{name}" bricklink studio design'
    q_encoded = urllib.parse.quote(query)
    google_url = f"https://www.google.com/search?q={q_encoded}"
    
    try:
        page.goto(google_url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(3000)
        
        # Check cookie consent
        try:
            reject_btn = page.locator('button:has-text("Rechazar todo"), button:has-text("Reject all"), button:has-text("Antes de ir a Google")')
            if reject_btn.count() > 0:
                reject_btn.first.click()
                page.wait_for_timeout(1000)
        except Exception:
            pass
            
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
        
        for l in links:
            decoded = urllib.parse.unquote(l)
            if "idModel=" in decoded:
                model_id = decoded.split("idModel=")[-1].split("&")[0]
                if model_id.isdigit():
                    print(f"    [Google] Encontrado ID en URL: {model_id}")
                    return model_id
    except Exception as e:
        print(f"    [-] Error en búsqueda de Google: {e}")
    return None

def search_bricklink_for_model_id(page, name):
    print(f"  [BrickLink] Buscando ID en galería interna para: '{name}'...")
    queries = get_search_queries(name)
    
    for q in queries:
        try:
            query_encoded = urllib.parse.quote(q)
            page.goto(f"https://www.bricklink.com/v3/studio/gallery.page?q={query_encoded}", wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(4000)
            
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
            
            target_clean = clean_string(q)
            # Exact check
            for card in cards:
                if target_clean == clean_string(card["name"]):
                    print(f"    [BrickLink] Coincidencia exacta: '{card['name']}' -> ID {card['id']}")
                    return card["id"]
                    
            # Token overlap check (70% matching words)
            if cards:
                target_words = set(re.findall(r'\b\w+\b', q.lower()))
                for card in cards:
                    card_words = set(re.findall(r'\b\w+\b', card["name"].lower()))
                    if target_words and card_words:
                        intersection = target_words.intersection(card_words)
                        overlap = len(intersection) / len(target_words)
                        if overlap >= 0.70:
                            print(f"    [BrickLink] Coincidencia por solapamiento ({overlap:.2f}): '{card['name']}' -> ID {card['id']}")
                            return card["id"]
        except Exception as e:
            print(f"    [-] Error buscando en galería de BrickLink con query '{q}': {e}")
            
    return None

def main(limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Iniciando Proceso Unificado de Ingesta (URLs, Imágenes y Archivos .io) ===")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get sets missing image_url
    query = """
        SELECT set_id, name, source_url 
        FROM sets 
        WHERE source = 'BrickLink' AND image_url IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"
        
    rows = c.execute(query).fetchall()
    conn.close()
    
    print(f"Total de modelos a procesar: {len(rows)}")
    if not rows:
        print("No hay modelos pendientes.")
        return
        
    delay_controller = AdaptiveDelay()
    ua = random.choice(USER_AGENTS)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")
        
        # Load Google first to establish a general context
        try:
            page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(2000)
        except Exception:
            pass
            
        for idx, (set_id, name, src_url) in enumerate(rows):
            print(f"\n[{idx+1}/{len(rows)}] Procesando '{name}' (ID/Clave: {set_id})...")
            
            model_id = None
            
            # If set_id is numeric, use it directly
            if set_id.isdigit():
                model_id = set_id
                print(f"  [+] ID numérico detectado directamente en set_id: {model_id}")
            # If source_url contains idModel
            elif src_url and "idModel=" in src_url:
                model_id = src_url.split("idModel=")[-1].split("&")[0]
                if model_id.isdigit():
                    print(f"  [+] ID extraído de source_url: {model_id}")
            
            # If no model ID found, resolve it
            if not model_id:
                # Step A: Google Search
                model_id = search_google_for_model_id(page, name)
                
                # Step B: Fallback to BrickLink Gallery Search
                if not model_id:
                    model_id = search_bricklink_for_model_id(page, name)
                    
            if not model_id:
                print("  [-] No se pudo resolver un ID de modelo para este set. Omitiendo.")
                continue
                
            # Now we have model_id! Update source_url
            design_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
            
            # Scrape page for images and download file
            print(f"  [~] Cargando ficha de diseño: {design_url}")
            try:
                page.goto(design_url, wait_until="domcontentloaded", timeout=25000)
                page.wait_for_timeout(3000)
                
                # Check block
                title = page.title().lower()
                if "cloudflare" in title or "attention required" in title or "security check" in title:
                    print("  [!] Bloqueo detectado (WAF/Cloudflare). Pausando 45s...")
                    time.sleep(45)
                    delay_controller.report_failure("WAF Block")
                    continue
                    
                # Extract image URLs
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
                main_img = found_urls[0] if found_urls else None
                
                if main_img:
                    print(f"  [+] Imagen principal: {main_img}")
                    print(f"  [+] Carrusel: {len(found_urls)} imágenes encontradas.")
                    
                    # Update database with source_url and image_url
                    db_conn = sqlite3.connect(DB_PATH)
                    db_cur = db_conn.cursor()
                    db_cur.execute("""
                        UPDATE sets 
                        SET source_url = ?, image_url = ? 
                        WHERE set_id = ?
                    """, (design_url, main_img, set_id))
                    
                    # Insert all carrusel images
                    for url in found_urls:
                        db_cur.execute("""
                            INSERT OR IGNORE INTO set_images (set_id, image_url, source) 
                            VALUES (?, ?, 'BrickLink')
                        """, (set_id, url))
                    db_conn.commit()
                    db_conn.close()
                else:
                    print("  [-] No se hallaron imágenes del modelo en su ficha.")
                    
                # Download .io file if available
                download_btn = page.locator('button[data-ts-name="studio-model__meta-button--download"]').first
                if download_btn.count() > 0:
                    print("  [+] Botón de descarga detectado. Descargando .io...")
                    try:
                        simulate_scroll(page)
                        long_inactivity_pause()
                        
                        # Click to download
                        with page.expect_download(timeout=15000) as download_info:
                            download_btn.click()
                        
                        download = download_info.value
                        filename = download.suggested_filename
                        save_path = os.path.join(OUTPUT_DIR, filename)
                        download.save_as(save_path)
                        
                        # Update file_path in DB
                        db_conn = sqlite3.connect(DB_PATH)
                        db_cur = db_conn.cursor()
                        db_cur.execute("UPDATE sets SET file_path = ? WHERE set_id = ?", (save_path, set_id))
                        db_conn.commit()
                        db_conn.close()
                        print(f"  [+] Archivo .io guardado en: {save_path}")
                    except Exception as download_err:
                        print(f"  [-] Fallo al intentar descargar el archivo .io: {download_err}")
                else:
                    print("  [~] El MOC es de exhibición solamente (no descargable).")
                    
                delay_controller.report_success()
                
            except Exception as e:
                print(f"  [-] Error procesando la página para el set: {e}")
                delay_controller.report_failure("Page navigation error")
                
            # Adaptive delay pacing
            pacing = delay_controller.get_delay()
            page.wait_for_timeout(int(pacing * 1000))
            
        browser.close()
    print("\n=== Proceso Unificado Finalizado ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limita el número de sets a procesar")
    args = parser.parse_args()
    main(args.limit)

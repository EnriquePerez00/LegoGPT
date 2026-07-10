import os
import sys
import sqlite3
import random
import time
import re
import urllib.parse
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.download_bricklink_model import USER_AGENTS

DB_PATH = "data/catalog/models_catalog.db"

def is_numeric_model_url(url):
    if not url or "idModel=" not in url:
        return False
    model_id = url.split("idModel=")[-1].split("&")[0]
    return model_id.isdigit()

def clean_string(s):
    if not s:
        return ""
    return "".join(c.lower() for c in s if c.isalnum())

class AdaptiveDelay:
    def __init__(self, min_delay=1.2, max_delay=20.0, initial_delay=1.5):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = initial_delay
        self.success_streak = 0

    def get_delay(self):
        return random.uniform(self.current_delay, self.current_delay + 0.8)

    def report_success(self):
        self.success_streak += 1
        if self.success_streak >= 5:
            self.current_delay = max(self.min_delay, self.current_delay - 0.4)
            self.success_streak = 0
            print(f"  [~] Adaptive Rate: 5 successful requests in a row. Speeding up! Base delay = {self.current_delay:.2f}s")

    def report_failure(self, reason="timeout/block"):
        self.success_streak = 0
        self.current_delay = min(self.max_delay, self.current_delay * 2.0)
        print(f"  [!] Adaptive Rate: Failure ({reason}). Slowing down! Base delay = {self.current_delay:.2f}s")

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
            
    # Brand/model specific fallbacks
    q_lower = q.lower()
    if "challenger" in q_lower:
        queries.append("Challenger Hellcat")
        queries.append("Dodge Challenger")
        queries.append("Challenger")
    if "tatra" in q_lower:
        queries.append("Tram Tatra")
        queries.append("Tatra 3")
        queries.append("Tatra")
    if "rally car" in q_lower:
        queries.append("Rally Car")
    if "convertible" in q_lower:
        queries.append("convertible")
    if "fiat" in q_lower:
        queries.append("fiat 500")
        
    seen = set()
    result = []
    for query in queries:
        cleaned_q = query.strip()
        if cleaned_q and cleaned_q.lower() not in seen:
            seen.add(cleaned_q.lower())
            result.append(cleaned_q)
    return result

def main():
    print("=== Iniciando Resolución Avanzada Directa y Cosecha de Imágenes ===")
    delay_controller = AdaptiveDelay(min_delay=1.2, max_delay=20.0, initial_delay=1.5)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Get all sets missing images (BrickLink)
    rows = c.execute("""
        SELECT set_id, name, source_url, (level_1_entorno IS NOT NULL AND level_1_entorno != 'Otros') as is_vehicle
        FROM sets 
        WHERE source = 'BrickLink' AND image_url IS NULL
    """).fetchall()
    
    conn.close()
    
    targets = []
    for row in rows:
        targets.append((row[0], row[1], row[2], bool(row[3])))
        
    print(f"Total a procesar: {len(targets)} sets.")
    if not targets:
        print("No hay sets pendientes de resolución de imagen.")
        return
        
    ua = random.choice(USER_AGENTS)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=ua)
        page = context.new_page()
        
        # Load gallery to establish cookie session
        try:
            page.goto("https://www.bricklink.com/v3/studio/gallery.page", wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(3000)
            # Dismiss overlay modal if visible
            overlay = page.locator('.custom-confirm__btn, .custom-confirm__btn--primary, button:has-text("Accept"), button:has-text("OK")').first
            if overlay.is_visible():
                overlay.click()
                print("  [~] Dismissed cookie/confirm overlay.")
        except Exception:
            pass
            
        for idx, (set_id, name, src_url, is_vehicle) in enumerate(targets):
            type_str = "Vehículo" if is_vehicle else "Otros"
            print(f"\n[{idx+1}/{len(targets)}] [{type_str}] Procesando '{name}' ({set_id})...")
            
            # Resolve URL if it's missing or non-numeric
            if not is_numeric_model_url(src_url):
                queries = get_search_queries(name)
                print(f"  [!] URL inválida o no-numérica detected. Consultas de búsqueda a probar: {queries}")
                resolved = False
                
                for q in queries:
                    print(f"  [~] Probando búsqueda directa: '{q}'...")
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
                        
                        matched_id = None
                        target_clean = clean_string(q)
                        
                        # 1. Exact match check
                        for card in cards:
                            card_clean = clean_string(card["name"])
                            if target_clean == card_clean:
                                matched_id = card["id"]
                                print(f"  [+] Exact match found: '{card['name']}' -> ID {matched_id}")
                                break
                                
                        # 2. Confident token-based overlap check
                        if not matched_id and cards:
                            target_words = set(re.findall(r'\b\w+\b', q.lower()))
                            for card in cards:
                                card_words = set(re.findall(r'\b\w+\b', card["name"].lower()))
                                if target_words and card_words:
                                    intersection = target_words.intersection(card_words)
                                    overlap = len(intersection) / len(target_words)
                                    # Accept if overlap is high (at least 70% of query words match)
                                    if overlap >= 0.70:
                                        matched_id = card["id"]
                                        print(f"  [~] Token overlap match found (overlap {overlap:.2f}): '{card['name']}' -> ID {matched_id}")
                                        break
                                        
                        if matched_id:
                            src_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={matched_id}"
                            conn = sqlite3.connect(DB_PATH)
                            cur = conn.cursor()
                            cur.execute("UPDATE sets SET source_url = ? WHERE set_id = ?", (src_url, set_id))
                            conn.commit()
                            conn.close()
                            print(f"  [+] URL resuelta exitosamente con '{q}': {src_url}")
                            resolved = True
                            break
                    except Exception as e:
                        print(f"  [-] Error probando búsqueda '{q}': {e}")
                        delay_controller.report_failure(f"search query '{q}' error")
                
                if not resolved:
                    # If set_id itself is numeric, fallback to that
                    if set_id.isdigit():
                        src_url = f"https://www.bricklink.com/v3/studio/design.page?idModel={set_id}"
                    else:
                        print(f"  [-] No se pudo resolver URL válida con ninguna consulta. Omitiendo.")
                        continue
                    
            # Scrape page for images
            model_id = src_url.split("idModel=")[-1].split("&")[0]
            print(f"  [+] Cosechando imágenes para ID {model_id}...")
            try:
                page.goto(src_url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(3000)
                
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
                    print(f"  [+] Principal: {main_img}")
                    print(f"  [+] Encontradas: {len(found_urls)} en carrusel.")
                    
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("UPDATE sets SET image_url = ? WHERE set_id = ?", (main_img, set_id))
                    for url in found_urls:
                        cur.execute("INSERT OR IGNORE INTO set_images (set_id, image_url, source) VALUES (?, ?, 'BrickLink')", (set_id, url))
                    conn.commit()
                    conn.close()
                else:
                    print("  [-] No se hallaron imágenes válidas del modelo en la página.")
            except Exception as e:
                print(f"  [-] Error raspando la página: {e}")
                
            # Check if we were blocked by WAF/Cloudflare
            try:
                title = page.title().lower()
                if "cloudflare" in title or "attention required" in title or "security check" in title:
                    print("  [!] WAF/Cloudflare block page detected!")
                    delay_controller.report_failure("WAF Block Page")
                    # Cool-down sleep
                    print("  [~] Cooling down for 45 seconds...")
                    time.sleep(45)
                else:
                    delay_controller.report_success()
            except Exception:
                pass

            curr_delay = delay_controller.get_delay()
            # print(f"  [~] Pacing: waiting {curr_delay:.2f} seconds before next set...")
            page.wait_for_timeout(int(curr_delay * 1000))
            
        browser.close()
        
    print("\n=== Resolución Finalizada ===")

if __name__ == "__main__":
    main()

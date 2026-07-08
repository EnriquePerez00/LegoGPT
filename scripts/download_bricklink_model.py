import os
import sys
import time
import random
import argparse
from playwright.sync_api import sync_playwright

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
]

def human_delay(min_sec=2.0, max_sec=5.0):
    """Introduces a randomized delay to simulate human reaction times."""
    delay = random.uniform(min_sec, max_sec)
    print(f"[Human Sim] Pausando por {delay:.2f} segundos...")
    time.sleep(delay)

def long_inactivity_pause():
    """Simulates a longer period of reading or examining page details."""
    delay = random.randint(20, 45)
    print(f"[Human Sim] Pausa larga (simulando lectura) por {delay} segundos...")
    time.sleep(delay)

def simulate_scroll(page):
    """Simulates smooth mouse scrolling behavior."""
    print("[Human Sim] Simulando scroll de página...")
    for _ in range(random.randint(2, 5)):
        # Scroll down by random offset
        scroll_y = random.randint(200, 600)
        page.evaluate(f"window.scrollBy(0, {scroll_y})")
        human_delay(1.5, 3.5)
    
    # Scroll back up slightly
    scroll_up = random.randint(100, 300)
    page.evaluate(f"window.scrollBy(0, -{scroll_up})")
    human_delay(1.0, 2.5)

def download_bricklink_model(model_id: str, output_dir: str = "data/bricklink_raw") -> bool:
    os.makedirs(output_dir, exist_ok=True)
    
    url = f"https://www.bricklink.com/v3/studio/design.page?idModel={model_id}"
    print(f"Iniciando descarga simulada para el modelo {model_id}...")
    print(f"Target URL: {url}")
    
    ua = random.choice(USER_AGENTS)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ]
        )
        
        # Configure context to simulate a clean Mac user session
        auth_state = "scratch/auth_state.json"
        storage = auth_state if os.path.exists(auth_state) else None

        context = browser.new_context(
            user_agent=ua,
            viewport={"width": 1280, "height": 800},
            locale="es-ES",
            timezone_id="Europe/Madrid",
            storage_state=storage
        )
        
        # Anti-bot JS masking
        page = context.new_page()
        page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")
        
        print("[Navegación] Cargando página de BrickLink...")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            human_delay(3.0, 7.0)
            
            # Check for Cloudflare CAPTCHA trigger
            title = page.title().lower()
            if "cloudflare" in title or "attention required" in title or "just a moment" in title:
                print("[ERROR] Cloudflare bloqueó la petición (JS Challenge o CAPTCHA detectado).")
                browser.close()
                return False, None, None
                
            print(f"[Navegación] Título de página: {page.title()}")
            
            # Extract preview image URL from OpenGraph meta tags
            image_url = None
            try:
                og_image_el = page.locator('meta[property="og:image"]')
                if og_image_el.count() > 0:
                    image_url = og_image_el.get_attribute("content")
                    print(f"[Navegación] Imagen de previsualización encontrada: {image_url}")
            except Exception as img_err:
                print(f"[Navegación Warning] No se pudo extraer og:image: {img_err}")
            
            # Simulate human looking at the page details (scrolling)
            simulate_scroll(page)
            
            # Longer pause to read description / details
            long_inactivity_pause()
            
            # Look for the download button in the page structure
            download_btn = page.locator('button[data-ts-name="studio-model__meta-button--download"]').first
            
            if download_btn.count() > 0:
                print("[Acción] Botón de descarga encontrado. Simulando hover...")
                download_btn.hover()
                human_delay(1.5, 3.0)
                
                print("[Acción] Haciendo click en el botón de descarga...")
                # Expect download event
                with page.expect_download(timeout=10000) as download_info:
                    download_btn.click()
                    
                download = download_info.value
                filename = download.suggested_filename
                save_path = os.path.join(output_dir, filename)
                
                download.save_as(save_path)
                print(f"[Éxito] Modelo descargado y guardado en: {save_path}")
                
                # Final inactivity pause after download
                human_delay(4.0, 8.0)
                browser.close()
                return True, image_url, save_path
            else:
                print("[ERROR] No se pudo encontrar el botón de descarga en la página.")
                # Save page screenshot for debugging
                screenshot_path = os.path.join(output_dir, f"error_{model_id}.png")
                page.screenshot(path=screenshot_path)
                print(f"Guardada captura de pantalla en {screenshot_path}")
                browser.close()
                return False, None, None
                
        except Exception as e:
            print(f"[ERROR] Excepción durante la automatización: {e}")
            browser.close()
            return False, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="180126", help="BrickLink Model ID")
    parser.add_argument("--output_dir", type=str, default="data/bricklink_raw", help="Output directory")
    args = parser.parse_args()
    
    success, img_url, path = download_bricklink_model(args.model_id, args.output_dir)
    sys.exit(0 if success else 1)

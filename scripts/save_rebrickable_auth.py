#!/usr/bin/env python3
"""
Captura la sesión de Rebrickable conectándose a una ventana real de Google Chrome
del sistema (lanzada con puerto de depuración), donde Cloudflare Turnstile NO detecta Playwright.
"""

import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

CHROME_USER_DATA = os.path.abspath("scratch/chrome_real_profile")
AUTH_PATH = os.path.abspath("scratch/rebrickable_storage_state.json")
DEBUG_PORT = 9222

def launch_real_chrome_and_save():
    os.makedirs(CHROME_USER_DATA, exist_ok=True)
    os.makedirs("scratch", exist_ok=True)
    
    print("==================================================================")
    print("1. Abriendo tu Google Chrome REAL del Mac...")
    print("2. Resuelve el checkbox de Cloudflare e inicia sesión en Rebrickable.")
    print("3. Cuando veas tu perfil / portada de Rebrickable, pulsa ENTER")
    print("   aquí en la terminal.")
    print("==================================================================")
    
    # Ruta de Chrome en macOS
    chrome_app = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_app):
        print(f"[Error] No se encontró Google Chrome en {chrome_app}")
        return

    # Lanzar proceso real de Google Chrome sin flags de automatización de Playwright
    cmd = [
        chrome_app,
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={CHROME_USER_DATA}",
        "https://rebrickable.com/login/"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    with sync_playwright() as p:
        try:
            # Conectarse al Chrome real vía CDP (Chrome DevTools Protocol)
            browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{DEBUG_PORT}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            
            print("\nEsperando a que completes la verificación en la ventana de Chrome...")
            print("Presiona ENTER aquí cuando hayas superado el captcha / iniciado sesión.")
            
            # Bucle no bloqueante o esperar entrada
            for _ in range(150):
                cookies = context.cookies()
                c_names = [c["name"] for c in cookies]
                if "sessionid" in c_names or "cf_clearance" in c_names:
                    print(f"  [Cookie detectada] {', '.join(c_names)}")
                    context.storage_state(path=AUTH_PATH)
                    print(f"\n[¡ÉXITO!] Cookies guardadas en: {AUTH_PATH}")
                    break
                time.sleep(2)
                
            context.storage_state(path=AUTH_PATH)
            print(f"[Listo] Estado final guardado en: {AUTH_PATH}")
            browser.close()
        except Exception as e:
            print(f"[Aviso Playwright CDP] {e}")
            
    # Terminar proceso de chrome si sigue abierto
    proc.terminate()
    print("Ventana de autenticación cerrada.")

if __name__ == "__main__":
    launch_real_chrome_and_save()

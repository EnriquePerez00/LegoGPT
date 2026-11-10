#!/usr/bin/env python3
"""
Mapeador de URLs de manuales e identificador/descargador de ficheros .io internos en Rebrickable.
- Conecta directamente a la sesión real de Chrome para evitar Cloudflare Turnstile.
- Extrae la URL canónica de instrucciones (PDF o externa).
- Si encuentra un archivo .io no descargado, lo descarga automáticamente a data/models_io/.
- Almacena las URLs en la tabla alternate_builds (pdf_url, instructions_url).
"""

import os
import sys
import time
import sqlite3
import subprocess
import urllib.parse
import re
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage_guard import assert_disk_space_or_exit, get_disk_status

DB_PATH = "data/catalog/models_catalog.db"
CHROME_USER_DATA = os.path.abspath("scratch/chrome_real_profile")
IO_DIR = "data/models_io"
PORT = 9555

def ensure_columns():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(alternate_builds);")
    cols = [c[1] for c in cursor.fetchall()]
    if "pdf_url" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN pdf_url TEXT;")
    if "instructions_url" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN instructions_url TEXT;")
    conn.commit()
    conn.close()

def run_mapper(batch_size: int = 100):
    assert_disk_space_or_exit(min_free_gb=10.0)
    ensure_columns()
    os.makedirs(IO_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    cursor = conn.cursor()

    # Traer MOCs que no tengan instructions_url mapeada aún
    cursor.execute("""
        SELECT ab.moc_id, ab.parent_set_id, ab.name, ab.source_url, ab.io_file_path
        FROM alternate_builds ab
        WHERE ab.source_url IS NOT NULL
          AND ab.instructions_url IS NULL
        ORDER BY ab.num_parts DESC
        LIMIT ?;
    """, (batch_size,))
    queue = cursor.fetchall()

    if not queue:
        print("[Info] No hay MOCs pendientes en este lote.")
        conn.close()
        return

    print(f"[{time.strftime('%H:%M:%S')}] Iniciando mapeo de instrucciones para {len(queue)} MOCs...")
    
    chrome_app = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome_app,
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={CHROME_USER_DATA}",
        "about:blank"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    mapped_count = 0
    new_io_count = 0

    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else context.new_page()

            for idx, (moc_id, parent_set_id, name, source_url, existing_io) in enumerate(queue, 1):
                assert_disk_space_or_exit(min_free_gb=10.0)

                try:
                    page.goto(source_url, wait_until="domcontentloaded", timeout=25000)
                    time.sleep(1.2)

                    # Verificar si nos topamos con un captcha
                    if "Just a moment..." in page.title():
                        print(f"⚠️ [Cloudflare] Bloqueo detectado en {moc_id}. Deteniendo lote.")
                        break

                    # Buscar links de instrucciones
                    links = page.query_selector_all("a[href*='/external/view/?url='], a[href*='bi_file'], a[href*='.pdf'], a[href*='.io'], a[href*='download']")
                    
                    found_pdf = None
                    found_io = None
                    inst_url = None

                    for l in links:
                        href = l.get_attribute("href") or ""
                        # Extraer si es un external view wrapper de Rebrickable
                        if "/external/view/?url=" in href:
                            parsed = urllib.parse.urlparse(href)
                            params = urllib.parse.parse_qs(parsed.query)
                            target = params.get("url", [""])[0]
                            if target:
                                if target.endswith(".pdf"):
                                    found_pdf = target
                                elif target.endswith(".io") or target.endswith(".ldr"):
                                    found_io = target
                                inst_url = target
                                break
                        elif ".pdf" in href.lower() or "bi_file" in href.lower():
                            found_pdf = href if href.startswith("http") else f"https://rebrickable.com{href}"
                            inst_url = found_pdf
                            break
                        elif any(ext in href.lower() for ext in [".io", ".ldr"]):
                            found_io = href if href.startswith("http") else f"https://rebrickable.com{href}"
                            inst_url = found_io
                            break

                    if not inst_url:
                        # Buscar si hay un enlace directo en el cuerpo a YouTube o similar
                        yt_link = page.query_selector("a[href*='youtube.com'], a[href*='youtu.be']")
                        if yt_link:
                            inst_url = yt_link.get_attribute("href")
                        else:
                            inst_url = source_url

                    # Si hay un .io nuevo que no tengamos en local, descargarlo
                    if found_io and not existing_io:
                        try:
                            # Descargar .io
                            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:35]
                            dest_path = os.path.join(IO_DIR, f"{moc_id}_{safe_name}.io")
                            
                            req = urllib.request.Request(found_io, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req, timeout=15) as r:
                                with open(dest_path, "wb") as f_out:
                                    f_out.write(r.read())

                            cursor.execute("""
                                UPDATE alternate_builds
                                SET io_url = ?, io_file_path = ?, download_status = 'io_downloaded'
                                WHERE moc_id = ?;
                            """, (found_io, dest_path, moc_id))
                            new_io_count += 1
                            print(f"  [Nuevo .io descargado] {moc_id}: {os.path.basename(dest_path)}")
                        except Exception as e:
                            print(f"  [Aviso descarga .io {moc_id}] {e}")

                    cursor.execute("""
                        UPDATE alternate_builds
                        SET pdf_url = ?,
                            instructions_url = ?
                        WHERE moc_id = ?;
                    """, (found_pdf, inst_url, moc_id))
                    conn.commit()
                    mapped_count += 1

                    if idx % 10 == 0:
                        print(f"[{idx}/{len(queue)}] {moc_id}: Instrucciones -> {inst_url[:60]}")

                except Exception as e:
                    print(f"  [Error {moc_id}]: {e}")
                    time.sleep(1)

            browser.close()
    finally:
        proc.terminate()

    conn.close()
    print(f"\n=======================================================")
    print(f"Lote finalizado: {mapped_count} manuales mapeados, {new_io_count} nuevos .io descargados.")
    print(f"=======================================================")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_mapper(batch_size=count)

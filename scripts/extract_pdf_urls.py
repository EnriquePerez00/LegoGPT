#!/usr/bin/env python3
"""
Extractor continuo de URLs directas a manuales PDF de Rebrickable.
- Se conecta a una ventana de Google Chrome real abierta en el puerto 9222.
- Filtra estrictamente MOCs de Rebrickable (source = 'rebrickable_alt').
- Extrae la URL directa del PDF (/external/view/?url=...pdf o /media/bi_files/...pdf).
- Si encuentra un archivo .io no descargado previamente, lo descarga a data/models_io/.
- Guarda la url directa en alternate_builds.pdf_url.
"""

import os
import sys
import time
import sqlite3
import urllib.parse
import urllib.request
import re
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage_guard import assert_disk_space_or_exit, get_disk_status

DB_PATH = "data/catalog/models_catalog.db"
IO_DIR = "data/models_io"
DEBUG_PORT = 9222

def ensure_columns():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(alternate_builds);")
    cols = [c[1] for c in cursor.fetchall()]
    if "pdf_url" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN pdf_url TEXT;")
    if "has_pdf_instructions" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN has_pdf_instructions BOOLEAN DEFAULT 0;")
    if "instructions_scanned" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN instructions_scanned BOOLEAN DEFAULT 0;")
    conn.commit()
    conn.close()

def extract_pdf_urls_worker(max_items: int = 500):
    assert_disk_space_or_exit(min_free_gb=10.0)
    ensure_columns()
    os.makedirs(IO_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    cursor = conn.cursor()

    # Traer MOCs de Rebrickable pendientes de escaneo de instrucciones
    cursor.execute("""
        SELECT ab.moc_id, ab.parent_set_id, ab.name, ab.source_url, ab.io_file_path
        FROM alternate_builds ab
        WHERE ab.source = 'rebrickable_alt'
          AND ab.source_url IS NOT NULL
          AND ab.instructions_scanned = 0
        ORDER BY ab.parent_set_id ASC, ab.num_parts DESC
        LIMIT ?;
    """, (max_items,))
    queue = cursor.fetchall()

    if not queue:
        print("[Info] No hay MOCs pendientes de Rebrickable.")
        conn.close()
        return

    print(f"[{time.strftime('%H:%M:%S')}] Iniciando extracción de URLs PDF para {len(queue)} MOCs de Rebrickable...")
    print(f"Conectando a Chrome en http://127.0.0.1:{DEBUG_PORT} ...")

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{DEBUG_PORT}")
        except Exception as e:
            print(f"[Error de conexión] No se pudo conectar a Chrome en el puerto {DEBUG_PORT}: {e}")
            conn.close()
            return

        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()

        pdf_found_count = 0
        io_found_count = 0
        total_scanned = 0

        for idx, (moc_id, parent_set_id, name, source_url, existing_io) in enumerate(queue, 1):
            assert_disk_space_or_exit(min_free_gb=10.0)

            try:
                page.goto(source_url, wait_until="domcontentloaded", timeout=25000)
                time.sleep(1.2)

                # Si Cloudflare salta, esperar por si se resuelve en pantalla
                if "Just a moment..." in page.title():
                    print(f"⚠️ [Cloudflare en {moc_id}] Esperando 6s...")
                    time.sleep(6)
                    if "Just a moment..." in page.title():
                        print("  Aún bloqueado por Cloudflare. Deteniendo lote.")
                        break

                # Buscar enlaces a PDFs y ficheros
                links = page.query_selector_all("a[href*='/external/view/?url='], a[href*='bi_file'], a[href*='.pdf'], a[href*='.io'], a[href*='download']")
                found_pdf = None
                found_io = None

                for l in links:
                    href = l.get_attribute("href") or ""
                    
                    if "/external/view/?url=" in href:
                        parsed = urllib.parse.urlparse(href)
                        params = urllib.parse.parse_qs(parsed.query)
                        target = params.get("url", [""])[0]
                        if target:
                            if target.lower().endswith(".pdf"):
                                found_pdf = target
                            elif any(target.lower().endswith(ext) for ext in [".io", ".ldr", ".mpd"]):
                                found_io = target
                    elif ".pdf" in href.lower() or "bi_file" in href.lower():
                        found_pdf = href if href.startswith("http") else f"https://rebrickable.com{href}"
                    elif any(ext in href.lower() for ext in [".io", ".ldr", ".mpd"]):
                        found_io = href if href.startswith("http") else f"https://rebrickable.com{href}"

                # Si encontramos un .io y no lo teníamos en local, guardarlo
                if found_io and not existing_io:
                    try:
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
                        io_found_count += 1
                        print(f"  [Nuevo .io descargado] {moc_id}: {os.path.basename(dest_path)}")
                    except Exception as e:
                        print(f"  [Aviso descarga .io {moc_id}] {e}")

                # Registrar URL del PDF y marcar como escaneado
                cursor.execute("""
                    UPDATE alternate_builds
                    SET pdf_url = ?,
                        has_pdf_instructions = ?,
                        instructions_scanned = 1
                    WHERE moc_id = ?;
                """, (found_pdf, 1 if found_pdf else 0, moc_id))
                conn.commit()
                total_scanned += 1

                if found_pdf:
                    pdf_found_count += 1
                    print(f"  [{idx}/{len(queue)}] {moc_id} (Set {parent_set_id}): PDF -> {found_pdf[:65]}...")
                elif idx % 10 == 0:
                    print(f"  [{idx}/{len(queue)}] {moc_id}: Escaneado (sin PDF directo)")

            except Exception as e:
                print(f"  [Error en {moc_id}]: {e}")
                time.sleep(1)

        browser.close()

    conn.close()
    print(f"\n=======================================================")
    print(f"Resumen del lote:")
    print(f"  - MOCs procesados: {total_scanned}")
    print(f"  - URLs de PDF extraídas: {pdf_found_count}")
    print(f"  - Nuevos .io guardados: {io_found_count}")
    print(f"=======================================================")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    extract_pdf_urls_worker(max_items=count)

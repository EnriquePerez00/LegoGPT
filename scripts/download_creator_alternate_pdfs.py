#!/usr/bin/env python3
"""
Crawler de Playwright para descargar manuales .PDF de los Alternate Builds de Rebrickable.
- Filtrado por tema: Creator 3-in-1 (u otros).
- Respeta estrictamente el límite de seguridad de 10.00 GB de disco (storage_guard).
- Utiliza la sesión y cookies guardadas en scratch/rebrickable_storage_state.json.
- Registra en alternate_builds: pdf_file_path, pdf_url, download_status.
"""

import os
import sys
import time
import sqlite3
import re
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage_guard import assert_disk_space_or_exit, get_disk_status

DB_PATH = "data/catalog/models_catalog.db"
AUTH_PATH = "scratch/rebrickable_storage_state.json"
PDF_DIR = "data/manuals/mocs"

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Asegurar columnas necesarias en alternate_builds
    cursor.execute("PRAGMA table_info(alternate_builds);")
    cols = [c[1] for c in cursor.fetchall()]
    if "pdf_url" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN pdf_url TEXT;")
    if "pdf_file_path" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN pdf_file_path TEXT;")
    if "has_pdf_instructions" not in cols:
        cursor.execute("ALTER TABLE alternate_builds ADD COLUMN has_pdf_instructions BOOLEAN DEFAULT 0;")
    conn.commit()
    conn.close()

def get_creator_mocs_queue(limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ab.moc_id, ab.parent_set_id, ab.name, ab.source_url
        FROM alternate_builds ab
        JOIN sets s ON ab.parent_set_id = s.set_id
        WHERE s.theme = 'Creator 3-in-1' 
          AND (ab.pdf_file_path IS NULL OR ab.download_status != 'pdf_downloaded')
          AND ab.source = 'rebrickable_alt'
        ORDER BY ab.num_parts DESC
        LIMIT ?;
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def download_creator_alternate_pdfs(max_downloads: int = 50, headless: bool = True):
    assert_disk_space_or_exit(min_free_gb=10.0)
    setup_db()
    os.makedirs(PDF_DIR, exist_ok=True)
    
    mocs = get_creator_mocs_queue(limit=max_downloads * 2)
    if not mocs:
        print("[Info] No hay alternate builds pendientes de descarga para Creator 3-in-1.")
        return

    print(f"--- Iniciando descarga de manuales .PDF para Creator 3-in-1 ({len(mocs)} en cola) ---")
    disk = get_disk_status()
    print(f"Espacio libre en disco: {disk['free_gb']} GB (Margen de parada: 10.00 GB).")
    
    has_auth = os.path.exists(AUTH_PATH)
    if not has_auth:
        print(f"[Aviso] No se encontró {AUTH_PATH}. Se iniciará navegador sin cookies guardadas.")
    else:
        print(f"[Auth] Utilizando cookies y sesión de: {AUTH_PATH}")

    downloaded_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context_kwargs = {
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "viewport": {"width": 1280, "height": 850},
            "accept_downloads": True
        }
        if has_auth:
            context_kwargs["storage_state"] = AUTH_PATH
            
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for moc_id, parent_set_id, moc_name, source_url in mocs:
            assert_disk_space_or_exit(min_free_gb=10.0)
            if downloaded_count >= max_downloads:
                break

            print(f"\nProcesando [{moc_id}] {moc_name[:35]} (Set {parent_set_id})...")
            try:
                page.goto(source_url, wait_until="domcontentloaded", timeout=30000)
                time.sleep(2)

                # Comprobar si Cloudflare bloquea
                title = page.title()
                if "Just a moment" in title:
                    print(f"  [Cloudflare Interception] Detectada pantalla de verificación. Esperando 5s...")
                    time.sleep(5)
                    title = page.title()
                    if "Just a moment" in title:
                        print("  [Error] Bloqueado por Cloudflare. Necesario regenerar scratch/rebrickable_storage_state.json.")
                        break

                # Buscar pestaña o sección de Building Instructions
                # En Rebrickable suele haber un tab #bi o links directos con .pdf o 'Download'
                download_links = page.query_selector_all("a[href*='download'], a[href$='.pdf'], a[href*='bi_file']")
                pdf_found = False

                for link in download_links:
                    href = link.get_attribute("href") or ""
                    text = link.inner_text().lower()
                    
                    if ".pdf" in href.lower() or "pdf" in text or "instruction" in text:
                        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', moc_name)[:40]
                        filename = f"{moc_id}_{safe_title}.pdf"
                        target_path = os.path.join(PDF_DIR, filename)

                        print(f"  Encontrado enlace de descarga: {href[:60]}...")
                        try:
                            with page.expect_download(timeout=15000) as download_info:
                                link.click()
                            download = download_info.value
                            download.save_as(target_path)

                            # Actualizar BD
                            cursor.execute("""
                                UPDATE alternate_builds
                                SET pdf_file_path = ?,
                                    pdf_url = ?,
                                    has_pdf_instructions = 1,
                                    download_status = 'pdf_downloaded'
                                WHERE moc_id = ?;
                            """, (target_path, href, moc_id))
                            conn.commit()
                            
                            downloaded_count += 1
                            pdf_found = True
                            print(f"  [Éxito] PDF guardado en: {target_path}")
                            break
                        except Exception as dl_err:
                            print(f"  [Error al descargar] {dl_err}")

                if not pdf_found:
                    cursor.execute("""
                        UPDATE alternate_builds
                        SET download_status = 'no_direct_pdf'
                        WHERE moc_id = ?;
                    """, (moc_id,))
                    conn.commit()
                    print(f"  [Sin PDF directo o de pago] Estado actualizado a 'no_direct_pdf'.")

            except Exception as e:
                print(f"  [Excepción de navegación] {e}")

            time.sleep(1.5)

        conn.close()
        browser.close()

    print(f"\n[Finalizado] Total manuales PDF descargados: {downloaded_count}")
    print(f"Espacio libre en disco: {get_disk_status()['free_gb']} GB")

if __name__ == "__main__":
    download_creator_alternate_pdfs(max_downloads=30)

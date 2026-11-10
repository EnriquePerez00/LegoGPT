#!/usr/bin/env python3
"""
Descargador de archivos de diseño 3D (.io / .ldr / .mpd) vinculados en alternate_builds.
- Descarga los archivos a data/models_io/
- Actualiza alternate_builds.io_file_path y alternate_builds.download_status = 'io_downloaded'
- Respeta estrictamente el límite de seguridad de 10.00 GB de disco (storage_guard).
"""

import os
import sys
import time
import sqlite3
import ssl
import urllib.request
import urllib.parse
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage_guard import assert_disk_space_or_exit, get_disk_status

DB_PATH = "data/catalog/models_catalog.db"
IO_DIR = "data/models_io"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def download_available_io_files(limit: int = 50):
    assert_disk_space_or_exit(min_free_gb=10.0)
    os.makedirs(IO_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT moc_id, name, io_url
        FROM alternate_builds
        WHERE io_url IS NOT NULL 
          AND (io_file_path IS NULL OR download_status != 'io_downloaded')
        LIMIT ?;
    """, (limit,))
    models_to_download = cursor.fetchall()

    if not models_to_download:
        print("[Info] No hay nuevos archivos .io pendientes de descarga.")
        conn.close()
        return

    print(f"--- Descargando {len(models_to_download)} archivos .io ---")
    downloaded = 0

    for moc_id, name, url in models_to_download:
        assert_disk_space_or_exit(min_free_gb=10.0)

        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".io"
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:35]
        local_filename = f"{moc_id}_{safe_name}{ext}"
        local_path = os.path.join(IO_DIR, local_filename)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LegoGPT-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
                content = r.read()
                with open(local_path, "wb") as f:
                    f.write(content)

            cursor.execute("""
                UPDATE alternate_builds
                SET io_file_path = ?,
                    download_status = 'io_downloaded'
                WHERE moc_id = ?;
            """, (local_path, moc_id))
            conn.commit()
            downloaded += 1
            print(f"  [{downloaded}/{len(models_to_download)}] Guardado: {local_path} ({len(content)//1024} KB)")
        except Exception as e:
            print(f"  [Error descargando {moc_id}] {e}")

        time.sleep(0.3)

    conn.close()
    print(f"\nDescarga finalizada: {downloaded} modelos descargados.")
    print(f"Espacio libre en disco: {get_disk_status()['free_gb']} GB")

if __name__ == "__main__":
    download_available_io_files(limit=100)

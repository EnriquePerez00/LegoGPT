#!/usr/bin/env python3
"""
Rastreador y descargador continuo de archivos .io / .ldr desde Bricksafe
con emparejamiento semántico robusto (Set ID + Nombre).
"""

import os
import sys
import time
import sqlite3
import ssl
import urllib.request
import urllib.parse
import re
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage_guard import assert_disk_space_or_exit, get_disk_status

DB_PATH = "data/catalog/models_catalog.db"
IO_DIR = "data/models_io"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def clean_tokens(text: str) -> set:
    text = urllib.parse.unquote(text).lower()
    tokens = set(re.findall(r'[a-z0-9]+', text))
    # Descartar extensiones y palabras comunes
    tokens.difference_update({"io", "ldr", "mpd", "pdf", "the", "and", "alt", "alternate", "build", "lego", "files", "pages", "copy"})
    return tokens

def scan_designer_bricksafe(designer_name: str) -> list:
    encoded_name = urllib.parse.quote(designer_name.strip())
    url = f"https://bricksafe.com/pages/{encoded_name}"
    req = urllib.request.Request(url, headers={"User-Agent": "LegoGPT-Crawler/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=12, context=ctx) as r:
            html = r.read().decode("utf-8", errors="ignore")
            subpages = re.findall(rf'href=[\"\'](/pages/{re.escape(encoded_name)}/[^\"\']+)[\"\']', html)
            
            found_files = []
            root_files = re.findall(rf'href=[\"\'](/files/{re.escape(encoded_name)}/[^\"\']+\.(?:io|ldr|mpd))[\"\']', html, re.I)
            found_files.extend(root_files)

            for sp in subpages[:20]:
                sp_url = f"https://bricksafe.com{sp}"
                try:
                    sp_req = urllib.request.Request(sp_url, headers={"User-Agent": "LegoGPT-Crawler/1.0"})
                    with urllib.request.urlopen(sp_req, timeout=10, context=ctx) as sp_r:
                        sp_html = sp_r.read().decode("utf-8", errors="ignore")
                        f_matches = re.findall(rf'href=[\"\'](/files/{re.escape(encoded_name)}/[^\"\']+\.(?:io|ldr|mpd))[\"\']', sp_html, re.I)
                        found_files.extend(f_matches)
                except Exception:
                    pass
                time.sleep(0.1)

            return list(set(found_files))
    except Exception:
        return []

def download_file(url: str, moc_id: str, name: str) -> str:
    os.makedirs(IO_DIR, exist_ok=True)
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".io"
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:35]
    local_path = os.path.join(IO_DIR, f"{moc_id}_{safe_name}{ext}")

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    req = urllib.request.Request(url, headers={"User-Agent": "LegoGPT-Downloader/1.0"})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
        content = r.read()
        with open(local_path, "wb") as f:
            f.write(content)
    return local_path

def run_harvest_and_download(batch_size: int = 100):
    assert_disk_space_or_exit(min_free_gb=10.0)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Obtener diseñadores pendientes o con archivos encontrados previamente
    cursor.execute("""
        SELECT ab.creator_username, COUNT(*) as moc_count
        FROM alternate_builds ab
        JOIN sets s ON ab.parent_set_id = s.set_id
        LEFT JOIN bricksafe_designer_scanned bds ON ab.creator_username = bds.designer_name
        WHERE s.theme = 'Creator 3-in-1' 
          AND ab.creator_username IS NOT NULL
          AND (bds.designer_name IS NULL OR (bds.files_found > 0 AND ab.io_url IS NULL))
        GROUP BY ab.creator_username
        ORDER BY moc_count DESC
        LIMIT ?;
    """, (batch_size,))
    pending_designers = cursor.fetchall()

    disk = get_disk_status()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rastreo inteligente para {len(pending_designers)} diseñadores.")
    print(f"Disco: {disk['free_gb']} GB libres (Margen mínimo de parada: 10.00 GB).")

    total_downloaded = 0
    total_matched = 0

    for d_name, count in pending_designers:
        assert_disk_space_or_exit(min_free_gb=10.0)
        files = scan_designer_bricksafe(d_name)

        if files:
            # Obtener todos los MOCs de este diseñador en BD
            cursor.execute("""
                SELECT moc_id, parent_set_id, name, source_url
                FROM alternate_builds
                WHERE creator_username = ?;
            """, (d_name,))
            mocs = cursor.fetchall()

            designer_matches = 0
            for moc_id, parent_set_id, moc_name, source_url in mocs:
                parent_num = parent_set_id.split("-")[0]
                moc_tokens = clean_tokens(moc_name)

                best_file = None
                best_score = 0

                for f_path in files:
                    file_tokens = clean_tokens(f_path)
                    # Debe coincidir al menos el set padre o un token distintivo
                    common = moc_tokens.intersection(file_tokens)
                    score = len(common)
                    if parent_num in file_tokens:
                        score += 3
                    
                    if score > best_score and (score >= 4 or (parent_num in file_tokens and len(common) >= 1)):
                        best_score = score
                        best_file = f_path

                if best_file:
                    full_url = f"https://bricksafe.com{best_file}"
                    cursor.execute("""
                        UPDATE alternate_builds
                        SET io_url = ?
                        WHERE moc_id = ?;
                    """, (full_url, moc_id))
                    designer_matches += 1
                    total_matched += 1

                    # Descargar de inmediato
                    try:
                        dl_path = download_file(full_url, moc_id, moc_name)
                        cursor.execute("""
                            UPDATE alternate_builds
                            SET io_file_path = ?,
                                download_status = 'io_downloaded'
                            WHERE moc_id = ?;
                        """, (dl_path, moc_id))
                        total_downloaded += 1
                        print(f"  [Descargado] {moc_id} - {moc_name[:30]} -> {os.path.basename(dl_path)}")
                    except Exception as e:
                        print(f"  [Error descarga] {moc_id}: {e}")

            print(f"  {d_name}: {len(files)} archivos 3D encontrados -> {designer_matches} modelos vinculados y descargados.")
        
        cursor.execute("""
            INSERT OR REPLACE INTO bricksafe_designer_scanned (designer_name, scanned_at, files_found)
            VALUES (?, CURRENT_TIMESTAMP, ?);
        """, (d_name, len(files)))
        conn.commit()
        time.sleep(0.2)

    conn.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fin del lote: {total_matched} vinculados, {total_downloaded} descargados.")
    print(f"Espacio libre en disco: {get_disk_status()['free_gb']} GB")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_harvest_and_download(batch_size=count)

#!/usr/bin/env python3
"""
Rastreador y descargador global de archivos 3D (.io, .ldr, .mpd) para TODOS los MOCs
de Rebrickable en la base de datos hasta alcanzar el límite de almacenamiento seguro.
Optimizado para Apple Silicon con concurrencia segura.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage_guard import assert_disk_space_or_exit, get_disk_status

DB_PATH = "data/catalog/models_catalog.db"
IO_DIR = "data/models_io"
MAX_DISK_FILL_GB = 10.0  # Límite estricto de margen de seguridad libre en disco (10 GB)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def clean_tokens(text: str) -> set:
    text = urllib.parse.unquote(text).lower()
    tokens = set(re.findall(r'[a-z0-9]+', text))
    tokens.difference_update({
        "io", "ldr", "mpd", "pdf", "the", "and", "alt", "alternate", 
        "build", "lego", "files", "pages", "copy", "instructions", "v1", "v2", "final"
    })
    return tokens

def scan_designer_bricksafe(designer_name: str) -> list:
    """Escanea recursivamente las páginas públicas de un diseñador en Bricksafe buscando .io, .ldr, .mpd"""
    encoded_name = urllib.parse.quote(designer_name.strip())
    url = f"https://bricksafe.com/pages/{encoded_name}"
    req = urllib.request.Request(url, headers={"User-Agent": "LegoGPT-GlobalCrawler/2.0"})

    try:
        with urllib.request.urlopen(req, timeout=12, context=ctx) as r:
            if r.status != 200:
                return []
            html = r.read().decode("utf-8", errors="ignore")
            
            # Subpáginas del diseñador
            subpages = re.findall(rf'href=[\"\'](/pages/{re.escape(encoded_name)}/[^\"\']+)[\"\']', html)
            
            found_files = set()
            root_files = re.findall(rf'href=[\"\'](/files/{re.escape(encoded_name)}/[^\"\']+\.(?:io|ldr|mpd))[\"\']', html, re.I)
            found_files.update(root_files)

            # Explorar hasta 30 subpáginas por diseñador
            for sp in subpages[:30]:
                sp_url = f"https://bricksafe.com{sp}"
                try:
                    sp_req = urllib.request.Request(sp_url, headers={"User-Agent": "LegoGPT-GlobalCrawler/2.0"})
                    with urllib.request.urlopen(sp_req, timeout=10, context=ctx) as sp_r:
                        sp_html = sp_r.read().decode("utf-8", errors="ignore")
                        f_matches = re.findall(rf'href=[\"\'](/files/{re.escape(encoded_name)}/[^\"\']+\.(?:io|ldr|mpd))[\"\']', sp_html, re.I)
                        found_files.update(f_matches)
                except Exception:
                    pass
                time.sleep(0.05)

            return list(found_files)
    except Exception:
        return []

def download_file(url: str, moc_id: str, name: str) -> str:
    os.makedirs(IO_DIR, exist_ok=True)
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".io"
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:40]
    local_path = os.path.join(IO_DIR, f"{moc_id}_{safe_name}{ext}")

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    req = urllib.request.Request(url, headers={"User-Agent": "LegoGPT-GlobalDownloader/2.0"})
    with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
        content = r.read()
        with open(local_path, "wb") as f:
            f.write(content)
    return local_path

def match_mocs_with_files(mocs: list, files: list) -> list:
    """Empareja MOCs con archivos 3D encontrados usando token matching semántico."""
    matched = []
    for moc_id, parent_set_id, moc_name in mocs:
        parent_num = parent_set_id.split("-")[0]
        moc_tokens = clean_tokens(moc_name)

        best_file = None
        best_score = 0

        for f_path in files:
            file_tokens = clean_tokens(f_path)
            common = moc_tokens.intersection(file_tokens)
            score = len(common)
            if parent_num in file_tokens:
                score += 3
            
            # Criterio de confianza
            if score > best_score and (score >= 4 or (parent_num in file_tokens and len(common) >= 1) or (parent_num in file_tokens and len(moc_tokens) <= 2)):
                best_score = score
                best_file = f_path

        if best_file:
            matched.append((moc_id, moc_name, best_file))
    return matched

def run_global_harvester():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando rastreador global de modelos 3D.")
    disk = get_disk_status()
    print(f"Espacio libre en disco: {disk['free_gb']} GB (Margen mínimo de seguridad: {MAX_DISK_FILL_GB} GB).")

    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    cursor = conn.cursor()

    # Obtener diseñadores pendientes de todos los temas ordenados por volumen de MOCs
    cursor.execute("""
        SELECT ab.creator_username, COUNT(*) as moc_count
        FROM alternate_builds ab
        LEFT JOIN bricksafe_designer_scanned bds ON ab.creator_username = bds.designer_name
        WHERE ab.creator_username IS NOT NULL
          AND bds.designer_name IS NULL
        GROUP BY ab.creator_username
        ORDER BY moc_count DESC;
    """)
    pending_designers = cursor.fetchall()
    print(f"Total diseñadores pendientes por escanear en todos los temas: {len(pending_designers)}")

    total_downloaded = 0
    total_scanned = 0

    for designer_name, moc_count in pending_designers:
        # Verificar espacio en disco antes de cada diseñador
        current_disk = get_disk_status()
        if current_disk['free_gb'] <= MAX_DISK_FILL_GB + 0.2:
            print(f"⚠️ Alerta: Espacio libre ({current_disk['free_gb']} GB) ha alcanzado el margen de seguridad de 10 GB. Deteniendo.")
            break

        files = scan_designer_bricksafe(designer_name)
        total_scanned += 1

        if files:
            # Traer los MOCs de este diseñador
            cursor.execute("""
                SELECT moc_id, parent_set_id, name
                FROM alternate_builds
                WHERE creator_username = ?;
            """, (designer_name,))
            designer_mocs = cursor.fetchall()

            matched = match_mocs_with_files(designer_mocs, files)

            for moc_id, moc_name, best_file in matched:
                full_url = f"https://bricksafe.com{best_file}"
                try:
                    dl_path = download_file(full_url, moc_id, moc_name)
                    cursor.execute("""
                        UPDATE alternate_builds
                        SET io_url = ?,
                            io_file_path = ?,
                            download_status = 'io_downloaded'
                        WHERE moc_id = ?;
                    """, (full_url, dl_path, moc_id))
                    total_downloaded += 1
                    print(f"  [Descargado] {moc_id} ({designer_name}): {moc_name[:30]} -> {os.path.basename(dl_path)}")
                except Exception as e:
                    print(f"  [Error descarga] {moc_id}: {e}")

            print(f"[{total_scanned}/{len(pending_designers)}] {designer_name}: {len(files)} archivos en Bricksafe -> {len(matched)} emparejados.")
        else:
            if total_scanned % 50 == 0:
                print(f"[{total_scanned}/{len(pending_designers)}] Escaneando... (último: {designer_name})")

        cursor.execute("""
            INSERT OR REPLACE INTO bricksafe_designer_scanned (designer_name, scanned_at, files_found)
            VALUES (?, CURRENT_TIMESTAMP, ?);
        """, (designer_name, len(files)))
        conn.commit()
        time.sleep(0.15)

    conn.close()
    disk_end = get_disk_status()
    print(f"\n=======================================================")
    print(f"Rastreo finalizado. Diseñadores escaneados: {total_scanned}")
    print(f"Nuevos modelos descargados: {total_downloaded}")
    print(f"Espacio libre en disco: {disk_end['free_gb']} GB")
    print(f"=======================================================")

if __name__ == "__main__":
    run_global_harvester()

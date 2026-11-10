#!/usr/bin/env python3
"""
Downloads official LEGO instruction manuals (PDFs) from the LEGO CDN based on:
1. Sets with < 100 parts
2. All BrickHeadz sets (regardless of parts)
3. All Animal/Creature sets (regardless of parts)

Features:
- Dynamic multi-core parallelism for Apple Silicon M4.
- Storage guard protection (aborts/pauses if free space < 15 GB).
- Fully resumable with database state tracking (`download_status` in `set_manuals`).
- Standard structured naming: data/manuals/official_sets/{set_id}_{booklet_idx}of{booklet_total}_{file_id}.pdf
"""

import os
import sys
import time
import sqlite3
import urllib.request
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = "data/catalog/models_catalog.db"
DEST_DIR = "data/manuals/official_sets"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
MIN_FREE_GB = 15.0

ANIMAL_KEYWORDS = [
    'animal', 'creature', 'pet', 'beast', 'dragon', 'dinosaur', 'dino',
    'cat', 'dog', 'puppy', 'kitten', 'bear', 'panda', 'lion', 'tiger',
    'elephant', 'monkey', 'bird', 'owl', 'parrot', 'eagle', 'duck',
    'fish', 'shark', 'whale', 'dolphin', 'turtle', 'snake', 'frog',
    'spider', 'horse', 'pony', 'unicorn', 'rabbit', 'bunny', 'hamster',
    'wolf', 'fox', 'deer', 'reindeer', 'giraffe', 'zebra', 'hippo',
    'otter', 'hedgehog', 'bee', 'chick', 'sheep', 'poodle', 'bulldog'
]


def check_disk_space():
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        print(f"\n[ALERTA SEGURIDAD] Espacio libre en disco crítico ({free_gb:.2f} GB < {MIN_FREE_GB} GB). Abortando para proteger el sistema.")
        sys.exit(1)
    return free_gb


def ensure_db_columns(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(set_manuals)")
    cols = [col[1] for col in cur.fetchall()]
    if "local_file_path" not in cols:
        conn.execute("ALTER TABLE set_manuals ADD COLUMN local_file_path TEXT")
    if "download_status" not in cols:
        conn.execute("ALTER TABLE set_manuals ADD COLUMN download_status TEXT DEFAULT 'pending'")
    conn.commit()


def get_target_manuals(conn):
    conditions = []
    for kw in ANIMAL_KEYWORDS:
        conditions.append(f"lower(s.name) LIKE '%{kw}%'")
        conditions.append(f"lower(s.theme) LIKE '%{kw}%'")
        conditions.append(f"lower(s.tags) LIKE '%{kw}%'")

    animal_clause = " OR ".join(conditions)

    sql = f"""
    SELECT 
        m.set_id, 
        m.file_id, 
        m.pdf_url, 
        m.booklet_idx, 
        m.booklet_total, 
        m.size_bytes, 
        s.name, 
        s.parts_count
    FROM set_manuals m
    JOIN sets s ON m.set_id = s.set_id
    WHERE (
        s.parts_count < 100
        OR lower(s.theme) LIKE '%brickheadz%'
        OR ({animal_clause})
    )
    AND (m.download_status IS NULL OR m.download_status != 'downloaded')
    ORDER BY m.size_bytes ASC
    """
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()


def download_manual(item):
    set_id, file_id, pdf_url, b_idx, b_tot, expected_size, set_name, parts = item
    
    idx_str = f"_{b_idx}of{b_tot}" if b_idx and b_tot else ""
    filename = f"{set_id}{idx_str}_{file_id}.pdf"
    dest_path = os.path.join(DEST_DIR, filename)

    # Si ya existe y el tamaño es correcto
    if os.path.exists(dest_path):
        actual_size = os.path.getsize(dest_path)
        if expected_size and actual_size >= expected_size:
            return set_id, file_id, dest_path, actual_size, "ALREADY_EXISTS", None

    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=45) as response:
            with open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file, length=64 * 1024)
        
        final_size = os.path.getsize(dest_path)
        return set_id, file_id, dest_path, final_size, "SUCCESS", None
    except Exception as e:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        return set_id, file_id, dest_path, 0, "ERROR", str(e)


def main():
    check_disk_space()
    os.makedirs(DEST_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    ensure_db_columns(conn)

    tasks = get_target_manuals(conn)
    total_tasks = len(tasks)
    print(f"============================================================")
    print(f" Descarga de Manuales Oficiales LEGO (PDFs)")
    print(f" Criterio: < 100 piezas | BrickHeadz | Animales/Criaturas")
    print(f" Total de PDFs pendientes: {total_tasks}")
    print(f" Disco disponible: {check_disk_space():.2f} GB (Guarda: {MIN_FREE_GB} GB)")
    print(f"============================================================")

    if total_tasks == 0:
        print("¡No hay manuales pendientes de descarga para estos criterios!")
        conn.close()
        return

    # M4 dynamic workers (I/O network bound)
    workers = min(12, max(2, (os.cpu_count() or 12) - 2))
    print(f"Iniciando descargas concurrentes con {workers} workers...\n")

    completed = 0
    errors = 0
    downloaded_bytes = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(download_manual, item): item for item in tasks}
        
        for future in as_completed(future_to_item):
            set_id, file_id, path, size, status, err = future.result()
            completed += 1

            if status in ("SUCCESS", "ALREADY_EXISTS"):
                downloaded_bytes += size
                conn.execute(
                    "UPDATE set_manuals SET local_file_path = ?, download_status = 'downloaded' WHERE set_id = ? AND file_id = ?",
                    (path, set_id, file_id)
                )
            else:
                errors += 1
                conn.execute(
                    "UPDATE set_manuals SET download_status = 'error' WHERE set_id = ? AND file_id = ?",
                    (set_id, file_id)
                )

            # Commit en batches
            if completed % 25 == 0 or completed == total_tasks:
                conn.commit()
                check_disk_space()
                mb_done = downloaded_bytes / (1024 * 1024)
                pct = (completed / total_tasks) * 100
                print(f"[{completed}/{total_tasks} - {pct:.1f}%] Descargados: {mb_done:.1f} MB | Errores: {errors}")

    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n============================================================")
    print(f" PROCESO COMPLETADO en {elapsed/60:.2f} minutos")
    print(f" Total procesados: {completed} | Errores: {errors}")
    print(f" Total descargado: {downloaded_bytes / (1024 * 1024 * 1024):.2f} GB")
    print(f" Directorio destino: {DEST_DIR}")
    print(f"============================================================")


if __name__ == "__main__":
    main()

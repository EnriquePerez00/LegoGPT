#!/usr/bin/env python3
"""
Downloads all identified PDF instruction manuals from Bricksafe for cataloged MOCs.
Strictly respects the 10.00 GB storage guard safety threshold.
Optimized for Apple Silicon M4 with concurrent worker pooling.
"""

import os
import sys
import sqlite3
import requests
import certifi
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.storage_guard import assert_disk_space_or_exit, get_disk_status


def download_single_pdf(item: tuple) -> dict:
    moc_id, name, pdf_url, dest_path = item
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return {"moc_id": moc_id, "status": "ALREADY_EXISTS", "size_kb": os.path.getsize(dest_path)/1024}

    assert_disk_space_or_exit(min_free_gb=10.0)

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        r = requests.get(pdf_url, headers=headers, timeout=25, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        size_kb = os.path.getsize(dest_path) / 1024.0
        return {"moc_id": moc_id, "status": "SUCCESS", "size_kb": size_kb}
    except Exception as e:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        return {"moc_id": moc_id, "status": "ERROR", "error": str(e)}


def main():
    assert_disk_space_or_exit(min_free_gb=10.0)
    print("Initial disk status:", get_disk_status())

    dest_dir = "data/manuals/mocs"
    os.makedirs(dest_dir, exist_ok=True)

    conn = sqlite3.connect("data/catalog/models_catalog.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT moc_id, name, pdf_url, io_file_path 
        FROM alternate_builds 
        WHERE pdf_url IS NOT NULL
        ORDER BY num_parts ASC
    """)
    rows = cur.fetchall()

    tasks = []
    for moc_id, name, pdf_url, io_path in rows:
        clean_url_name = os.path.basename(pdf_url.split("?")[0])
        dest_path = os.path.join(dest_dir, f"{moc_id}_{clean_url_name}")
        tasks.append((moc_id, name, pdf_url, dest_path))

    print(f"Total PDFs in catalog to process: {len(tasks)}")

    # Reserve 2-4 cores for OS responsiveness on 12-core M4
    max_workers = max(1, (os.cpu_count() or 12) - 4)
    print(f"Starting concurrent download pool with {max_workers} workers...")

    completed = 0
    errors = 0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(download_single_pdf, item): item for item in tasks}
        for future in as_completed(future_map):
            res = future.result()
            completed += 1
            if res["status"] in ("SUCCESS", "ALREADY_EXISTS"):
                total_bytes += res.get("size_kb", 0)
                if completed % 25 == 0 or completed == len(tasks):
                    print(f"  Progress: {completed}/{len(tasks)} (Current disk: {get_disk_status()['free_gb']} GB free)")
            else:
                errors += 1
                print(f"  Failed {res['moc_id']}: {res.get('error')}")

    print(f"\nFinished! Downloaded/Verified: {completed} files ({total_bytes/1024:.2f} MB), Errors: {errors}")
    print("Final disk status:", get_disk_status())


if __name__ == "__main__":
    main()

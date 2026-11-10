#!/usr/bin/env python3
"""
Continuous mass sync of alternate builds metadata from Rebrickable API for all sets in the catalog.
Optimized for Apple Silicon M4 with dynamic memory checks, resilient rate-limiting backoff,
and strict 10.0 GB disk safety guardrails.
"""

import os
import sys
import json
import time
import sqlite3
import ssl
import urllib.request
import urllib.error
try:
    import psutil
except ImportError:
    psutil = None
from datetime import datetime

# Contexto SSL seguro y compatible con certificados de macOS/Python
ssl_context = ssl.create_default_context()
try:
    import certifi
    ssl_context.load_verify_locations(certifi.where())
except Exception:
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage_guard import assert_disk_space_or_exit, get_disk_status

DB_PATH = "data/catalog/models_catalog.db"

def get_rebrickable_api_key():
    key = os.getenv("REBRICKABLE_API_KEY", "")
    if not key and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("REBRICKABLE_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    break
    return key

def fetch_alternates_page(set_id: str, api_key: str, page: int = 1, page_size: int = 100):
    url = f"https://rebrickable.com/api/v3/lego/sets/{set_id}/alternates/?page={page}&page_size={page_size}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"key {api_key}",
        "Accept": "application/json",
        "User-Agent": "LegoGPT-Catalog/1.0"
    })
    
    backoff = 2
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=25, context=ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"results": [], "next": None, "count": 0}
            elif e.code == 429:
                print(f"  [Rate Limit 429] Reintento {attempt+1}/4 en {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"  [HTTP {e.code}] Set {set_id}: {e}")
                return {"results": [], "next": None, "count": 0}
        except Exception as e:
            print(f"  [Error red] Set {set_id}: {e}. Reintentando en {backoff}s...")
            time.sleep(backoff)
            backoff *= 2
            
    return {"results": [], "next": None, "count": 0}

def fetch_all_alternates(set_id: str, api_key: str):
    alternates = []
    page = 1
    while True:
        res = fetch_alternates_page(set_id, api_key, page=page)
        results = res.get("results", [])
        alternates.extend(results)
        
        if not res.get("next"):
            break
        page += 1
        time.sleep(0.35)
        
    return alternates

def sync_all_catalog_alternates(batch_commit_size: int = 25):
    assert_disk_space_or_exit(min_free_gb=10.0)
    api_key = get_rebrickable_api_key()
    if not api_key:
        print("[Error Fatal] REBRICKABLE_API_KEY no encontrada.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rebrickable_sync_progress (
            set_id TEXT PRIMARY KEY,
            checked_at TIMESTAMP,
            alts_found INTEGER
        );
    """)
    conn.commit()

    # Query sets prioritized by theme interest and part count
    cursor.execute("""
        SELECT s.set_id, s.name, s.theme, s.parts_count 
        FROM sets s
        LEFT JOIN rebrickable_sync_progress p ON s.set_id = p.set_id
        WHERE p.set_id IS NULL
        ORDER BY 
            CASE 
                WHEN s.theme LIKE "%Creator 3-in-1%" THEN 1
                WHEN s.theme LIKE "%Speed Champions%" THEN 2
                WHEN s.theme LIKE "%Technic%" THEN 3
                WHEN s.theme LIKE "%Creator%" THEN 4
                WHEN s.theme LIKE "%Star Wars%" THEN 5
                WHEN s.theme LIKE "%Ninjago%" THEN 6
                WHEN s.theme LIKE "%City%" THEN 7
                ELSE 8
            END,
            s.parts_count DESC;
    """)
    pending_sets = cursor.fetchall()
    total_to_process = len(pending_sets)

    disk = get_disk_status()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando indexación completa de {total_to_process} sets.")
    print(f"Espacio libre en disco: {disk['free_gb']} GB (Límite de seguridad: 10.00 GB).")

    processed = 0
    total_new_mocs = 0
    start_time = time.time()

    for set_id, name, theme, parts_count in pending_sets:
        # Check disk space limit every set
        assert_disk_space_or_exit(min_free_gb=10.0)

        # Dynamic RAM check (Apple Silicon M4)
        if psutil:
            mem = psutil.virtual_memory()
            if mem.percent > 85.0:
                print(f"[Aviso RAM {mem.percent}%] Pausando 2s para liberar memoria...")
                time.sleep(2.0)

        alts = fetch_all_alternates(set_id, api_key)
        alts_count = len(alts)

        for alt in alts:
            moc_id = alt.get("set_num")
            alt_name = alt.get("name")
            num_parts = alt.get("num_parts", 0)
            moc_url = alt.get("moc_url")
            designer_name = alt.get("designer_name")

            cursor.execute("""
                INSERT INTO alternate_builds (
                    moc_id, parent_set_id, name, source, creator_username,
                    num_parts, source_url, download_status, is_free
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 1)
                ON CONFLICT(moc_id) DO UPDATE SET
                    name = excluded.name,
                    num_parts = excluded.num_parts,
                    creator_username = excluded.creator_username,
                    source_url = excluded.source_url
            """, (
                moc_id,
                set_id,
                alt_name,
                "rebrickable_alt",
                designer_name,
                num_parts,
                moc_url
            ))
            total_new_mocs += 1

        cursor.execute("""
            INSERT OR REPLACE INTO rebrickable_sync_progress (set_id, checked_at, alts_found)
            VALUES (?, CURRENT_TIMESTAMP, ?);
        """, (set_id, alts_count))

        processed += 1

        if processed % batch_commit_size == 0:
            conn.commit()
            elapsed = time.time() - start_time
            rate = processed / elapsed
            disk_now = get_disk_status()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progreso: {processed}/{total_to_process} ({processed/total_to_process*100:.1f}%) | "
                  f"Total MOCs: {total_new_mocs} | Ritmo: {rate:.1f} sets/s | Disco: {disk_now['free_gb']} GB")
        else:
            if alts_count > 0:
                print(f"  [{processed}/{total_to_process}] {set_id} ({theme[:15]}) {name[:25]} -> {alts_count} alts")

        time.sleep(0.3)

    conn.commit()
    conn.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ¡Indexación completa finalizada!")
    print(f"Total sets analizados: {processed} | Total alternate builds descubiertos: {total_new_mocs}")
    print(f"Espacio libre final en disco: {get_disk_status()['free_gb']} GB")

if __name__ == "__main__":
    sync_all_catalog_alternates()

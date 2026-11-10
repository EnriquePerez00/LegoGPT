#!/usr/bin/env python3
"""
Mass sync alternate builds metadata for high-yield themes from Rebrickable API.
Prioritizes Creator 3-in-1, Speed Champions, Technic, and Star Wars.
Monitors free disk space dynamically with hard safety stop at 10.0 GB.
"""

import os
import sys
import json
import time
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime

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
    
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"results": [], "next": None, "count": 0}
        elif e.code == 429:
            print(f"[Rate Limit 429] Pausa de 5s...")
            time.sleep(5)
            return None
        else:
            print(f"[HTTP {e.code}] Set {set_id}: {e}")
            return {"results": [], "next": None, "count": 0}
    except Exception as e:
        print(f"[Red/Timeout] Set {set_id}: {e}")
        return None

def fetch_all_alternates(set_id: str, api_key: str):
    alternates = []
    page = 1
    while True:
        res = fetch_alternates_page(set_id, api_key, page=page)
        if res is None:
            time.sleep(2)
            res = fetch_alternates_page(set_id, api_key, page=page)
            if res is None:
                break
        
        results = res.get("results", [])
        alternates.extend(results)
        
        if not res.get("next"):
            break
        page += 1
        time.sleep(0.4)
        
    return alternates

def sync_prioritized_alternates(limit_sets: int = 100):
    assert_disk_space_or_exit(min_free_gb=10.0)
    api_key = get_rebrickable_api_key()
    if not api_key:
        print("[Error] REBRICKABLE_API_KEY no encontrada.")
        return

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

    cursor.execute("""
        SELECT s.set_id, s.name, s.theme, s.parts_count 
        FROM sets s
        LEFT JOIN rebrickable_sync_progress p ON s.set_id = p.set_id
        WHERE p.set_id IS NULL AND (
            s.theme LIKE "%Creator%" OR 
            s.theme LIKE "%Technic%" OR 
            s.theme LIKE "%Speed Champions%" OR
            s.theme LIKE "%Star Wars%"
        )
        ORDER BY 
            CASE 
                WHEN s.theme LIKE "%Creator 3-in-1%" THEN 1
                WHEN s.theme LIKE "%Speed Champions%" THEN 2
                WHEN s.theme LIKE "%Technic%" THEN 3
                WHEN s.theme LIKE "%Creator%" THEN 4
                ELSE 5
            END,
            s.parts_count DESC
        LIMIT ?;
    """, (limit_sets,))
    sets_to_check = cursor.fetchall()

    disk = get_disk_status()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando sincronización de {len(sets_to_check)} sets priorizados.")
    print(f"Espacio libre actual: {disk['free_gb']} GB (Margen de parada: 10.00 GB).")

    total_new_mocs = 0
    checked_count = 0

    for set_id, name, theme, parts_count in sets_to_check:
        assert_disk_space_or_exit(min_free_gb=10.0)

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

        conn.commit()
        checked_count += 1
        status_tag = f"-> {alts_count} alts" if alts_count > 0 else "-> 0 alts"
        print(f"  [{checked_count}/{len(sets_to_check)}] {set_id} ({theme}) {name[:25]} {status_tag}")
        time.sleep(0.3)

    conn.close()
    print(f"\n[Sincronización Completada] Sets procesados: {checked_count} | Nuevos Alternate Builds guardados: {total_new_mocs}")
    print(f"Espacio libre final: {get_disk_status()['free_gb']} GB")

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    sync_prioritized_alternates(limit_sets=limit)

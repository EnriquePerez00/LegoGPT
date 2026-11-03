import os
import sys
import re
import time
import urllib.request
import urllib.error
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.storage_guard import assert_disk_space_or_exit, get_disk_status, MIN_FREE_GB

OUTPUT_DIR = "data/omr_raw"
DB_PATH = "data/catalog/models_catalog.db"
BASE_URL = "https://library.ldraw.org"

def download_single_omr_set(set_id: int):
    # Check disk space before every operation
    if not get_disk_status()["is_safe"]:
        return None, "DISK_LIMIT_REACHED"
        
    set_url = f"{BASE_URL}/omr/sets/{set_id}"
    req = urllib.request.Request(set_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8")
    except Exception:
        return None, "NOT_FOUND"
        
    # Extract MPD links
    mpd_matches = re.findall(r'href=[\"\'](https://library\.ldraw\.org/media/omr_models/\d+/([^\"\']+\.mpd)[^\"\']*)[\"\']', html)
    if not mpd_matches:
        return None, "NO_MPD"
        
    results = []
    for mpd_url, raw_filename in mpd_matches:
        if not get_disk_status()["is_safe"]:
            return results, "DISK_LIMIT_REACHED"
            
        clean_filename = re.sub(r'[^\w\-_\.]', '_', raw_filename)
        save_path = os.path.join(OUTPUT_DIR, clean_filename)
        
        if os.path.exists(save_path):
            continue
            
        try:
            file_req = urllib.request.Request(mpd_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(file_req, timeout=20) as f_resp:
                content = f_resp.read()
                
            with open(save_path, "wb") as f:
                f.write(content)
                
            text = content.decode("utf-8", errors="ignore")
            steps = len(re.findall(r'0\s+STEP', text, re.IGNORECASE))
            parts = len([l for l in text.splitlines() if l.strip().startswith("1 ")])
            
            results.append({
                "filename": clean_filename,
                "path": save_path,
                "url": mpd_url,
                "steps": steps,
                "parts": parts,
                "size_kb": len(content) / 1024.0
            })
        except Exception as e:
            pass
            
    return results, "OK"

def batch_harvest_omr(start_id: int = 1, end_id: int = 100, max_workers: int = 5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    assert_disk_space_or_exit(MIN_FREE_GB)
    
    initial_disk = get_disk_status()
    print("=" * 60)
    print(f"[OMR Harvester] Iniciando descarga de IDs {start_id} a {end_id}...")
    print(f"Espacio libre en disco: {initial_disk['free_gb']} GB (Límite de seguridad: {MIN_FREE_GB} GB).")
    print(f"Hilos concurrentes: {max_workers}")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    total_downloaded = 0
    total_steps = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_omr_set, sid): sid for sid in range(start_id, end_id + 1)}
        
        for future in as_completed(futures):
            sid = futures[future]
            try:
                res, status = future.result()
                if status == "DISK_LIMIT_REACHED":
                    print("\n[ALERTA DE SEGURIDAD] Límite de 10 GB de disco alcanzado. Deteniendo descargas.")
                    break
                    
                if res:
                    for item in res:
                        total_downloaded += 1
                        total_steps += item["steps"]
                        print(f"  [Set ID {sid:03d}] ✅ {item['filename']} ({item['size_kb']:.1f} KB | {item['parts']} piezas | {item['steps']} pasos)")
                        
                        set_match = re.search(r'(\d+-\d+|\d+)', item["filename"])
                        set_id_hint = set_match.group(1) if set_match else f"OMR_{sid}"
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO alternate_builds (
                                moc_id, parent_set_id, name, source, creator_username,
                                num_parts, num_steps, has_discrete_steps, io_file_path,
                                source_url, download_status
                            ) VALUES (?, ?, ?, 'omr_ldraw', 'OMR Community', ?, ?, ?, ?, ?, 'downloaded')
                        """, (
                            f"OMR_{item['filename']}",
                            set_id_hint,
                            item["filename"].replace(".mpd", "").replace("_", " "),
                            item["parts"],
                            item["steps"],
                            1 if item["steps"] > 1 else 0,
                            item["path"],
                            item["url"]
                        ))
                    conn.commit()
            except Exception as e:
                pass
                
    conn.close()
    final_disk = get_disk_status()
    print("\n" + "=" * 60)
    print(f"Resumen de lote completado:")
    print(f"  - Nuevos archivos .mpd descargados: {total_downloaded}")
    print(f"  - Pasos de construcción 0 STEP detectados: {total_steps}")
    print(f"  - Espacio libre restante en disco: {final_disk['free_gb']} GB")
    print("=" * 60)

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    batch_harvest_omr(start_id=start, end_id=end, max_workers=5)

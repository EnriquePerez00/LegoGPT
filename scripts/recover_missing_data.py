#!/usr/bin/env python3
"""Fase 1: Recuperar y regenerar ficheros .io, .pdf y referenciados en models_catalog.db"""
import os, json, sqlite3, subprocess, sys, time, hashlib
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CATALOG_DB = DATA_DIR / "catalog" / "models_catalog.db"
DIRS = {
    "normalized_io": DATA_DIR / "normalized_io",
    "models_io": DATA_DIR / "models_io",
    "manuals_mocs": DATA_DIR / "manuals" / "mocs",
    "manuals_official": DATA_DIR / "manuals" / "official_sets",
    "omr_raw": DATA_DIR / "omr_raw",
    "ldr_extracted": DATA_DIR / "ldr_extracted",
}
LOG_FILE = REPO_ROOT / "outputs" / "recover_missing_data.log"
STATS_FILE = REPO_ROOT / "outputs" / "recover_missing_data_stats.json"

class DataRecoveryLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.stats = {
            "start_time": time.time(),
            "dirs_created": [],
            "files_checked": 0,
            "files_missing": 0,
            "files_downloaded": 0,
            "errors": [],
        }
    def log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
    def save_stats(self):
        self.stats["end_time"] = time.time()
        self.stats["duration_sec"] = self.stats["end_time"] - self.stats["start_time"]
        with open(STATS_FILE, "w") as f:
            json.dump(self.stats, f, indent=2)

def ensure_dirs(logger):
    logger.log("=== Creando directorios ===")
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.log(f"  ✓ {path}")
        logger.stats["dirs_created"].append(str(path))

def query_missing_files(logger):
    logger.log("=== Consultando BD para identificar ficheros faltantes ===")
    if not CATALOG_DB.exists():
        logger.log(f"  ✗ BD no encontrada: {CATALOG_DB}")
        return {"io": [], "pdf": []}
    
    conn = sqlite3.connect(CATALOG_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    missing = {"io": [], "pdf": []}
    
    try:
        cursor.execute("SELECT moc_id, io_file_path, io_url FROM alternate_builds WHERE io_file_path IS NOT NULL")
        for row in cursor.fetchall():
            io_path = row["io_file_path"]
            full_path = DATA_DIR / io_path if io_path else None
            if full_path and not full_path.exists():
                missing["io"].append({"moc_id": row["moc_id"], "path": io_path, "url": row["io_url"], "full_path": str(full_path)})
    except Exception as e:
        logger.log(f"  ✗ Error alternate_builds: {e}")
        logger.stats["errors"].append(str(e))
    
    try:
        cursor.execute("SELECT pair_id, normalized_io_path, pdf_manual_path, model_download_url FROM ground_truth_pairs")
        for row in cursor.fetchall():
            io_path = row["normalized_io_path"]
            if io_path:
                full_path = DATA_DIR / io_path
                if not full_path.exists():
                    missing["io"].append({"pair_id": row["pair_id"], "path": io_path, "url": row["model_download_url"], "full_path": str(full_path), "type": "normalized"})
            pdf_path = row["pdf_manual_path"]
            if pdf_path:
                full_path = DATA_DIR / pdf_path
                if not full_path.exists():
                    missing["pdf"].append({"pair_id": row["pair_id"], "path": pdf_path, "full_path": str(full_path)})
    except Exception as e:
        logger.log(f"  ✗ Error ground_truth_pairs: {e}")
        logger.stats["errors"].append(str(e))
    
    conn.close()
    logger.log(f"  Missing .io files: {len(missing['io'])}")
    logger.log(f"  Missing .pdf files: {len(missing['pdf'])}")
    logger.stats["files_missing"] = len(missing["io"]) + len(missing["pdf"])
    return missing

def download_file(url, dest_path, timeout=30):
    if not url or not url.startswith("http"):
        return False
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(["curl", "-L", "-o", str(dest_path), "-m", str(timeout), url], capture_output=True, timeout=timeout + 5)
        return result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0
    except:
        return False

def download_missing_files(missing, logger, max_workers=4):
    logger.log("=== Descargando ficheros faltantes ===")
    tasks = []
    for item in missing["io"]:
        url = item.get("url")
        path = item["full_path"]
        if url and not Path(path).exists():
            id_ = item.get("moc_id") or item.get("pair_id")
            tasks.append(("io", url, path, id_))
    logger.log(f"  Total tasks: {len(tasks)}")
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, path): (ftype, id_, path) for ftype, url, path, id_ in tasks}
        for future in as_completed(futures):
            ftype, id_, path = futures[future]
            try:
                if future.result():
                    downloaded += 1
                    logger.log(f"  ✓ {ftype}: {id_}")
            except Exception as e:
                logger.log(f"  ✗ {ftype}: {id_} failed")
    logger.stats["files_downloaded"] = downloaded
    logger.log(f"  Downloaded: {downloaded}")

def summary_report(logger):
    logger.log("=== Reporte Final ===")
    logger.log(f"  Directorios: {len(logger.stats['dirs_created'])}")
    logger.log(f"  Ficheros faltantes: {logger.stats['files_missing']}")
    logger.log(f"  Descargados: {logger.stats['files_downloaded']}")
    logger.log(f"  Errores: {len(logger.stats['errors'])}")
    logger.log(f"  Duración: se completará en save_stats")

def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = DataRecoveryLogger(LOG_FILE)
    logger.log("=== Fase 1: Recuperación de datos ===")
    logger.log(f"Repo: {REPO_ROOT}")
    ensure_dirs(logger)
    missing = query_missing_files(logger)
    download_missing_files(missing, logger)
    summary_report(logger)
    logger.save_stats()
    logger.log("=== Completada ===")

if __name__ == "__main__":
    main()

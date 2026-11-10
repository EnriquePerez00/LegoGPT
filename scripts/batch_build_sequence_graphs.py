import os
import sys
import glob
import sqlite3
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.storage_guard import assert_disk_space_or_exit, get_disk_status, MIN_FREE_GB
from scripts.build_sequence_graph import process_model_to_sequence_graph

DB_PATH = "data/catalog/models_catalog.db"
OUTPUT_DIR = "data/sequences"

def check_memory_headroom(max_ram_percent: float = 80.0) -> bool:
    """Monitors Apple Silicon Unified Memory (48 GB) to maintain a 75-80% safety buffer."""
    vm = psutil.virtual_memory()
    return vm.percent < max_ram_percent

def process_worker(file_path: str):
    # Check disk space limit and RAM safety headroom
    if not get_disk_status()["is_safe"]:
        return None, "DISK_LIMIT_REACHED"
    if not check_memory_headroom(85.0):
        return {"model": os.path.basename(file_path), "error": "Memory threshold (>85%) exceeded", "success": False}, "MEMORY_PRESSURE"
    try:
        res = process_model_to_sequence_graph(file_path, output_pt_dir=OUTPUT_DIR)
        return res, "OK"
    except Exception as e:
        return {"model": os.path.basename(file_path), "error": str(e), "success": False}, "ERROR"

def run_batch_graph_extraction(max_workers: int = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    assert_disk_space_or_exit(MIN_FREE_GB)
    
    if max_workers is None:
        # User hardware rules: Apple Silicon M4 with 12 cores, reserve 2-4 cores
        max_workers = max(1, (os.cpu_count() or 12) - 2)
        
    files = sorted(
        glob.glob("data/omr_raw/*.mpd")
        + glob.glob("data/mocs/*.mpd")
        + glob.glob("data/mocs/*.io")
        + glob.glob("data/*.ldr")
        + glob.glob("data/*.mbx")
    )
    # Filter only relevant sequence files
    valid_files = [f for f in files if not f.endswith("_sequence_vlm.ldr")]
    
    print("=" * 60)
    print(f"[Graph Generator] Iniciando procesamiento paralelo de {len(valid_files)} modelos 3D...")
    print(f"Workers concurrentes (M4): {max_workers}")
    print(f"Espacio libre en disco: {get_disk_status()['free_gb']} GB")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    success_count = 0
    fail_count = 0
    total_edges = 0
    total_parts = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_worker, f): f for f in valid_files}
        
        for future in as_completed(futures):
            fpath = futures[future]
            fname = os.path.basename(fpath)
            
            res, status = future.result()
            if status == "DISK_LIMIT_REACHED":
                print(f"\n[ALERTA] Límite de 10 GB alcanzado. Abortando procesamiento de grafos.")
                break
                
            if res and res.get("success"):
                success_count += 1
                total_edges += res.get("edges_count", 0)
                total_parts += res.get("parts_count", 0)
                pt_path = res.get("graph_path")
                
                # Update DB
                cursor.execute("""
                    UPDATE alternate_builds
                    SET graph_tensor_path = ?, has_discrete_steps = ?
                    WHERE moc_id = ? OR moc_id = ?
                """, (pt_path, 1 if res.get("has_discrete_steps") else 0, f"OMR_{fname}", fname))
                conn.commit()
                
                print(f"[{success_count + fail_count}/{len(valid_files)}] ✅ {res['model']} -> {res['parts_count']} partes, {res['steps_count']} pasos, {res['edges_count']} aristas")
            else:
                fail_count += 1
                err = res.get("error") if res else "Unknown error"
                print(f"[{success_count + fail_count}/{len(valid_files)}] ⚠️ {fname}: {err}")
                
    conn.close()
    elapsed = time.time() - start_time
    final_disk = get_disk_status()
    
    print("\n" + "=" * 60)
    print(f"Procesamiento completado en {elapsed:.2f}s:")
    print(f"  - Grafos generados exitosamente: {success_count}")
    print(f"  - Conexiones físicas (aristas) totales: {total_edges}")
    print(f"  - Modelos fallidos/omitidos: {fail_count}")
    print(f"  - Espacio libre restante en disco: {final_disk['free_gb']} GB")
    print("=" * 60)

if __name__ == "__main__":
    run_batch_graph_extraction()

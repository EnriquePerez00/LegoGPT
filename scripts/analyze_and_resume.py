import os
import sys
import glob
import sqlite3
import json
import zipfile

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mpd_parser import flatten_mpd

DB_PATH = "data/catalog/models_catalog.db"
PROGRESS_PATH = "data/ingestion_progress.json"

def count_parts_in_io(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as archive:
            ldr_files = [n for n in archive.namelist() if n.lower().endswith('.ldr')]
            if not ldr_files:
                return 0
            main_ldr = "model.ldr"
            for m_name in ["model.ldr", "modelv2.ldr", "model2.ldr"]:
                if m_name in ldr_files:
                    main_ldr = m_name
                    break
            if main_ldr not in ldr_files:
                main_ldr = ldr_files[0]
                
            try:
                content = archive.read(main_ldr, pwd=b"soho0909").decode("utf-8", errors="ignore")
            except Exception:
                content = archive.read(main_ldr).decode("utf-8", errors="ignore")
            
            # Count parts (lines starting with '1 ')
            parts_count = 0
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("1 "):
                    parts_count += 1
            return parts_count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def main():
    print("=== Analyzing Database and .io Files ===")
    
    # 1. Ingested statistics
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ingested_rows = c.execute("SELECT set_id, parts_count FROM sets WHERE source = 'BrickLink'").fetchall()
    conn.close()
    
    # 2. Progress times
    times = []
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                progress = json.load(f)
                for item in progress.get("items", []):
                    if "elapsed_seconds" in item and item["status"] == "success":
                        times.append(item["elapsed_seconds"])
        except Exception:
            pass
            
    avg_time = sum(times) / len(times) if times else 0.0
    print(f"Average time per ingestion: {avg_time:.2f} seconds (based on {len(times)} runs)")
    
    # 3. Analyze parts counts of ingested models
    ingested_less_100 = sum(1 for r in ingested_rows if r[1] < 100)
    ingested_less_200 = sum(1 for r in ingested_rows if r[1] < 200)
    print(f"Ingested sets: {len(ingested_rows)}")
    print(f"  Ingested < 100 parts: {ingested_less_100}")
    print(f"  Ingested < 200 parts: {ingested_less_200}")
    
    # 4. Scan and count parts for missing .io files
    io_files = glob.glob("data/bricklink_raw/*.io")
    db_sets = set(r[0] for r in ingested_rows)
    missing_files = [f for f in io_files if os.path.splitext(os.path.basename(f))[0] not in db_sets]
    
    print(f"Missing files to process: {len(missing_files)}")
    
    missing_less_100 = 0
    missing_less_200 = 0
    
    for idx, f in enumerate(missing_files):
        count = count_parts_in_io(f)
        if count < 100:
            missing_less_100 += 1
        if count < 200:
            missing_less_200 += 1
            
    print(f"Missing < 100 parts: {missing_less_100}")
    print(f"Missing < 200 parts: {missing_less_200}")
    
    # Totals
    total_less_100 = ingested_less_100 + missing_less_100
    total_less_200 = ingested_less_200 + missing_less_200
    total_all = len(ingested_rows) + len(missing_files)
    
    # Print results to a json file to be read by the agent
    stats = {
        "avg_time": round(avg_time, 2),
        "total_bl_sets": total_all,
        "total_less_100": total_less_100,
        "total_less_200": total_less_200,
        "ingested_count": len(ingested_rows),
        "missing_count": len(missing_files)
    }
    
    with open("data/bricklink_stats.json", "w") as sf:
        json.dump(stats, sf, indent=2)
        
    print("=== Analysis Completed ===")

if __name__ == "__main__":
    main()

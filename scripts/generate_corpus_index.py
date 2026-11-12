import os
import json
import sqlite3

DB_PATH = "data/catalog/models_catalog.db"
OUT_PATH = "data/dataset_splits/corpus_all_pairs.json"

COMPLEXITY_TIERS = [
    ("MICRO", 0, 50),
    ("SMALL", 51, 150),
    ("MEDIUM", 151, 400),
    ("COMPLEX", 401, 1000),
    ("EXPERT", 1001, 3000),
    ("MONUMENTAL", 3001, 999999),
]

def assign_tier(parts):
    for code, low, high in COMPLEXITY_TIERS:
        if low <= parts <= high:
            return code
    return "UNKNOWN"

def main():
    if not os.path.exists("data/omr_raw/"):
        print("ERROR: data/omr_raw/ not found")
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT pair_id, model_file_path, parts_count, source FROM ground_truth_pairs WHERE model_file_path LIKE '%omr_raw%' ORDER BY parts_count").fetchall()

    corpus = []
    found = 0
    for pair_id, model_file_path, parts_count, source in rows:
        if model_file_path is None:
            continue
        # Check file exists
        if not os.path.exists(model_file_path):
            continue
        category = "OFFICIAL_OMR" if source == "ldraw_omr" else source
        entry = {
            "moc_id": pair_id,
            "io_path": model_file_path,
            "parts_count": parts_count,
            "complexity_tier": assign_tier(parts_count),
            "category": category,
        }
        corpus.append(entry)
        found += 1
    conn.close()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Corpus index generated with {found} entries to {OUT_PATH}")

if __name__ == "__main__":
    main()
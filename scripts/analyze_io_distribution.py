"""Analizar distribución de piezas y pasos en corpus .io."""
from pathlib import Path
import sys
import sqlite3

# Importar io_loader desde LdrawNova
ldraw_nova = Path(__file__).parent.parent.parent / "LdrawNova"
sys.path.insert(0, str(ldraw_nova))

from scripts.bench.io_loader import count_steps_and_parts

io_dir = Path(__file__).parent.parent / "data" / "normalized_io"
db_path = Path(__file__).parent.parent / "data" / "catalog" / "models_catalog.db"

print(f"\n=== ANALIZANDO DISTRIBUCIÓN DE .io FILES ===")
print(f"Directorio: {io_dir}")
print(f"Buscando *.io files...")

io_files = sorted(io_dir.glob("*.io"))
print(f"Total .io files: {len(io_files)}\n")

part_counts = []
step_counts = []
pair_map = {}  # pair_id -> (steps, parts)

for i, io_file in enumerate(io_files):
    result = count_steps_and_parts(io_file)
    if result:
        steps, parts = result
        part_counts.append(parts)
        step_counts.append(steps)
        pair_id = io_file.stem
        if pair_id.endswith("_normalized"):
            pair_id = pair_id[:-len("_normalized")]
        pair_map[pair_id] = (steps, parts)
        
        if i < 20 or (i % 100 == 0):
            print(f"  [{i+1:3d}] {io_file.name:40s} | steps={steps:3d} | parts={parts:5d}")

print(f"\n=== ESTADÍSTICAS REALES ===")
print(f"Total procesados: {len(part_counts)}")
print(f"Min parts: {min(part_counts)} | Max parts: {max(part_counts)} | Avg parts: {sum(part_counts) / len(part_counts):.1f}")
print(f"Min steps: {min(step_counts)} | Max steps: {max(step_counts)} | Avg steps: {sum(step_counts) / len(step_counts):.1f}")

# Buckets
buckets = {
    "micro (2-50)": [p for p in part_counts if 2 <= p <= 50],
    "small (51-200)": [p for p in part_counts if 51 <= p <= 200],
    "medium (201-500)": [p for p in part_counts if 201 <= p <= 500],
    "large (501-1k)": [p for p in part_counts if 501 <= p <= 1000],
    "xlarge (1k+)": [p for p in part_counts if p > 1000],
}

print(f"\n=== DISTRIBUCIÓN POR TIERS ===")
for tier, counts in buckets.items():
    if counts:
        print(f"{tier:20s}: {len(counts):4d} ({100*len(counts)/len(part_counts):5.1f}%) | min={min(counts):5d} max={max(counts):5d} avg={sum(counts)/len(counts):7.1f}")
    else:
        print(f"{tier:20s}: {len(counts):4d} (0.0%)")

print(f"\n=== ACTUALIZANDO BD CON VALORES REALES ===")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

updated = 0
for pair_id, (steps, parts) in pair_map.items():
    cursor.execute(
        "UPDATE ground_truth_pairs SET steps_count = ?, parts_count = ? WHERE pair_id = ?",
        (steps, parts, pair_id)
    )
    updated += cursor.rowcount

conn.commit()
print(f"Actualizados {updated} registros en la BD")
conn.close()

print(f"\n✅ Distribución analizada y BD actualizada")

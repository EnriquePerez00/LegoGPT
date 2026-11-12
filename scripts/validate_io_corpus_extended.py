"""Validar corpus completo con loader extendido (Nivel 1 + 2).

Re-procesa todos los 927 .io files con soporte para:
- Nivel 1: Plain text LDraw (.mpd/.ldr formatos OMR)
- Nivel 2: Resolución de submodelos MPD

Genera reporte actualizado de parts_count/steps_count reales.
"""
from pathlib import Path
import sys
import json
from datetime import datetime

ldraw_nova = Path(__file__).parent.parent.parent / "LdrawNova"
sys.path.insert(0, str(ldraw_nova))

from scripts.bench.io_loader_extended import count_steps_and_parts_extended

io_dir = Path(__file__).parent.parent / "data" / "normalized_io"
output_dir = Path(__file__).parent.parent / "outputs"
output_dir.mkdir(exist_ok=True)

print(f"\n=== VALIDANDO CORPUS CON LOADER EXTENDIDO ===")
print(f"Directorio: {io_dir}")
print(f"Timestamp: {datetime.now().isoformat()}\n")

io_files = sorted(io_dir.glob("*.io"))
print(f"Total .io files: {len(io_files)}")

results = []
failed = []

for i, io_file in enumerate(io_files):
    pair_id = io_file.stem.replace("_normalized", "")
    result = count_steps_and_parts_extended(io_file)
    
    if result:
        steps, parts = result
        results.append({
            "pair_id": pair_id,
            "filename": io_file.name,
            "steps": steps,
            "parts": parts,
            "size_bytes": io_file.stat().st_size,
        })
    else:
        failed.append({
            "pair_id": pair_id,
            "filename": io_file.name,
            "error": "Failed to parse",
        })
    
    if (i + 1) % 100 == 0 or (i + 1) == len(io_files):
        pct_parsed = 100 * len(results) / (i + 1)
        print(f"  [{i+1:3d}/{len(io_files)}] Parsed: {len(results):3d} | Failed: {len(failed):3d} ({pct_parsed:5.1f}%)")

# Estadísticas
if results:
    parts_counts = [r["parts"] for r in results]
    steps_counts = [r["steps"] for r in results]
    print(f"\n=== ESTADÍSTICAS REALES ===")
    print(f"Total parsed: {len(results)}")
    print(f"Total failed: {len(failed)}")
    print(f"Min parts: {min(parts_counts)}, Max: {max(parts_counts)}, Avg: {sum(parts_counts)/len(parts_counts):.1f}")
    print(f"Min steps: {min(steps_counts)}, Max: {max(steps_counts)}, Avg: {sum(steps_counts)/len(steps_counts):.1f}")
    
    # Buckets
    buckets = {
        "micro (2-50)": [p for p in parts_counts if 2 <= p <= 50],
        "small (51-200)": [p for p in parts_counts if 51 <= p <= 200],
        "medium (201-500)": [p for p in parts_counts if 201 <= p <= 500],
        "large (501-1000)": [p for p in parts_counts if 501 <= p <= 1000],
        "xlarge (1001+)": [p for p in parts_counts if p > 1000],
    }
    print(f"\n=== DISTRIBUCIÓN POR TIERS ===")
    for tier, counts in buckets.items():
        if counts:
            print(f"  {tier:20s}: {len(counts):4d} ({100*len(counts)/len(results):5.1f}%)")

# Guardar reporte
report = {
    "timestamp": datetime.now().isoformat(),
    "total_files": len(io_files),
    "parsed_count": len(results),
    "failed_count": len(failed),
    "parsed_pct": 100 * len(results) / len(io_files),
    "results": sorted(results, key=lambda x: x["pair_id"]),
    "failed": failed,
}

report_path = output_dir / "io_validation_extended_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Reporte guardado: {report_path}")

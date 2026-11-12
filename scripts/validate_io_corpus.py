"""Validar integridad del corpus .io.

Genera lista de archivos válidos e inválidos.
Guarda reporte en outputs/io_validation_report.json
"""
from pathlib import Path
import json
import sys
import zipfile
from datetime import datetime

# Importar io_loader desde LdrawNova
ldraw_nova = Path(__file__).parent.parent.parent / "LdrawNova"
sys.path.insert(0, str(ldraw_nova))

from scripts.bench.io_loader import count_steps_and_parts

io_dir = Path(__file__).parent.parent / "data" / "normalized_io"
output_dir = Path(__file__).parent.parent / "outputs"
output_dir.mkdir(exist_ok=True)

print(f"\n=== VALIDANDO CORPUS .io ===")
print(f"Directorio: {io_dir}")
print(f"Timestamp: {datetime.now().isoformat()}\n")

io_files = sorted(io_dir.glob("*.io"))
print(f"Total .io files: {len(io_files)}")

valid = []
invalid = []

for i, io_file in enumerate(io_files):
    # Verificar si es ZIP válido
    is_zip = zipfile.is_zipfile(io_file)
    
    # Intentar contar steps/parts
    try:
        result = count_steps_and_parts(io_file)
        parseable = result is not None
        if parseable:
            steps, parts = result
        else:
            steps, parts = None, None
    except Exception as e:
        parseable = False
        steps, parts = None, None
    
    entry = {
        "pair_id": io_file.stem.replace("_normalized", ""),
        "filename": io_file.name,
        "size_bytes": io_file.stat().st_size,
        "is_valid_zip": is_zip,
        "is_parseable": parseable,
        "steps": steps,
        "parts": parts,
    }
    
    if parseable:
        valid.append(entry)
    else:
        invalid.append(entry)
    
    if (i + 1) % 100 == 0 or (i + 1) == len(io_files):
        print(f"  [{i+1:3d}/{len(io_files)}] Valid: {len(valid):3d} | Invalid: {len(invalid):3d}")

# Generar reporte
report = {
    "timestamp": datetime.now().isoformat(),
    "total_files": len(io_files),
    "valid_count": len(valid),
    "invalid_count": len(invalid),
    "valid_pct": 100 * len(valid) / len(io_files),
    "invalid_pct": 100 * len(invalid) / len(io_files),
    "valid_files": valid,
    "invalid_files": invalid,
}

# Guardar reporte
report_path = output_dir / "io_validation_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n=== RESUMEN ===")
print(f"VÁLIDOS:   {len(valid):3d} ({100*len(valid)/len(io_files):5.1f}%)")
print(f"INVÁLIDOS: {len(invalid):3d} ({100*len(invalid)/len(io_files):5.1f}%)")

print(f"\n✅ Reporte guardado: {report_path}")

# Crear lista de pair_ids válidos
valid_pair_ids = [e["pair_id"] for e in valid]
valid_ids_path = output_dir / "valid_pair_ids.txt"
with open(valid_ids_path, 'w') as f:
    for pair_id in sorted(valid_pair_ids):
        f.write(f"{pair_id}\n")

print(f"✅ Lista de pair_ids válidos: {valid_ids_path}")

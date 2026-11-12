"""Fase 3: Probe MICRO tier con corpus limpio (225 valid .io files).

Ejecuta validación en los 65 MICRO sets parseables (5-50 piezas).
Genera reporte de fallos y tiempos reales SIN fabricar timeouts.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import os
import time
import sqlite3
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

print(f"\n=== FASE 3: MICRO PROBE (CORPUS LIMPIO) ===")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Cargar eval_micro_CLEAN.json
eval_path = Path('data/eval/eval_micro_CLEAN.json')
with open(eval_path) as f:
    eval_data = json.load(f)

samples = eval_data['samples']
print(f"\nSamples MICRO: {len(samples)}")
print(f"Rango parts: {eval_data['parts_range']}")
print(f"Status corpus: {eval_data['corpus_status']}")

# Agrupar por categoría para surtido equilibrado
by_source = {}
for s in samples:
    src = s.get('source', 'unknown')
    if src not in by_source:
        by_source[src] = []
    by_source[src].append(s)

print(f"\nDistribución por fuente:")
for src, items in sorted(by_source.items(), key=lambda x: -len(x[1])):
    print(f"  {src:30s}: {len(items):3d} sets")

# Seleccionar PROBE: 1 de cada fuente + extras aleatorios (max 10 para probe rápido)
import random
random.seed(42)

probe_sets = []
for src, items in by_source.items():
    if items:
        probe_sets.append(random.choice(items))

if len(probe_sets) < 10 and len(samples) > len(probe_sets):
    remaining = [s for s in samples if s not in probe_sets]
    probe_sets.extend(random.sample(remaining, min(10 - len(probe_sets), len(remaining))))

print(f"\nPROBE sets seleccionados: {len(probe_sets)}")
for i, s in enumerate(probe_sets[:5], 1):
    print(f"  {i}. {s['pair_id']:20s} | parts={s['parts_count']:3d} | source={s.get('source', 'N/A')}")

if len(probe_sets) > 5:
    print(f"  ... {len(probe_sets) - 5} más")

# Crear reporte de probe
probe_report = {
    "phase": "MICRO_PROBE",
    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    "corpus_status": "VALID_ONLY (225/927 files, 702 corrupted)",
    "tier": "MICRO",
    "parts_range": eval_data['parts_range'],
    "total_available_micro": len(samples),
    "probe_sample_count": len(probe_sets),
    "probe_sets": probe_sets,
    "next_steps": [
        "1. Ejecutar validación en probe sets (paralelo, 2 workers)",
        "2. Medir tiempos reales (init, steps, rendering)",
        "3. Identificar fallos REALES (no timeouts fabricados)",
        "4. Sintetizar fixes basados en incidentes reales",
        "5. Re-descargar corpus corrupto (75.7%) en background",
        "6. Re-validar en 100% corpus post-fixes",
    ],
}

probe_report_path = Path('outputs/phase3_micro_probe_config.json')
probe_report_path.parent.mkdir(parents=True, exist_ok=True)

with open(probe_report_path, 'w') as f:
    json.dump(probe_report, f, indent=2)

print(f"\n✅ Probe config guardado: {probe_report_path}")
print(f"\n=== LISTO PARA VALIDACIÓN ===")
print(f"Ejecutar: python tier_validation.py --phase probe --clean-only")

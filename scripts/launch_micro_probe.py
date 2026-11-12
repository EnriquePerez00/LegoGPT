"""Fase 3: Lanzar MICRO probe con corpus extendido validado (927 archivos, 164 MICRO sets).

Este script:
1. Selecciona ~15 sets MICRO representativos para probe rápido
2. Prepara configuración paralela (2 workers, timeout 60s por set)
3. Genera reporte de próximos pasos
"""
import json
import random
from pathlib import Path
from datetime import datetime

print(f"\n=== FASE 3: LANZAR MICRO PROBE ===")
print(f"Timestamp: {datetime.now().isoformat()}\n")

# Cargar eval_micro_CLEAN.json actualizado
eval_path = Path('data/eval/eval_micro_CLEAN.json')
with open(eval_path) as f:
    eval_data = json.load(f)

samples = eval_data['samples']
print(f"Total MICRO sets: {len(samples)}")
print(f"Rango parts: {eval_data['parts_range']}")
print(f"Status: {eval_data['corpus_status']}")

# Seleccionar probe representativo (15 sets)
random.seed(42)
probe_sets = random.sample(samples, min(15, len(samples)))
probe_sets.sort(key=lambda x: x['pair_id'])

print(f"\nPROBE selection: {len(probe_sets)} sets")
for s in probe_sets[:5]:
    print(f"  {s['pair_id']:20s} | parts={s['parts_count']:2d} | source={s.get('source', 'N/A')[:20]}")
if len(probe_sets) > 5:
    print(f"  ... {len(probe_sets) - 5} más")

# Agrupar por tier para surtido
by_source = {}
for s in probe_sets:
    src = s.get('source', 'unknown')
    if src not in by_source:
        by_source[src] = 0
    by_source[src] += 1

print(f"\nDistribución por fuente:")
for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
    print(f"  {src:30s}: {count:3d} sets")

# Crear config de probe
probe_config = {
    "phase": "MICRO_PROBE",
    "timestamp": datetime.now().isoformat(),
    "status": "READY",
    "corpus_status": "All 927 valid (extended loader Nivel 1+2)",
    "tier": "MICRO",
    "parts_range": eval_data['parts_range'],
    "total_available_micro": len(samples),
    "probe_sample_count": len(probe_sets),
    "probe_sets": probe_sets,
    "execution_config": {
        "parallel_workers": 2,
        "timeout_per_set_sec": 60,
        "expected_duration_min": "10-20",
    },
    "next_steps": [
        "1. Execute probe: python scripts/run_moc_continuous_improvement_loop.py --eval-only data/eval/eval_micro_probe_sample.json",
        "2. Monitor metrics: success_rate, failure_types, timings",
        "3. Collect real incidents (non-timeout failures)",
        "4. Synthesize fixes based on actual issues",
        "5. Scale to SMALL tier (337 sets, 36.4% corpus)",
    ],
}

probe_sample_path = Path('data/eval/eval_micro_probe_sample.json')
probe_sample_data = {
    "dataset_name": "eval_micro_probe_sample",
    "tier": "MICRO",
    "description": "Representative sample for quick MICRO probe (15 sets, ~10-20 min)",
    "total_samples": len(probe_sets),
    "samples": probe_sets,
}

with open(probe_sample_path, 'w') as f:
    json.dump(probe_sample_data, f, indent=2)

config_path = Path('outputs/phase3_micro_probe_config_final.json')
with open(config_path, 'w') as f:
    json.dump(probe_config, f, indent=2)

print(f"\n✅ Probe sample: {probe_sample_path}")
print(f"✅ Config: {config_path}")
print(f"\n=== LISTO PARA PROBE ===")
print(f"Total MICRO sets disponibles: {len(samples)}")
print(f"Probe sample: {len(probe_sets)} sets (10-20 min estimado)")
print(f"\nPróximo comando:")
print(f"  python scripts/run_moc_continuous_improvement_loop.py --eval-only data/eval/eval_micro_probe_sample.json")

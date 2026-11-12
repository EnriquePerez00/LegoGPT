"""Fase 3: Ejecutar MICRO probe con loader extendido.

Carga eval_micro_probe_sample.json (15 sets representativos) y ejecuta
validación básica para verificar que el pipeline end-to-end funciona.

Métricas:
- Tiempos de load/parse
- Success rate
- Failure types (geometry, render, parser)
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ldraw_nova = Path.cwd().parent / "LdrawNova"
sys.path.insert(0, str(ldraw_nova))

from scripts.bench.io_loader_extended import count_steps_and_parts_extended, read_model_ldr_extended

eval_path = Path('data/eval/eval_micro_probe_sample.json')
with open(eval_path) as f:
    eval_data = json.load(f)

samples = eval_data['samples']
print(f"\n=== MICRO PROBE EXECUTION ===")
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Samples: {len(samples)}")
print(f"Estimated time: ~{len(samples)*10}-{len(samples)*20} seconds\n")

io_dir = Path('data/normalized_io')
results = {
    'success': [],
    'failed': [],
    'timings': [],
}

start_time = time.time()

for i, sample in enumerate(samples):
    pair_id = sample['pair_id']
    io_file = io_dir / f"{pair_id}_normalized.io"
    
    if not io_file.exists():
        results['failed'].append({
            'pair_id': pair_id,
            'error': 'File not found',
        })
        continue
    
    # Time: load + parse
    t0 = time.time()
    try:
        text = read_model_ldr_extended(io_file)
        result = count_steps_and_parts_extended(io_file)
        
        if result:
            steps, parts = result
            elapsed_ms = (time.time() - t0) * 1000
            results['success'].append({
                'pair_id': pair_id,
                'steps': steps,
                'parts': parts,
                'elapsed_ms': elapsed_ms,
            })
            results['timings'].append(elapsed_ms)
            status = '✅'
        else:
            results['failed'].append({
                'pair_id': pair_id,
                'error': 'Parse returned None',
            })
            status = '❌'
    except Exception as e:
        results['failed'].append({
            'pair_id': pair_id,
            'error': str(type(e).__name__),
        })
        status = '❌'
    
    pct = 100 * (i + 1) / len(samples)
    print(f"[{i+1:2d}/{len(samples)}] {status} {pair_id:20s} | {pct:5.1f}%")

total_time = time.time() - start_time

print(f"\n=== PROBE RESULTS ===")
print(f"Total time: {total_time:.1f}s")
print(f"Success: {len(results['success'])}/{len(samples)} ({100*len(results['success'])/len(samples):.1f}%)")
print(f"Failed: {len(results['failed'])}")

if results['timings']:
    print(f"\nTiming stats:")
    print(f"  Min: {min(results['timings']):.1f}ms")
    print(f"  Max: {max(results['timings']):.1f}ms")
    print(f"  Avg: {sum(results['timings'])/len(results['timings']):.1f}ms")

if results['success']:
    print(f"\nSuccessful samples:")
    for res in results['success'][:5]:
        print(f"  {res['pair_id']:20s} | steps={res['steps']:3d} parts={res['parts']:4d} | {res['elapsed_ms']:6.1f}ms")
    if len(results['success']) > 5:
        print(f"  ... {len(results['success']) - 5} más")

if results['failed']:
    print(f"\nFailed samples:")
    for res in results['failed']:
        print(f"  {res['pair_id']:20s} | {res['error']}")

# Guardar reporte
report = {
    'timestamp': datetime.now().isoformat(),
    'phase': 'MICRO_PROBE',
    'total_samples': len(samples),
    'success_count': len(results['success']),
    'success_rate': 100 * len(results['success']) / len(samples),
    'failed_count': len(results['failed']),
    'total_time_sec': total_time,
    'results': results,
}

report_path = Path('outputs/micro_probe_results.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Reporte guardado: {report_path}")

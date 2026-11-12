#!/usr/bin/env python3
"""
Quick test: FASE 0 prueba minimal - ejecuta evaluate_moc_fail_fast sobre 1 set MICRO
para verificar que el histograma recall@N se calcula correctamente.
"""

import sys
import os

# Ensure src is importable from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.oracle_step_debugger import OracleStepDebugger, GeneralizedFixBundle
from src.geometry_cache import GeometryCache

def test_recall_at_n():
    print("=" * 80)
    print("FASE 0 - Test: Metric recall@N en 1 set MICRO")
    print("=" * 80)
    
    # Load GT
    io_path = "data/omr_raw/10001-1.mpd"
    if not os.path.exists(io_path):
        print(f"❌ No encontrado: {io_path}")
        return False
    
    print(f"Loading GT from: {io_path}")
    
    # Initialize debugger
    geom = GeometryCache.get()
    bundle = GeneralizedFixBundle()
    debugger = OracleStepDebugger(geom_resolver=geom, fix_bundle=bundle)
    
    # Evaluate
    print("Evaluating MOC (fail_fast mode)...")
    result = debugger.evaluate_moc_fail_fast(
        moc_id="TEST-10001-1",
        io_path=io_path,
        abort_on_first_error=False,  # Don't abort early, evaluate all steps
        max_steps_to_eval=10,
    )
    
    print("\n" + "=" * 80)
    print("RESULTADOS:")
    print("=" * 80)
    print(f"MOC ID:                    {result['moc_id']}")
    print(f"Success:                   {result['success']}")
    print(f"Completed all steps:       {result['completed_all_steps']}")
    print(f"Steps survived:            {result['steps_survived']}/{result['total_steps']}")
    print(f"Survival ratio:            {result['survival_ratio']:.1%}")
    
    if "recall_at_n" in result:
        print(f"\n🎯 RECALL@N METRIC (NEW):")
        print("-" * 80)
        for n in range(1, 9):
            recall = result["recall_at_n"][n]
            bar = "█" * int(recall * 40)
            print(f"  Recall@{n}: {recall*100:5.1f}% {bar}")
        print(f"\nSummary: {result.get('recall_at_n_summary', 'N/A')}")
    else:
        print(f"\n⚠️  recall_at_n NOT in result keys!")
        print(f"Available keys: {list(result.keys())}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETO - metric recall@N está funcional\n")
    return True

if __name__ == "__main__":
    try:
        test_recall_at_n()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


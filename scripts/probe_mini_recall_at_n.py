#!/usr/bin/env python3
"""
FASE 1 - PROBE MINIMALISTA: Test recall@N en 1 set real

Ejecuta evaluate_moc_fail_fast directamente sin la sobrecarga del tier_validation.py
para validar que la métrica recall@N funciona correctamente.
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.oracle_step_debugger import OracleStepDebugger, GeneralizedFixBundle
from src.geometry_cache import GeometryCache

def mini_probe():
    print("=" * 80)
    print("FASE 1 - PROBE MINIMALISTA: Validación recall@N")
    print("=" * 80)
    
    # Select 1 MICRO set
    io_path = "data/omr_raw/10001-1.mpd"
    
    if not os.path.exists(io_path):
        print(f"❌ {io_path} no existe")
        return False
    
    print(f"\n📂 GT file: {io_path}")
    print(f"   (OMR-10001-1, MICRO tier)")
    
    print("\n⏱️  Inicializando OracleStepDebugger (puede tomar ~30s, solo primera vez)...")
    t0 = time.time()
    
    try:
        geom = GeometryCache.get()
        bundle = GeneralizedFixBundle()
        debugger = OracleStepDebugger(geom_resolver=geom, fix_bundle=bundle)
        
        elapsed_init = time.time() - t0
        print(f"   ✅ Inicializado en {elapsed_init:.1f}s")
        
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n⏱️  Evaluando MOC (evaluate_moc_fail_fast)...")
    t0 = time.time()
    
    try:
        result = debugger.evaluate_moc_fail_fast(
            moc_id="OMR-10001-1",
            io_path=io_path,
            abort_on_first_error=False,  # Evalúa todos los pasos para tener histograma completo
            max_steps_to_eval=30,
        )
        elapsed_eval = time.time() - t0
        
    except Exception as e:
        print(f"❌ Error evaluando: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"   ✅ Evaluado en {elapsed_eval:.1f}s")
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTADOS:")
    print("=" * 80)
    
    print(f"\n📋 Set: {result['moc_id']}")
    print(f"   Success:               {result['success']}")
    print(f"   Completed all steps:   {result['completed_all_steps']}")
    print(f"   Steps survived:        {result['steps_survived']}/{result['total_steps']}")
    print(f"   Survival ratio:        {result['survival_ratio']:.1%}")
    print(f"   Total incidents:       {len(result['incidents'])}")
    
    # The key part: recall@N
    if "recall_at_n" in result:
        print(f"\n🎯 RECALL@N METRIC (★ NEW ★):")
        print("   " + "-" * 76)
        
        recall_dict = result["recall_at_n"]
        for n in range(1, 9):
            if n in recall_dict:
                recall = recall_dict[n]
                bar = "█" * int(recall * 60)
                print(f"   Recall@{n}: {recall*100:5.1f}% │{bar:<60}│")
        
        print("   " + "-" * 76)
        print(f"\n   Summary: {result.get('recall_at_n_summary', 'N/A')}")
        
        # Analysis
        print(f"\n📊 ANÁLISIS:")
        for n in range(1, 9):
            if n in recall_dict and recall_dict[n] >= 0.95:
                print(f"   ✅ N* >= {n} para Recall >= 95%")
                break
        else:
            print(f"   ⚠️  Recall no alcanza 95% incluso en N=8")
        
        return True
    else:
        print(f"\n❌ recall_at_n NOT in result!")
        print(f"   Available keys: {list(result.keys())}")
        return False

if __name__ == "__main__":
    print()
    success = mini_probe()
    print("\n" + "=" * 80)
    if success:
        print("✅ TEST EXITOSO: Métrica recall@N funcional y listado en resultado")
    else:
        print("❌ TEST FALLIDO")
    print("=" * 80 + "\n")
    sys.exit(0 if success else 1)

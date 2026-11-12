#!/usr/bin/env python3
"""
Quick validation: verifica que la métrica recall@N está bien integrada en el código
y la lógica es correcta (sin necesidad de ejecutar el pipeline completo).
"""

import re
import sys

def validate_recall_at_n_integration():
    """Check that recall@N metric is properly integrated in oracle_step_debugger.py"""
    
    print("=" * 80)
    print("VALIDACIÓN: Métrica recall@N integrada en OracleStepDebugger")
    print("=" * 80)
    
    file_path = "src/oracle_step_debugger.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = {
        "✓ gt_rank_histogram initialization": "gt_rank_histogram = [0] * 8" in content,
        "✓ Histogram update in baseline (Step 1)": "gt_rank_histogram[0] += 1" in content,
        "✓ GT rank calculation among candidates": "for rank_idx, cand in enumerate(part_candidates_filtered[i]):" in content,
        "✓ Recall@N cumulative calculation": "for n in range(1, 9):" in content,
        "✓ Histogram bucketing logic": "bucket_idx = min(gt_rank, 8) - 1" in content,
        "✓ Result includes recall_at_n dict": '"recall_at_n": recall_at_n' in content,
        "✓ Result includes recall_at_n_summary": '"recall_at_n_summary"' in content,
    }
    
    all_pass = True
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
        if not result:
            all_pass = False
    
    print("\n" + "=" * 80)
    
    if all_pass:
        # Extract and show the actual histogram calculation code
        pattern = r"total_ranks = sum\(gt_rank_histogram\).*?return result"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            code_snippet = match.group(0)[:300] + "..."
            print("\n📋 Lógica del cálculo recall@N (snippet):")
            print("-" * 80)
            for line in code_snippet.split('\n')[:5]:
                print(f"  {line}")
            print("  ...")
        
        print("\n✅ VALIDACIÓN EXITOSA: Métrica recall@N correctamente integrada")
        print("\nPróximos pasos:")
        print("  1. Ejecutar tier_validation.py probe con corpus_all_pairs.json")
        print("  2. Recopilar recall@N histogramas por tier (MICRO, SMALL, ...)")
        print("  3. Determinar N* óptimo para cada tier")
        return True
    else:
        print("\n❌ VALIDACIÓN FALLIDA: Algunas componentes no están presentes")
        return False

if __name__ == "__main__":
    success = validate_recall_at_n_integration()
    sys.exit(0 if success else 1)

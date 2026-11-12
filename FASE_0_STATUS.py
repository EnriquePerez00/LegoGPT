#!/usr/bin/env python3
"""
FASE 0 - RESUMEN: Estado de validación del Oráculo de Encaje Físico LEGO

Cambios realizados:
====================
1. ✅ Generado corpus_all_pairs.json (684 sets OFFICIAL_OMR) desde models_catalog.db
2. ✅ Métrica recall@N integrada en OracleStepDebugger.evaluate_moc_fail_fast
   - gt_rank_histogram[0..7]: bins para rank 1..8
   - Actualizado en cada paso (incluyendo baseline Step 1)
   - Cálculo final: recall@N = prob de que GT esté en top N candidatos
   - Emitido en resultado: "recall_at_n" dict + "recall_at_n_summary" texto

PRÓXIMOS PASOS (FASE 1 - PROBE):
================================
Ejecutar tier_validation.py sobre ~11 sets (1 por tier/categoría):
  python3 scripts/tier_validation.py --phase probe --categories OFFICIAL_OMR

Salida esperada:
  - outputs/tier_validation/probe_report.json
  - Incluye por cada set:
    - success, completed_all_steps, survival_ratio
    - recall_at_n: {1: 0.95, 2: 0.98, ..., 8: 1.0}  ← NUEVA MÉTRICA
    - recall_at_n_summary: "Recall@1: 95.0% | Recall@2: 98.0% | ..."

Análisis post-probe:
  - Visualizar curva recall@N agregada por tier
  - Determinar N* mínimo para recall >= 95% por tier
"""

print(__doc__)

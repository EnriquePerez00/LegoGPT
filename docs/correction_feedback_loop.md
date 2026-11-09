# Bucle de Retroalimentacion: Correccion Humana -> Mejora Generica

## Vision general

```
[Viewer 3D]  ->  apply_corrections.py  ->  diagnose_corrections.py  ->  run_moc_continuous_improvement_loop.py
   (human GT)      (grafo corregido           (causa raiz +               (re-entrena con bundle
                    + dataset.jsonl)            GeneralizedFixBundle)        para TODOS los MOCs)
```

## Etapa A: Viewer - confirmar correccion

1. Clic en pieza -> gizmo de traslacion.
2. Arrastra flechas (mover) / Rotar para rotar.
3. OK o Enter -> registra SOLO esa pieza (dedup por part_index).
4. Exportar JSON -> corrections_<set>_<ts>.json.

## Etapa B: apply_corrections.py

```bash
python scripts/apply_corrections.py \
  --corrections corrections_75295-1_<ts>.json \
  --graph outputs/75295-1_construction_graph.json \
  --out outputs/75295-1_construction_graph.corrected.json \
  --dataset outputs/corrections_dataset.jsonl
```

Que hace:
- Convierte delta_pos viewer -> LDraw (invierte Y).
- Actualiza absolute_transform del nodo.
- Re-deriva conexiones via LDrawGeometry.get_part_ports() + proximidad.
- Acumula corrections_dataset.jsonl (senal de entrenamiento GNN).

## Etapa C: diagnose_corrections.py

```bash
python scripts/diagnose_corrections.py \
  --corrections corrections_75295-1_<ts>.json \
  --graph outputs/75295-1_construction_graph.json \
  --out-diagnosis outputs/corrections_diagnosis.json \
  --out-bundle outputs/fix_bundle.json \
  --prior-bundle outputs/fix_bundle_prev.json
```

Por cada correccion:
1. Reconstruye prior_gt_nodes (piezas antes de ese paso).
2. Re-genera candidatos con PhysicalHypothesisGenerator (mismos datos).
3. OracleStepDebugger.diagnose_step_failure(gt_part=pose_corregida, ...).
4. Produce StepFailureIncident con:
   - root_cause_category: VISUAL_SCORER_RANKING_ERROR / HYPOTHESIS_NOT_GENERATED / ...
   - sub_cause: CHAMFER_WEIGHT_IMBALANCE / BEAM_WIDTH_TOO_NARROW / ...
   - gt_rank_in_candidates: posicion de la pose correcta en el ranking
5. PatternClusterAnalyzer -> GeneralizedFixBundle (mejoras genericas).

Outputs:
- corrections_diagnosis.json: informe causal por pieza.
- fix_bundle.json: bundle de mejoras genericas para el solver.

## Etapa D: Cerrar el bucle

```bash
python scripts/run_moc_continuous_improvement_loop.py \
  --input-fixes outputs/fix_bundle.json \
  --moc-dir data/mocs/ --iterations 1
```

Las mejoras son GENERICAS: afectan a todas las decisiones del solver con la misma
causa raiz, no solo a la pieza corregida manualmente.

## Por que esto elimina los if startswith("75295")

Cada error que antes necesitaba un if page==5: se convierte en una entrada del
GeneralizedFixBundle que funciona para cualquier set_num con el mismo patron.

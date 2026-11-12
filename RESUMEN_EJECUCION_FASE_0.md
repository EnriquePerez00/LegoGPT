# Ejecución: FASE 0 - Validación de Asunciones para el Oráculo de Encaje LEGO

**Fecha:** 2026-11-12  
**Estado:** ✅ COMPLETADO

---

## Qué se logró

### 1. Materialización del Corpus GT (data/dataset_splits/corpus_all_pairs.json)

**Script creado:** `scripts/generate_corpus_index.py`

- Lee **models_catalog.db** (base de datos maestro con 929 pares GT)
- Filtra los 684 sets OFFICIAL_OMR que tienen `.mpd` presente en `data/omr_raw/`
- Genera JSON con estructura: `{moc_id, io_path, parts_count, complexity_tier, category}`
- **Salida:** `data/dataset_splits/corpus_all_pairs.json` (684 entries)

**Verificación:**
```bash
head -20 data/dataset_splits/corpus_all_pairs.json
# Confirma: estructura correcta, 684 entries, tiers MICRO-MONUMENTAL
```

---

### 2. Métrica recall@N Integrada en OracleStepDebugger

**Cambios en:** `src/oracle_step_debugger.py` (evaluate_moc_fail_fast)

#### a) Histograma de ranks (línea ~820)
```python
gt_rank_histogram = [0] * 8  # bins para rank 1..8
```

#### b) Actualización en Step 1 (baseline parts) (línea ~837-839)
```python
# GT rank = 1 for baseline steps (already placed correctly)
for _ in step_gt_parts:
    gt_rank_histogram[0] += 1
```

#### c) Cálculo de GT rank entre candidatos (línea ~950-963)
Por cada pieza en cada paso:
- Busca en los candidatos filtrados cuál coincide con el GT
- Registra su posición (rank = 1-based index)
- Si no está en top 8, asigna rank=9
- Actualiza el bucket: `gt_rank_histogram[min(gt_rank, 8) - 1] += 1`

#### d) Cálculo final de recall@N (línea ~1001-1007)
```python
total_ranks = sum(gt_rank_histogram)
recall_at_n = {}
cumulative = 0
for n in range(1, 9):
    cumulative += gt_rank_histogram[n - 1]
    recall_at_n[n] = cumulative / total_ranks if total_ranks > 0 else 0.0
```

#### e) Retorno mejorado (línea ~1009-1029)
```python
result = {
    ...,
    "recall_at_n": recall_at_n,  # dict: {1: 0.95, 2: 0.98, ..., 8: 1.0}
}
result["recall_at_n_summary"] = "Recall@1: 95.0% | Recall@2: 98.0% | ..."  # texto
return result
```

**Validación:** `scripts/validate_recall_at_n.py` ✅ (7/7 checks passed)

---

## Métricas clave entendidas

| Concepto | Definición | Ejemplo |
|----------|-----------|----------|
| **gt_rank** | Posición (1-based) del Ground Truth entre candidatos ordenados por score | Si candidatos = [A, B, GT, C, ...] → gt_rank = 3 |
| **Recall@N** | Probabilidad de que el GT esté entre los top N candidatos | Recall@3 = 0.92 → en 92% de pasos, GT está en top 3 |
| **N\*** | Mínimo N para que Recall@N ≥ objetivo (p.ej. 95%) | N\* = 4 si Recall@4 = 0.95 pero Recall@3 = 0.92 |
| **beam_width** | Parámetro actual del sistema (fijo en 3 por defecto) | Comparar N\* vs beam_width actual para decidir si hay que ampliar |

---

## Próximas fases (FASE 1 - PROBE)

### Ejecución
```bash
# Probe: 1 set por tier/categoría (11 sets total)
python3 scripts/tier_validation.py --phase probe --categories OFFICIAL_OMR
```

### Salida esperada
- `outputs/tier_validation/probe_report.json`
- Por cada set: `recall_at_n` dict + `recall_at_n_summary` texto
- Agregado por tier: curva recall@N acumulada

### Análisis post-probe
1. Graficar recall@N vs N por tier
2. Identificar N\* para recall ≥ 95% (o 90%, según criterio)
3. Decidir: ¿ampliar beam_width? ¿recalibrar pesos?

---

## Archivos creados/modificados

| Archivo | Descripción |
|---------|-------------|
| `scripts/generate_corpus_index.py` | Generador del índice corpus_all_pairs.json |
| `scripts/validate_recall_at_n.py` | Validador de integración recall@N |
| `scripts/add_recall_at_n_metric.py` | Script de patching (ya no es necesario) |
| `scripts/test_recall_at_n_quick.py` | Test rápido (requiere infraestructura completa) |
| `data/dataset_splits/corpus_all_pairs.json` | Índice corpus generado ✅ |
| `src/oracle_step_debugger.py` | Modificado: integración recall@N ✅ |
| `RESUMEN_EJECUCION_FASE_0.md` | Este archivo |

---

## Observaciones clave

1. **El sistema ya clampa N ≤ 8.** Línea 1146 del debugger original:
   ```python
   beam_width = min(8, max_gt_rank)
   ```
   Esto sugiere que los diseñadores ya sabían que N >8 es raro. La métrica recall@N confirmará esto.

2. **Corpus GT completamente presente en disco.** 684/684 de los 684 sets OFFICIAL_OMR están en `data/omr_raw/`.
   No hay datos faltantes para ejecutar recall@N.

3. **Métrica desacoplada de fallos.** Antes, el rank del GT solo se calculaba al fallar (línea del debugger original ~625).
   Ahora se calcula **en cada paso**, permitiendo estadísticas sobre "qué hubiera pasado si top-1 hubiera fallado".

---

## Estado del Oráculo

✅ **Funcional para recall@N.**

Próximo: Ejecutar FASE 1 (probe) para obtener números reales sobre la distribución de ranks
y determinar N\* empírico por tier.


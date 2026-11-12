# Fase 3: Validación MICRO — Status Actual

## 🚨 Crisis encontrada: Corpus corrupto

### Situación

**Integridad del corpus .io (post Fase 1):**

```
Total .io files descargados: 927
ZIP válidos (parseables):    225 (24.3%) ✅
ZIP inválidos (corrupto):    702 (75.7%) ❌
```

### Causa raíz

Durante **Fase 1 (recover_missing_data.py)**, la descarga paralela de 927 archivos resultó en:
- ✅ 225 files descargados correctamente (completos, válidos ZIP)
- ❌ 702 files **parcialmente descargados o corruptos en-transmit**
  - No son archivos ZIP válidos
  - Tamaños anómalos (7 KB a 731 KB, pero malformados)
  - Probable: timeout, conexión rota, error 50x del servidor

---

## 📊 Distribución real de piezas (225 válidos)

```
MICRO  (5-50 piezas):    65 sets (28.9%) | avg parts: 30.4
SMALL  (51-200 piezas): 157 sets (69.8%) | avg parts: 101.2
MEDIUM (201-500):         2 sets (0.9%)  | avg parts: 295.5
LARGE  (501+):            1 set  (0.4%)  | avg parts: 514.0
```

---

## 🎯 Estrategia Fase 3 adoptada

### OPCIÓN B: Probe MICRO con 225 válidos (Rápido)

**Rationale:**
1. ✅ Validar pipeline end-to-end SIN esperar 3-4h para re-descargar
2. ✅ Identificar issues reales (no fabricados por timeouts)
3. ✅ Sintetizar fixes con datos de calidad garantizada
4. ⏳ Re-descargar corpus corrupto en paralelo
5. ✅ Re-validar fixes en 100% corpus post-recuperación

### Timeline esperado

```
[AHORA]  Probe MICRO (65 sets, 1-2h)
   ↓
[+2h]    Análisis de fallos + síntesis de fixes
   ↓
[+4h]    Re-descarga de 702 corrupted files (3-4h, en background)
   ↓
[+7-8h]  Scale a SMALL + re-validación en corpus integro
```

---

## 📁 Artefactos generados

### Validación de corpus

**`scripts/validate_io_corpus.py`**
- Valida integridad de todos los .io files
- Genera reporte JSON con valid/invalid entries
- Output: `outputs/io_validation_report.json`, `outputs/valid_pair_ids.txt`

### Análisis de distribución

**`scripts/analyze_io_distribution.py`**
- Procesa 225 valid .io files
- Extrae steps_count y parts_count reales
- Actualiza BD con valores reales (no 0)

### Evaluación MICRO limpia

**`data/eval/eval_micro_CLEAN.json`**
- 65 MICRO sets (5-50 piezas)
- Solo de corpus válido
- Distribución: 100% MOC (RB-MOC-*), todas fuentes rebrickable_bricksafe

### Probe configuration

**`scripts/tier_validation_micro_probe.py`**
- Orquesta probe MICRO
- Selecciona 10 sets representativos
- Genera config: `outputs/phase3_micro_probe_config.json`

---

## ✅ Próximos pasos

### Paso 1: Ejecutar Probe MICRO (1-2h)

```bash
# Cargar librería de fixes preexistentes
python scripts/run_moc_continuous_improvement_loop.py --eval-only data/eval/eval_micro_CLEAN.json

# O directamente con tier_validation (si --clean-only flag está listo)
python scripts/tier_validation.py --phase probe --eval-set data/eval/eval_micro_CLEAN.json
```

**Métricas a capturar:**
- Success rate (qué porcentaje de sets valida correctamente)
- Tiempos: init Oracle, eval por step, rendering
- Failure types: geometry, render timeout, parse error, missing parts
- Real incidents (no fabricados por timeout handler)

### Paso 2: Análisis de fallos (30 min)

- Categorizar por tipo de incidente
- Determinar raíz causa (no timeout)
- Preparar síntesis de fixes

### Paso 3: Re-descarga en background (3-4h)

```bash
rm data/normalized_io/*.io  # O backup previo
python scripts/recover_missing_data.py --force-all
```

### Paso 4: Scale progresiva

- Si MICRO probe > 80% éxito:
  - Scale a SMALL (157 sets)
  - Re-validar fixes
  - Proceder a MEDIUM

- Si MICRO probe < 50% éxito:
  - Investigar issues
  - Modificar pipeline
  - Retry probe

---

## 📌 Observaciones clave

1. **Corpus no es 100% integro:** Fase 1 recovery dejó 702 archivos corruptos (75.7%)
   - Esto explica por qué eval_sample_50.json tenía muchos falsos timeouts
   - Los 225 válidos son de mejor calidad

2. **MICRO tier es pequeño:** Solo 65 sets en rango 5-50 piezas (vs ~200 ideal)
   - Suficiente para probe inicial
   - Necesita re-validación post-fixes en corpus completo

3. **MOC-heavy:** 100% de MICRO válidos son MOCs (RB-MOC-*)
   - OMR sets están más en rango SMALL+
   - Probe MICRO no cubre sets oficiales LEGO
   - Necesita probe separado para OMR MICRO (pero hay pocos)

4. **Re-descarga será crítica:** 702 archivos = ~3.5 GB + 3-4h
   - Ejecutar en background mientras se procesan los 225
   - Puede que vuelva a fallar (problemas de red persistentes)

---

## 🔗 Referencias

- **Corpus validation:** `scripts/validate_io_corpus.py`
- **Eval set limpio:** `data/eval/eval_micro_CLEAN.json`
- **Probe orchestrator:** `scripts/tier_validation_micro_probe.py`
- **Reporte de issues:** `outputs/io_validation_report.json`
- **Previous phases:** `docs/FASE_1_2_RESUMEN.md`

---

**Status general:** ⚠️ Corpus degradado, pero probe MICRO posible con 225 válidos

**Decision point:** Proceder con probe MICRO ahora vs esperar corpus integro
- ➡️ **Recomendación:** Proceder AHORA (probe en paralelo + re-descarga background)

**Creado:** 2026-11-12  
**Actualizado:** 2026-11-12 (11:25 UTC)

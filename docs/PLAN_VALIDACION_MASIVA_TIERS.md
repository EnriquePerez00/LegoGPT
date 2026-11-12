# Plan de Validación Masiva Progresiva por Tier

> Objetivo: validar la cadena de inferencia geométrica (Oracle) sobre el corpus
> completo de 927 sets, de forma progresiva y con evidencia por tier.

## 1. Definición de tiers

Los tiers se definen por complejidad (nº de piezas), con la **misma lógica** que
`scripts/stratify_corpus_dataset.py`:

| Tier | Rango de piezas | Descripción |
|---|---|---|
| MICRO | 2 – 50 | Micro |
| SMALL | 51 – 150 | Pequeño |
| MEDIUM | 151 – 400 | Medio |
| COMPLEX | 401 – 1000 | Complejo |
| EXPERT | 1001 – 3000 | Experto |
| MONUMENTAL | > 3000 | Monumental |

Categorías: `MOC` (sets propios) y `OFFICIAL_OMR` (manuales oficiales).

### Población del corpus (`data/dataset_splits/corpus_all_pairs.json`)

| Categoría | MICRO | SMALL | MEDIUM | COMPLEX | EXPERT | MONUMENTAL | Total |
|---|---|---|---|---|---|---|---|
| MOC | 192 | 35 | 10 | 5 | 0 | 1 | 243 |
| OFFICIAL_OMR | 77 | 152 | 168 | 112 | 114 | 61 | 684 |
| **Total** | **269** | **187** | **178** | **117** | **114** | **62** | **927** |

> Nota: `MOC` no tiene sets en tier EXPERT. Las combinaciones vacías se saltan
> automáticamente en todas las fases.

## 2. Las 3 fases

### FASE 1 — PROBE (1 set por tier)
- Valida **1 set por cada combinación (categoría, tier)** → 11 sets.
- Detecta fallos, identifica causas raíz, mide tiempos base.
- Muestreo **espaciado** sobre el orden por nº de piezas → cubre todo el rango
  de complejidad del tier (no solo los más pequeños).
- Salida: `outputs/tier_validation/probe_report.json`

### FASE 2 — FIX
- `PatternClusterAnalyzer` agrupa los incidentes del probe en patrones.
- Sintetiza un `GeneralizedFixBundle` con las correcciones generalizadas.
- Compara el bundle nuevo contra el previo y **lista explícitamente los cambios**.
- Salida: `outputs/tier_validation/fix_bundle.json` + `fix_report.json`

### FASE 3 — SCALE (escalado progresivo)
- Evalúa N sets adicionales por tier (excluye los ya validados en probe).
- Usa el bundle de la fase 2 → mide el efecto real de las correcciones.
- Proyecta el coste de validar el corpus completo usando los tiempos medidos.
- Salida: `outputs/tier_validation/scale_report_N.json`

## 3. Ejecución

```bash
# Ciclo completo (probe → fix → scale con 5 sets/tier)
bash scripts/run_tier_validation_bg.sh all 5

# Solo una fase
bash scripts/run_tier_validation_bg.sh probe
bash scripts/run_tier_validation_bg.sh scale 20

# Resultados por pantalla
bash scripts/monitor_tier_validation.sh          # snapshot
bash scripts/monitor_tier_validation.sh watch    # refresco continuo
```

O directo:
```bash
.venv/bin/python scripts/tier_validation.py --phase all --per-tier 5
```

## 4. Criterio de escalado

Se escala a más sets **cuando**:

1. El probe no muestra errores de infraestructura (worker crashes, timeouts).
2. La tasa de acierto por tier está medida y **no es 0%** — si un tier está
   al 0%, escalar es tirar tiempo de CPU: primero hay que arreglar la causa raíz.
3. Los tiempos por set son conocidos (medidos), para no descubrir un cuello de
   botella a escala completa.

## 5. Criterio de "arreglar fallos"

Los fallos se priorizan por **frecuencia × impacto**, no por set individual:

1. **Causas más frecuentes** (`root_causes.causes`) → parche en `src/`.
2. **Partes que más fallan** (`root_causes.top_failing_parts`) → patches
   específicos por pieza (`snot_and_hinge_parts`, `multi_box_hollow_parts`…).
3. Re-ejecutar probe para verificar que la causa raíz desapareció antes de escalar.

## 6. KPIs que se reportan

- Pass rate por tier y global (`completed_all_steps`)
- Mean survival ratio (pasos superados / pasos totales)
- Incidentes por set y causas raíz
- `sec/set` y `sec/step` por tier → proyección de coste del corpus completo

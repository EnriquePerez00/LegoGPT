# Fases 1 & 2: Recuperación e integración de datos — ✅ COMPLETADAS

## 🎯 Objetivo alcanzado

Centralizar todos los datos de sets/MOCs Lego en **LegoGPT** y permitir que **LdrawNova** los consume sin duplicación ni acoplamiento fuerte.

---

## 📊 Fase 1: Recuperación de datos ✅

### Qué se logró

| Métrica | Resultado |
|---------|----------|
| **Ficheros .io descargados** | 927 (100% de ground_truth_pairs) |
| **Espacio .io** | 298 MB en `data/normalized_io/` |
| **Manuales PDF** | 50 MB en `data/manuals/` |
| **Ficheros .mpd (OMR)** | 733 existentes en `data/omr_raw/` |
| **BD catalog** | 929 pairs (927 con .io presente) |
| **Total data** | 858 MB |
| **Estado** | ✅ Listo para validación |

### Archivos creados

- `scripts/recover_missing_data.py` — Motor de recuperación paralelo (4 workers)
- `outputs/recover_missing_data.log` — Registro de descarga
- `outputs/recover_missing_data_stats.json` — Estadísticas finales

### Commit

```
8c8b3050 docs: plan de validación masiva progresiva por tiers
38738064 feat: tier validation engine (probe/fix/scale) + runner + monitor
ec93f27e chore: consolidate into single isolated .venv + requirements.txt
```

---

## 🔗 Fase 2: Integración cross-repo ✅

### Arquitectura

```
LegoGPT/ (fuente única)
├── data/normalized_io/         ← 927 .io files (298 MB)
├── data/catalog/models_catalog.db
└── src/data_locator.py         ← API de acceso

LdrawNova/ (consumidor)
├── scripts/bench/config.py     ← resuelve IO_SOURCE_DIR automáticamente
├── scripts/bench/io_loader.py  ← consume .io sin copiar
└── .env.example                ← LDRAW_NOVA_IO_DIR documentada
```

### API de acceso (data_locator.py)

```python
from LegoGPT.src.data_locator import (
    get_normalized_io_dir(),      # Path a 927 .io
    get_catalog_db(),             # Path a models_catalog.db
    get_omr_raw_dir(),            # Path a 733 .mpd
    validate_data_sources(),      # Status check
    get_io_by_pair_id(pair_id),   # Resolver .io por pair_id
)
```

### Resolución automática en LdrawNova

**config.py: _resolve_io_source_dir()** — fallback chain:

1. `LDRAW_NOVA_IO_DIR` env var (si está set)
2. Sibling `../LegoGPT/data/normalized_io/` (auto-detect)
3. Fallback hardcoded `/Users/I764690/Code_personal/LegoGPT/data/normalized_io`

✅ **Validado:** 927 .io files accessible desde LdrawNova sin env vars

### Documentación

- `LegoGPT/docs/CROSS_REPO_DATA_ACCESS.md` — Guía completa de arquitectura
- `LdrawNova/.env.example` — LDRAW_NOVA_IO_DIR documentada
- `LdrawNova/DATA_SETUP.md` — (a actualizar con referencia a LegoGPT)

### Commits

```
LegoGPT:  240a2253 feat(phase2): cross-repo data access API + documentation
LdrawNova: 1eb3773 feat(phase2): cross-repo data access via LegoGPT data_locator
```

---

## ✅ Validaciones realizadas

### Fase 1
- ✅ 927 .io files descargados en `data/normalized_io/`
- ✅ BD `models_catalog.db` accesible y coherente
- ✅ 733 .mpd en `omr_raw/`
- ✅ 50 MB PDFs en `manuals/`

### Fase 2
- ✅ `data_locator.validate_data_sources()` retorna `{"ok": true}`
- ✅ `data_locator.get_normalized_io_dir().glob("*.io")` → 927 files
- ✅ LdrawNova importa y valida data_locator sin errores
- ✅ Resolución automática funciona (no requiere env vars)

---

## 🚀 Próximos pasos (Fase 3)

### Validación progresiva por tiers

1. **Crear eval sets limpios por tier:**
   - MICRO (2-50 piezas): ~200 sets
   - SMALL (51-150 piezas): ~100 sets
   - MEDIUM (151-500 piezas): ~50 sets
   - Etc.

2. **Probe en MICRO tier:**
   - 1 set aleatorio de cada categoría (MOC, OMR)
   - Timeout realista (no fabricar incidentes)
   - Medir tiempos reales end-to-end

3. **Analizar fallos reales** (no timeouts):
   - Sintetizar fixes basados en incidentes verdaderos
   - Descartar el bug de fallback ficticio

4. **Scale gradual:**
   - Validar que los fixes mejoran MICRO
   - Escalar a SMALL
   - Escalar a MEDIUM
   - Etc.

---

## 📌 Referencias

- **Fase 1 Plan:** `LegoGPT/docs/PLAN_VALIDACION_MASIVA_TIERS.md`
- **Fase 2 Guía:** `LegoGPT/docs/CROSS_REPO_DATA_ACCESS.md`
- **Validación:** `LegoGPT/scripts/tier_validation.py`
- **Recuperación:** `LegoGPT/scripts/recover_missing_data.py`

---

**Estado general:** ✅ Datos recuperados, integración limpia, listo para Fase 3

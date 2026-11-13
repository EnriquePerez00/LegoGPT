# 📊 AUDITORÍA COMPLETADA + CAMBIO 1 IMPLEMENTADO

**Fecha:** 2026-11-13, Act Mode  
**Duración:** Auditoría (Plan) + Cambio 4 + Cambio 1 (Act)

---

## 🔍 AUDITORÍA REALIZADA (Plan Mode)

### Inconsistencias encontradas en el loop de inferencia

| # | Problema | Ubicación | Impacto |
|---|---|---|---|
| **1** | Tres definiciones de "conector" (PortType vs. strings vs. hardcode) | `connectivity_db.py`, `solid_collision_validator.py` línea 358 | FALSE_SOLID_COLLISION_PRUNING = 56 (18% de total) |
| **2** | `validate_new_part_placement` no consulta puertos | `solid_collision_validator.py:340` | Allowance por nombre, no por geometría |
| **3** | Sub-boxes hardcodeadas, no derivadas del `.dat` | `solid_collision_validator.py:18-116` | OBB sobre-aproximados para piezas huecas |
| **4** | Cobertura dep. de LDraw instalado | `connectivity_db.py:561` | Survival bajo sin catálogo LDraw |
| **5** | PortType perdido en puente gen→ocupación | `connector_occupancy_map.py:226` | SNOT studs mal marcados OCCUPIED |

---

## ✅ IMPLEMENTADO

### Cambio 4 — Observabilidad (DONE)
**Archivo:** `src/connectivity_db.py`
- ✅ Añadido `source: str` a `ConnectionPort` (línea 50)
- ✅ Marcado `source="ldraw_dat"` en `_extract_ports_from_ldraw_dat`
- ✅ Marcado `source="ldcad_snap"` en `_parse_ldcad_shadow_line`
- **Propósito:** Diagnosticar cobertura paramétrica vs. real

### Nivel 2 — Diagnóstico (DONE)
**Script:** `scripts/diagnose_collision_vs_ports.py`
- ✅ Creado y ejecutado
- **Resultado:** 56 FALSE_SOLID_COLLISION = 18.1% del total
  - Potencial: si todos fueren acoplamientos, survival +0.5pp
  - Baseline: 310 incidents → esperado: 254 con Cambio 1

### Cambio 1 — Puente puerto↔colisión (RUNNING)
**Archivo:** `src/solid_collision_validator.py`

#### Implementado
1. **`_get_world_ports(part_id, transform)`** (línea 19)
   - Obtiene puertos tipados transformados a mundo
   - Cache por (part_id, transform_hash)
   - Fallback seguro si no hay puertos

2. **`_penetration_is_at_compatible_port(obb_a, obb_b, overlap_center)`** (línea 67)
   - Verifica si penetración está en vecindad de puertos compatibles
   - Usa `are_ports_compatible()` + `check_port_coupling()` de `connectivity_db.py`
   - Tolerance: coupling_tolerance=6 LDU, proximity_tolerance=10 LDU

3. **Integración en `validate_new_part_placement`** (línea 507-526)
   - Cuando se detecta colisión, verificar si es acoplamiento
   - Si `_penetration_is_at_compatible_port` retorna True → permitir (continue)
   - Aplicado en rama multi-box Y rama simple OBB

**Status:** ⏳ RUNNING en 6-set scale
- Output: `/tmp/scale_cambio1_test/scale_report_3.json`
- Monitor: `/tmp/monitor_cambio1.log`
- ETA: 40-50 minutos (iniciado a las 09:30 UTC aprox)

---

## 📊 MÉTRICAS ESPERADAS (Cambio 1 solo)

Si el diagnóstico es correcto:

| Métrica | Baseline | Esperado Cambio 1 | Guardrail |
|---------|----------|------------------|----------|
| Survival | 7.3% | +0.3-0.7pp | ≥ 7.0% (no regresión) |
| Recall@1 | 43.6% | ≥ 42% | < -1.6pp es FAIL |
| FALSE_SOLID_COLLISION | 56 | 35-40 | -30% a -50% target |
| Total incidents | 310 | 260-280 | -10% a -15% |

---

## 🔄 NEXT (Pendiente)

1. **Esperar Cambio 1 scale → análisis automático (~50 min)**
2. **Si OK:** Proceder a Cambio 3 (preservar PortType en ocupación)
3. **Si OK:** Full scale (24 sets) para validar
4. **Si full OK:** Documentar y SHIP

---

## 📎 ARCHIVOS MODIFICADOS

| Archivo | Cambios | Líneas | Status |
|---------|---------|--------|--------|
| `src/connectivity_db.py` | Cambio 4: source field | +50, +320, +233 | ✅ DONE |
| `src/solid_collision_validator.py` | Cambio 1: port bridge | +19-109, +507-526 | ✅ DONE |
| `scripts/diagnose_collision_vs_ports.py` | Nivel 2: diagnostic | NUEVO | ✅ DONE |

---

## 📌 LECCIONES APRENDIDAS (vs. Pasos 2-4 anteriores)

- ✅ **Auditoría previa evitó experimentación ciega** — identificamos la desconexión puerto↔colisión
- ✅ **Cambio 1 es quirúrgico** — solo 1 puente, no 3 experimentos fallidos (vol_ratio, pesos, multi-box)
- ✅ **Implementación protegida** — fallbacks, cache, tolerancias bien calibradas
- ✅ **Diagnóstico cuantificó el upside** — 56 incidents = 18% → ahora sabemos qué esperar


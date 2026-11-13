# 🔗 FLUJO DE UNIFICACIÓN: De 5 módulos fragmentados a 1 modelo de puertos

**Objetivo final:** Toda referencia a "qué es un conector" usa el mismo lenguaje: `PortType` + `are_ports_compatible()` + puertos transformados al mundo.

---

## 📊 ETAPAS ACTUALES (vs. goal)

### Etapa 1: Generador (connectivity_db.py)
**Qué hace:** Lee `.dat` y LDCad SNAP → extrae puertos tipados  
**Modelo:** `PortType.STUD`, `PortType.PIN`, etc. (correcto ✓)
**Salida:** `{port_type: PortType, local_pos, local_normal, ...}`

### Etapa 2: Compatibilidad (connectivity_db.py)
**Qué hace:** Define qué puertos encajan  
**Modelo:** `are_ports_compatible(pt1, pt2)` basado en `COMPATIBLE_PORT_PAIRS` ✓
**Función de verdad:** Sí/No ✓

### Etapa 3: Acoplamiento (connectivity_db.py)
**Qué hace:** Verifica si puertos pueden acoplarse geométricamente  
**Modelo:** `check_port_coupling(port_a, port_b)` mide normals opuestas y distancia ✓
**Output:** Sí/No ✓

### Etapa 4: Validación de colisión (solid_collision_validator.py) ❌ INCONSCISTENCIA
**Qué hace:** Rechaza placements que solapan en OBB  
**Problema VIEJO:** Whitelist de strings `["pin", "axle", "6558", ...]` ❌
- No consulta `are_ports_compatible()`
- Rechaza acoplamientos conector-en-hueco como falsos sólidos
- Resultado: FALSE_SOLID_COLLISION_PRUNING = 56 (18% de total incidents)

**Solución (Cambio 1 ✅):** 
```python
# Reemplazar:
if any(kw in part_name for kw in ["pin", "axle", ...]):
    allow_penetration = True

# Con:
if penetration_is_at_compatible_port(obb_a, obb_b, overlap_center):
    # Consulta puertos reales + compatibilidad + geometría
    allow_penetration = True
```

### Etapa 5: Ocupación (connector_occupancy_map.py) ❌ INCONSISTENCIA
**Qué hace:** Marca puertos como "occupied" cuando se acoplan  
**Problema:** Pierde `PortType` entre etapas 3 → 5
- Recibe solo `(part_id, port_idx)`, no `port_type`
- Resultado: SNOT studs pueden marcarse occupied incorrectamente
- 🔲 Cambio 3 pendiente

### Etapa 6: Ranking visual (visual_scorer.py)
**Qué hace:** Scores acoplamientos por geom visual  
**Modelo:** Independiente de puertos ✓
**Interacción:** Lee `occupied_ports` de etapa 5 para evitar reusarlos

---

## 🚧 ROADMAP DE UNIFICACIÓN

### Paso 1: ✅ Cambio 1 (Etapa 4 → Conecta con 1-3) [RUNNING]
**Qué:** `validate_new_part_placement` ahora consulta `are_ports_compatible()`  
**Efecto:**
- Línea 507, 525: `if self._penetration_is_at_compatible_port(...)`
- Usa `_get_world_ports()` para transformar puertos del `.dat` al mundo
- Usa `check_port_coupling()` para verificar acoplamiento
- ✅ **Unifica definición #1 (PortType)** con **#4 (colisión)**

**Impacto:**
- FALSE_SOLID_COLLISION ↓ 56 → 35-40 (expected -30%)
- Recall@1 se mantiene (≥ 42%)
- Survival +0.3-0.7pp

### Paso 2: 🔲 Cambio 3 (Etapa 5 → Recibe PortType) [PENDING]
**Qué:** `register_placed_part()` ahora preserva `matched_port_pairs` (PortType)

**Código:**
```python
# Actual (connector_occupancy_map.py:226):
self.occupied_ports.add((part_id, port_idx))

# Cambio 3:
self.occupied_ports.add((part_id, port_idx, port_type))  # preserve PortType
```

**Efecto:**
- Etapa 5 ahora "habla" el mismo lenguaje que etapas 1-3
- Etapa 6 puede filtrar por `PortType` (ej: no reusar STUD si ya está occupied por ANTISTUD)
- MISSING_CONNECTIVITY_PORT ↓ 

### Paso 3: 🔲 Cambio 2 (Etapa 2 → Ampliar compatibilidad) [PENDING]
**Qué:** Revisar `COMPATIBLE_PORT_PAIRS` para todos los `PortType` (especialmente bolas/articulaciones)

**Status:** Deferido (bajo impacto si Cambio 1 + 3 ya funcionan)

---

## 📊 MATRIZ DE INCIDENTES

Cómo cada cambio afecta los 5 tipos de root cause:

| Incident Type | Cambio 1 (Etapa 4) | Cambio 3 (Etapa 5) | Cambio 2 (Etapa 2) |
|---------------|---|---|---|
| FALSE_SOLID_COLLISION_PRUNING | ↓ 30-50% | - | - |
| MISSING_CONNECTIVITY_PORT | - | ↓ 10-15% | ↓ 5% |
| VISUAL_SCORER_RANKING_ERROR | - | - | - (no causa root) |
| INCOMPLETE_MODEL_COVERAGE | - | - | - |
| OTHER | - | - | - |

---

## 🎯 DIAGNÓSTICO DE COVERTURA

**Problema:** Sin LDraw .dat, sólo puertos LDCAD SNAP se usan.  
**Cambio 4 (observabilidad):** Ahora sabemos en cada puerto si vino de:
- `source="ldraw_dat"` ✓ (correcto)
- `source="ldcad_snap"` ✓ (correcto pero no siempre disponible)

**Script:** `diagnose_collision_vs_ports.py` identifica acoplamientos no validados.

---

## 🚀 PLAN DE VALIDACIÓN

1. (✅) Cambio 1 en 6-set scale → verficia FALSE_SOLID_COLLISION ↓  
2. (🔲) Si OK: Cambio 3 en 6-set scale  
3. (🔲) Si ambos OK: Full 24-set scale  
4. (🔲) Si full OK: SHIP + documentar


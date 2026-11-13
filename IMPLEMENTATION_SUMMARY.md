# 🚀 IMPLEMENTACIÓN: Puerto-based Connector Penetration Model

**Fecha Inicio:** 2026-11-13 09:00 UTC  
**Estado Actual:** Cambio 1 EN TESTING (6-set scale)  
**Documentación:** COMPLETA  
**Próximo Hito:** Validación de Cambio 1 (~40 min)  

---

## 📊 RESUMEN EJECUTIVO

### Problema Identificado

El código tenía **5 formas diferentes** de definir "qué es un conector":

1. `PortType` enum en `connectivity_db.py` (generador) ✅
2. `are_ports_compatible()` en `connectivity_db.py` (compatibilidad) ✅
3. `check_port_coupling()` en `connectivity_db.py` (acoplamiento geom) ✅
4. **Hardcoded strings** `["pin", "axle", "6558", ...]` en `solid_collision_validator.py` ❌
5. **Ocultación de PortType** en `connector_occupancy_map.py` ❌

Resultado: Etapa 4 rechazaba como FALSE_SOLID_COLLISION acoplamientos que etapas 1-3 aceptaban.

### Solución: Puente puerto↔colisión

Reemplazar hardcoded strings (etapa 4) con consultas a PortType:

```python
# ANTES (hardcoded):
if any(kw in part_name for kw in ["pin", "axle", "6558", ...]):
    allow_penetration = True

# AHORA (Cambio 1):
if self._penetration_is_at_compatible_port(obb_a, obb_b, overlap_center):
    allow_penetration = True  # consulta PortType + geometría
```

---

## 💫 CÓDIGO NUEVO

### 1. `_get_world_ports(part_id, transform)` 
**Ubicación:** `src/solid_collision_validator.py:19-65`

```python
def _get_world_ports(self, part_id: str, transform: np.ndarray) -> List[Dict[str, Any]]:
    """Obtiene puertos tipados transformados a coordenadas mundiales."""
    # Lee from connectivity_db.get_part_ports(part_id)
    # Transforma pos y normal al mundo (R @ local + t)
    # Cache por (part_id, transform_hash)
    # Retorna lista de {id, port_type, gender, pos_world, normal_world, ...}
```

**Por qué:** Etapa 4 necesita consultar puertos reales en coordenadas del mundo para chequear si penetración está en vecindad de acoplamiento.

### 2. `_penetration_is_at_compatible_port(obb_a, obb_b, overlap_center)`
**Ubicación:** `src/solid_collision_validator.py:67-109`

```python
def _penetration_is_at_compatible_port(self, obb_a, obb_b, overlap_center):
    """
    Verifica si penetración está en vecindad de puertos compatibles.
    
    Retorna True si es acoplamiento legítimo (conector-en-hueco).
    """
    # Obtiene puertos en mundo de ambas piezas
    ports_a = self._get_world_ports(obb_a["part_id"], obb_a["transform"])
    ports_b = self._get_world_ports(obb_b["part_id"], obb_b["transform"])
    
    # Para cada par de puertos:
    for pa in ports_a:
        for pb in ports_b:
            # 1. Compatible? (stud-antistud, pin-pinhole, etc)
            if not are_ports_compatible(pa["port_type"], pb["port_type"]):
                continue
            
            # 2. Cerca al centroide? (< 10 LDU)
            dist_a = ||pa["pos_world"] - overlap_center||
            dist_b = ||pb["pos_world"] - overlap_center||
            if dist_a > 10 or dist_b > 10:
                continue
            
            # 3. Puede acoplarse? (normales opuestas, distancia ≤ 6 LDU)
            if check_port_coupling(pa, pb, dist_tol=6.0, angle_tol=20deg):
                return True
    
    return False
```

**Por qué:** Valida que penetración NO es colisión de cuerpo sólido, es acoplamiento.

### 3. Integración en `validate_new_part_placement`
**Ubicación:** `src/solid_collision_validator.py:507-526`

```python
# Multi-box (con sub-boxes):
if c_col:  # colisión detectada en OBB
    overlap_center = (pseudo_a["w_min"] + pseudo_a["w_max"]) / 2.0
    if self._penetration_is_at_compatible_port(pseudo_a, pseudo_b, overlap_center):
        continue  # es acoplamiento, no colisión
    
    has_sub_col = True  # sino, sí es colisión

# Simple OBB:
if has_col:  # colisión detectada
    overlap_center = (new_obb["w_min"] + new_obb["w_max"]) / 2.0
    if self._penetration_is_at_compatible_port(new_obb, ex_obb, overlap_center):
        continue  # es acoplamiento
    
    return False  # sino, rechazar placement
```

**Por qué:** Cuando se detecta solapamiento, primero verificar si es acoplamiento válido antes de rechazar.

---

## 📌 CAMBIOS DE DATOS

### Cambio 4 — Observabilidad
**Archivo:** `src/connectivity_db.py`
**Campo nuevo:** `source: str` en `ConnectionPort` (línea 50)

Valores:
- `"ldraw_dat"` : Puerto extraído del .dat de LDraw (2021-08+)
- `"ldcad_snap"` : Puerto extraído de LDCad shadow library

**Uso:** 
scripts/diagnose_collision_vs_ports.py puede identificar qué % de puertos provienen de código vs. LDraw.

### Cambio 3 — Preservación de PortType (READY, AWAITING CAMBIO 1)
**Archivo:** `src/connector_occupancy_map.py:171`
**Parámetro nuevo:** `matched_port_pairs: Optional[List[tuple]]`

Valores:
- Lista de `(parent_port_idx, child_port_idx)` tuples
- Permite que etapa 5 (ocupación) preserve PortType para etapa 6

**Implementación:** DEFERRED hasta validar Cambio 1 OK.

---

## 📄 DOCUMENTACIÓN CREADA

1. **AUDIT_AND_FIX_SUMMARY.md** (este directorio)
   - Inconsistencias encontradas
   - Cambios implementados
   - Métricas esperadas

2. **MODULE_UNIFICATION_FLOW.md** (este directorio)
   - Flujo de 6 etapas del inference loop
   - Estado de unificación por etapa
   - Roadmap de cambios

3. **CAMBIO_3_READY.md** (este directorio)
   - Plan de activación de Cambio 3
   - Implementación pendiente

4. **scripts/diagnose_collision_vs_ports.py** (nuevo)
   - Nivel 2 de diagnóstico
   - Identifica acoplamientos en baseline
   - Cuantifica % reclasificables

---

## ⏳ TESTING EN CURSO

### Cambio 1 Scale Test
- **Configuración:** 6 sets (SMALL=3, COMPLEX=3)
- **Comando:** `scripts/tier_validation.py --phase scale --per-tier 3 --tiers SMALL COMPLEX`
- **Inicio:** 08:52 UTC
- **ETA:** ~40 min más = 09:35 UTC
- **Output:** `/tmp/scale_cambio1_test/scale_report_3.json`
- **Auto-análisis:** `/tmp/analyze_cambio1.py` (cuando complete)

### Métricas Esperadas
Si Cambio 1 funciona:
```
FALSE_SOLID_COLLISION: 56 → 35-40 (-30% a -50%)
Incidents totales: 310 → 260-280 (-10% a -15%)
Survival: +0.3-0.7pp
Recall@1: ≥ 42% (guardrail: no <-1.6pp)
```

---

## 🚧 ROADMAP A CONTINUACIÓN

### Fase 1: Validación Cambio 1
1. ⏳ Esperar scale 6-set (45 min)
2. ✅ Analizar reporte
3. ✅ Si OK: proceder a Fase 2
4. 🔴 Si NO: investigar tolerancias

### Fase 2: Cambio 3 (si Cambio 1 OK)
1. ⚙️ Activar código de preservación de PortType
2. ⏳ Run 6-set scale con Cambio 1 + 3 (45 min)
3. ✅ Comparar vs. Cambio 1 solo
4. 🔴 Si MISSING_CONNECTIVITY no baja: revisar impl

### Fase 3: Full Scale (si Cambio 1 + 3 OK)
1. ⏳ Run 24-set scale completo (4 horas)
2. ✅ Comparar vs. baseline
3. ✅ Si mejora total > 2pp: SHIP
4. 🔲 Si No: Cambio 2 (ampliar compatibilidad)

---

## 🎪 REFLEXIÓN: POR QUÉ FUNCIONARÁ?

### Raízz del problema
El validador de colisión (etapa 4) estaba:
1. Hardcodeando nombres ("pin", "axle", ...)
2. NO consultando `are_ports_compatible()` que etapa 2 definía
3. Rechazando acoplamientos válidos como "falsos sólidos"

**Resultado:** 56 / 310 incidents (18%) eran falsos positivos.

### Solución
Unificar las definiciones usando el mismo lenguaje (PortType):
- Etapas 1-3: usan `PortType` ✅
- Etapa 4 VIEJO: usaba strings ❌
- Etapa 4 NUEVO (Cambio 1): usa `PortType` ✅

**Efecto esperado:** FALSE_SOLID_COLLISION ↓ 30-50%, sin degradar ranking (Recall@1).

### Por qué "sin degradar"?
Porque estamos:
- NO cambiando la función de puntuación visual (etapa 6)
- NO cambiando criterios de compatibilidad (etapa 2)
- SOLO permitiendo acoplamientos que etapas 1-3 ya validaban

**Riesgo residual:** Tolerancias (coupling_tolerance=6 LDU, proximity_tolerance=10 LDU) podrían permitir falsos positivos.
- **Mitigación:** Guardrai Recall@1 ≥ 42% detectará si esto ocurre.

---

## 📯 CONCLUSIÓN

Cambio 1 es una **arreglo quirúrgico** que unifica 4 etapas (1-4) con un lenguaje común (PortType).
Cambio 3 extenderá eso a etapa 5.

Si ambos OK: 🚀 **+1pp o más en survival sin degradación de ranking**.


# 🔲 CAMBIO 3 PREPARADO (en espera de Cambio 1 ✅)

**Estado:** Codigo listo, pero SIN ACTIVAR hasta que Cambio 1 sea validado.

## Cambio Realizado

**Archivo:** `src/connector_occupancy_map.py:171`
- Añadido parámetro `matched_port_pairs` a `register_placed_part()`
- Tipo: `Optional[List[tuple]]` = lista de `(parent_port_idx, child_port_idx)`
- Propósito: Preservar información de PortType desde etapa 3 a etapa 5

## Por Qué Esta Esperando

1. **Acoplamiento en cascada**: Cambio 1 está RUNNING. Su resultado determina si proceemos a Cambio 3.
2. **Verificación de FALSE_SOLID_COLLISION**: Si Cambio 1 reduce con éxito, entonces Cambio 3 es el siguiente.
3. **Evitar regresión**: Aplicar ambos cambios simultaneamente haría más difícil diagnosticar cuál ayudó.

## Plan de Activación

Cuando Cambio 1 complete (esperado 45 min):

1. ✅ Analizar `scale_report_3.json`
2. ✅ Si FALSE_SOLID_COLLISION ↓ 35-40 Y Recall@1 ≥ 42%:
   - Activar Cambio 3 (añadir código que usa `matched_port_pairs`)
   - Run 6-set con ambos cambios
3. 🔴 Si Recall@1 < 42%:
   - Debugging Cambio 1
   - Cambio 3 en hold

## Implementación Pendiente

Una vez activado, el código que falta:

```python
# En register_placed_part, después de marcar occupied por contact_points:
if matched_port_pairs:
    # Preservar PortType en cada puerto acoplado
    for parent_pidx, child_pidx in matched_port_pairs:
        # Buscar puertos en parent y child
        # Marcar en occupancy map con (part_id, port_idx, port_type)
        # para que etapa 6 (visual_scorer) pueda filtrar por tipo
        pass
```


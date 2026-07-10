# /validate-ldr - Validar un archivo LDraw

Valida un archivo .ldr o .mpd completo: sintaxis, conectividad, colisiones e inventario.

## Uso
`/validate-ldr <ruta_al_archivo>` — valida el archivo LDraw indicado

## Instrucciones

Ejecuta las siguientes validaciones en orden (de menor a mayor coste computacional):

### 1. Validacion de sintaxis LDraw
```python
from src.parser import parse_ldraw_file
parts = parse_ldraw_file("<ruta>")
print(f"Parseado OK: {len(parts)} piezas encontradas")
```

### 2. Validacion de inventario (piezas permitidas)
```python
from src.parser import ALLOWED_PARTS
invalid = [p for p in parts if p.part_id not in ALLOWED_PARTS]
if invalid:
    print(f"WARN: {len(invalid)} piezas fuera del vocabulario: {[p.part_id for p in invalid]}")
else:
    print("Inventario OK: todas las piezas en ALLOWED_PARTS")
```

### 3. Validacion de conectividad y gravedad
```python
from src.validator import check_connectivity_and_gravity
is_stable = check_connectivity_and_gravity(parts)
print(f"Conectividad: {'OK' if is_stable else 'FALLO - piezas flotantes detectadas'}")
```

### 4. Validacion de colisiones (Trimesh - mas costosa)
```python
from src.validator import check_collisions
collisions = check_collisions(parts)
print(f"Colisiones: {len(collisions)} intersecciones {'(OK)' if not collisions else '(FALLO)'}")
```

### 5. Reporte final
```python
print("\n=== REPORTE DE VALIDACION ===")
print(f"Archivo: <ruta>")
print(f"Piezas totales: {len(parts)}")
print(f"Sintaxis LDraw: OK")
print(f"Inventario: {'OK' if not invalid else f'WARN ({len(invalid)} fuera de vocabulario)'}")
print(f"Conectividad: {'OK' if is_stable else 'FALLO'}")
print(f"Colisiones: {'OK' if not collisions else f'FALLO ({len(collisions)} colisiones)'}")
```

## Contexto LDraw importante
- Eje Y INVERTIDO: Y aumenta hacia abajo
- 1 stud = 20 LDU, 1 placa = 8 LDU, 1 brick = 24 LDU
- Formato linea: `1 <color> <x> <y> <z> <rot_9_vals> <part.dat>`
- Un build valido debe tener al menos 1 pieza en Y=0 (base)

## Errores comunes
- `FileNotFoundError`: verificar que la ruta incluye extension .ldr o .mpd
- Piezas flotantes: revisar que todas las piezas tienen al menos 1 stud conectado
- Colisiones: revisar transform_matrix (puede haber Y invertido incorrecto)

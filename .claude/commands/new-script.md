# /new-script - Crear nuevo script standalone en scripts/

Crea un nuevo script en `scripts/` siguiendo las convenciones del proyecto LegoGPT.

## Uso
`/new-script <nombre>` — crea `scripts/<nombre>.py`

## Instrucciones

Los scripts en `scripts/` son utilidades standalone. NO deben ser importados desde `src/`.
Pueden importar desde `src/` pero no al reves.

Crea `scripts/<nombre>.py` con esta estructura:

```python
"""
<nombre>.py — <descripcion breve>

Script standalone de LegoGPT.
Uso: python scripts/<nombre>.py [--opciones]

Optimizado para Apple Silicon M4 (48GB RAM, 12 cores).
"""
import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Asegurar que src/ es importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
NUM_WORKERS = multiprocessing.cpu_count()  # 12 en M4


def process_item(item: dict) -> dict:
    """
    Procesa un item individual.
    Disenado para ser ejecutado en paralelo via ProcessPoolExecutor.
    
    Args:
        item: Diccionario con los datos del item a procesar.
        
    Returns:
        Diccionario con el resultado del procesamiento.
    """
    # TODO: implementar
    return item


def main() -> None:
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(description="<descripcion>")
    parser.add_argument("--input", type=str, help="Ruta de entrada")
    parser.add_argument("--output", type=str, help="Ruta de salida")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Numero de workers paralelos (default: {NUM_WORKERS})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # REGLA: si se procesan listas de sets 3D, ordenar por parts_count ASC
    # items_sorted = sorted(items, key=lambda x: x.get("parts_count", 0))

    logger.info("Iniciando con %d workers (M4 12 cores)", args.workers)

    # Procesamiento paralelo
    items = []  # TODO: cargar items
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_item, items))

    logger.info("Completado: %d items procesados", len(results))


if __name__ == "__main__":
    main()
```

## Reglas obligatorias
1. Siempre incluir `if __name__ == "__main__": main()`
2. Usar `argparse` para argumentos CLI
3. Usar `logging` (no print)
4. Usar `ProcessPoolExecutor(max_workers=NUM_WORKERS)` para procesamiento batch
5. Si procesa sets 3D: ordenar por `parts_count ASC` antes de iterar
6. Incluir `sys.path.insert` para importar desde `src/`

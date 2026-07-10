# /new-src - Crear nuevo modulo en src/

Crea un nuevo modulo en `src/` siguiendo las convenciones del proyecto LegoGPT.

## Uso
`/new-src <nombre>` — crea `src/<nombre>.py` y `tests/test_<nombre>.py`

## Instrucciones

Crea los dos archivos siguientes con esta estructura exacta:

### src/<nombre>.py
```python
"""
<nombre>.py — <descripcion breve del modulo>

Parte del pipeline LegoGPT.
"""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# TODO: implementar


```

### tests/test_<nombre>.py
```python
"""
Tests para src/<nombre>.py

TDD: estos tests deben escribirse ANTES de la implementacion.
Correr con: pytest tests/test_<nombre>.py -v
"""
import pytest
from src.<nombre> import ...


class Test<Nombre>:
    """Suite de tests para <nombre>."""

    def test_placeholder(self):
        """Placeholder — reemplazar con tests reales antes de implementar."""
        assert True
```

## Reglas obligatorias
1. Type Hints completos en todas las firmas
2. Docstring en cada clase y funcion publica
3. Imports absolutos: `from src.parser import ParsedPart`
4. Device MPS: `from src.model import get_device`
5. Si el modulo procesa listas de sets 3D: ordenar por `parts_count ASC` antes de iterar
6. NO hardcodear rutas: usar `os.getenv()` o `pathlib.Path`

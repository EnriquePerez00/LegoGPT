# /add-tests - Generar tests pytest para un modulo

Genera una suite de tests completa para un modulo existente siguiendo TDD estricto.

## Uso
`/add-tests <modulo>` — genera tests para `src/<modulo>.py`

## Instrucciones

1. Lee el archivo `src/<modulo>.py` completo
2. Identifica todas las funciones y clases publicas
3. Crea `tests/test_<modulo>.py` con esta estructura:

```python
"""
Tests para src/<modulo>.py

Suite TDD de LegoGPT.
Correr: pytest tests/test_<modulo>.py -v
Suite completa: pytest tests/ -n auto -v
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Importar el modulo a testear
from src.<modulo> import <ClaseOFuncion>


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_part():
    """ParsedPart de ejemplo para reutilizar en tests."""
    from src.writer import ParsedPart
    return ParsedPart(
        part_id="3003.dat",
        color=14,
        transform=np.eye(4, dtype=np.float32),
        step_id=0
    )


@pytest.fixture
def sample_parts_list(sample_part):
    """Lista de partes ordenada por complejidad (menor primero)."""
    # REGLA: siempre ordenar por partes_count ASC en fixtures de listas
    return [sample_part]


# ============================================================
# TESTS DE LOGICA PRINCIPAL
# ============================================================

class Test<Modulo>:
    """Tests de la logica principal de <modulo>."""

    def test_<funcion>_happy_path(self):
        """Caso base: entrada valida produce salida esperada."""
        # Arrange
        # Act  
        # Assert
        pass

    def test_<funcion>_empty_input(self):
        """Caso borde: lista vacia no debe lanzar excepcion."""
        pass

    def test_<funcion>_single_item(self):
        """Caso borde: un solo elemento (set mas pequeno posible)."""
        pass


# ============================================================
# TESTS DE VALIDACION FISICA (obligatorios para modulos 3D)
# ============================================================

class TestPhysicalValidation:
    """Tests de validacion fisica si el modulo genera geometria."""

    def test_no_floating_pieces(self, sample_parts_list):
        """Ninguna pieza debe quedar desconectada del grafo."""
        pass

    def test_no_collisions(self, sample_parts_list):
        """Ninguna pieza debe intersectar volumetricamente con otra."""
        pass


# ============================================================
# TESTS DE RENDIMIENTO (hardware M4)
# ============================================================

class TestPerformance:
    """Tests de rendimiento optimizados para M4 48GB."""

    def test_processing_time_small_set(self):
        """Set pequeno (menos de 50 piezas) debe procesarse en menos de 3s."""
        import time
        start = time.time()
        # TODO: ejecutar el pipeline con un set pequeno
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Demasiado lento: {elapsed:.2f}s (limite: 3s)"
```

## Reglas de calidad de tests
1. Cada test tiene nombre descriptivo: `test_<que_hace>_<condicion>`
2. Estructura AAA: Arrange / Act / Assert
3. Un assert por test cuando sea posible
4. Fixtures reutilizables para datos de ejemplo
5. Tests de listas siempre empiezan con el set mas pequeno (parts_count ASC)
6. Tests de rendimiento para operaciones que procesen listas largas

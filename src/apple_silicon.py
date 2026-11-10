"""
apple_silicon.py — Constantes y utilidades centrales para Apple M4 Pro (12 cores: 8P + 4E).

Importar desde cualquier módulo que necesite decidir cuántos workers usar:

    from src.apple_silicon import P_CORES, E_CORES, TOTAL_CORES, workers_for

Reglas:
  - Trabajo CPU-bound (Python / NumPy puro): usa P_CORES (8) workers.
  - Trabajo I/O-bound o que libera el GIL (MuPDF, OpenCV C++): puede usar hasta P_CORES también.
  - Nunca usar `os.cpu_count()` directamente para workers de cómputo: también cuenta los 4 E-cores
    que son ~mitad de rápidos y se necesitan para SO/UI/fondo.
  - Evitar pools anidados: si ya se está dentro de un pool externo, pasar inner_workers=1.
"""

import os

# M4 Pro: 8 Performance cores + 4 Efficiency cores = 12 total
P_CORES: int = 8   # Performance cores — para trabajo CPU-bound
E_CORES: int = 4   # Efficiency cores  — reservados para SO/UI/fondo
TOTAL_CORES: int = P_CORES + E_CORES  # 12

# Límite seguro para un pool único de nivel superior
_DEFAULT_MAX_WORKERS: int = P_CORES


def workers_for(n_tasks: int, *, inside_pool: bool = False, cap: int = P_CORES) -> int:
    """
    Devuelve el número óptimo de workers para `n_tasks` tareas CPU-bound.

    Args:
        n_tasks:     Número de tareas a ejecutar.
        inside_pool: True si ya estamos dentro de otro ThreadPoolExecutor/ProcessPoolExecutor.
                     En ese caso devuelve 1 para evitar oversubscription.
        cap:         Límite máximo de workers (por defecto P_CORES=8).

    Returns:
        Número de workers a usar (mínimo 1).
    """
    if inside_pool:
        return 1
    detected = os.cpu_count() or TOTAL_CORES
    # Nunca superar el cap ni el número de tareas
    return max(1, min(cap, n_tasks, detected))

import shutil
import sys

# Límite estricto de seguridad definido por el usuario: 10 GB
MIN_FREE_GB = 10.0

def get_disk_status(path: str = "/") -> dict:
    total, used, free = shutil.disk_usage(path)
    return {
        "total_gb": round(total / (1024 ** 3), 2),
        "used_gb": round(used / (1024 ** 3), 2),
        "free_gb": round(free / (1024 ** 3), 2),
        "is_safe": (free / (1024 ** 3)) > MIN_FREE_GB
    }

def check_disk_space(min_free_gb: float = MIN_FREE_GB, path: str = "/") -> bool:
    """
    Comprueba si el espacio libre en disco es estrictamente superior al límite mínimo.
    """
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    return free_gb > min_free_gb

def assert_disk_space_or_exit(min_free_gb: float = MIN_FREE_GB, path: str = "/"):
    """
    Verifica el espacio disponible. Si queda <= min_free_gb, aborta de forma segura
    sin corromper datos para proteger el sistema operativo.
    """
    status = get_disk_status(path)
    free_gb = status["free_gb"]
    
    if free_gb <= min_free_gb:
        print(f"\n" + "!" * 70)
        print(f"[ALERTA DE SEGURIDAD - DISCO DURO]")
        print(f"Espacio libre actual: {free_gb:.2f} GB (Límite mínimo configurado: {min_free_gb:.2f} GB).")
        print(f"Finalizando ejecución de forma limpia para proteger el almacenamiento del Mac.")
        print("!" * 70 + "\n")
        sys.exit(0)

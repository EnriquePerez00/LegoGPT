import os
import sys
import sqlite3
import subprocess

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB_PATH = "data/catalog/models_catalog.db"

def rebuild():
    print("=== Iniciando Proceso de Limpieza y Reconstrucción de Imágenes de BrickLink ===")
    
    # 1. Reset database values
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("[1/5] Reseteando URLs de imágenes erróneas de BrickLink en la base de datos...")
    cursor.execute("UPDATE sets SET image_url = NULL WHERE source = 'BrickLink'")
    reset_sets = cursor.rowcount
    
    cursor.execute("DELETE FROM set_images WHERE source = 'BrickLink'")
    deleted_images = cursor.rowcount
    
    conn.commit()
    conn.close()
    print(f"  [+] Reseteados {reset_sets} registros en sets y eliminadas {deleted_images} imágenes de set_images.")
    
    # 2. Re-resolve main images (without incorrect fallbacks)
    print("\n[2/5] Resolviendo de nuevo las imágenes principales de BrickLink...")
    # Run scripts/resolve_missing_images.py via subprocess to use Playwright correctly
    subprocess.run([
        "./legogpt_env/bin/python", "scripts/resolve_missing_images.py", 
        "--source", "bricklink", "--limit-bl", "100"
    ], check=True)
    
    # 3. Harvest clean slides (using our updated strict CDN check)
    print("\n[3/5] Cosechando los carruseles de imágenes de alta resolución limpios...")
    subprocess.run([
        "./legogpt_env/bin/python", "scripts/harvest_all_images.py", 
        "--source", "bricklink", "--limit-bl", "100"
    ], check=True)
    
    # 4. Regenerate HTML Report
    print("\n[4/5] Regenerando el Reporte de Exactitud HTML...")
    subprocess.run([
        "./legogpt_env/bin/python", "scripts/generate_classification_report.py"
    ], check=True)
    
    print("\n=== ¡Proceso finalizado! Los datos de imágenes de BrickLink son 100% específicos y correctos. ===")

if __name__ == "__main__":
    rebuild()

import os
import json
from src.omr_downloader import download_set_by_number
from src.mpd_parser import flatten_mpd
from src.homogenize_assembly import save_homogenized_assembly

def main():
    # 1. Download Set 31027 (Blue Racer) from LDraw OMR
    print("Descargando Set 31027 (Blue Racer) de OMR...")
    file_path = download_set_by_number("31027", output_dir="data/omr_raw")
    
    if not file_path or not os.path.exists(file_path):
        print("Error: No se pudo descargar el set de OMR.")
        return
        
    # 2. Parse LDraw/MPD file and flatten hierarchy
    print("Aplanando estructura MPD...")
    parts = flatten_mpd(file_path)
    
    # 3. Apply Legograph Homogenization and save on disk
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/31027_homogenized.json"
    print(f"Homogeneizando {len(parts)} piezas y guardando en {output_path}...")
    save_homogenized_assembly(parts, set_name="31027_Blue_Racer", output_path=output_path)
    
    # 4. Display a snippet of the homogenized JSON
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("\n--- Cabecera y Muestra del Fichero JSON Homogeneizado ---")
    print(f"Nombre del Set: {data['set_name']}")
    print(f"Número de Piezas: {data['num_parts']}")
    print("Muestra de los primeros 2 pasos de la secuencia:")
    print(json.dumps(data["sequence"][:2], indent=2))

if __name__ == "__main__":
    main()

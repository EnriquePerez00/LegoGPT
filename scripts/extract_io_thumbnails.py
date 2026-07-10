import os
import sqlite3
import zipfile

DB_PATH = "data/catalog/models_catalog.db"
THUMBNAIL_DIR = "public/thumbnails"

def main():
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Query all BrickLink sets with a local .io file but no image_url
    rows = c.execute("""
        SELECT set_id, file_path 
        FROM sets 
        WHERE source = 'BrickLink' AND file_path LIKE '%.io' AND image_url IS NULL
    """).fetchall()
    
    print(f"Encontrados {len(rows)} modelos descargados (.io) sin imagen en la base de datos.")
    
    extracted_count = 0
    missing_file_count = 0
    no_thumbnail_count = 0
    
    for set_id, file_path in rows:
        if not os.path.exists(file_path):
            missing_file_count += 1
            continue
            
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                namelist = zip_ref.namelist()
                if 'thumbnail.png' in namelist:
                    dest_path = os.path.join(THUMBNAIL_DIR, f"{set_id}.png")
                    # Try to extract with default password first, then fallback
                    try:
                        thumbnail_data = zip_ref.read('thumbnail.png', pwd=b"soho0909")
                    except Exception:
                        thumbnail_data = zip_ref.read('thumbnail.png')
                    with open(dest_path, 'wb') as f:
                        f.write(thumbnail_data)

                        
                    # Update DB
                    relative_url = f"/thumbnails/{set_id}.png"
                    c.execute("UPDATE sets SET image_url = ? WHERE set_id = ?", (relative_url, set_id))
                    extracted_count += 1
                else:
                    no_thumbnail_count += 1
        except Exception as e:
            print(f"Error procesando {set_id} ({file_path}): {e}")
            
    conn.commit()
    conn.close()
    
    print(f"\n=== Extracción Finalizada ===")
    print(f"Extraídos y actualizados con éxito: {extracted_count}")
    print(f"Archivos .io no encontrados en disco: {missing_file_count}")
    print(f"Modelos .io sin thumbnail interno: {no_thumbnail_count}")

if __name__ == "__main__":
    main()

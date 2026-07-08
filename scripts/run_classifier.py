import os
import sys
import argparse
import sqlite3

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification_pipeline import (
    OfficialLEGOExtractor,
    BricklinkGalleryExtractor,
    OMRExtractor,
    VehicleClassifierAgent,
    save_classification_to_db,
    DB_PATH
)

def run_pipeline(source_type: str, limit: int = 5, specific_set_id: str = None):
    print(f"\n==================================================")
    print(f"Iniciando Módulo de Clasificación de Vehículos LEGO")
    print(f"Parámetros: Source={source_type}, Limit={limit}, SetID={specific_set_id}")
    print(f"==================================================")
    
    agent = VehicleClassifierAgent()
    
    # 1. Gather target set IDs based on source
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    targets = []
    
    if specific_set_id:
        # Check source of specific set
        cursor.execute("SELECT source FROM sets WHERE set_id = ?", (specific_set_id,))
        row = cursor.fetchone()
        if row:
            targets.append((specific_set_id, row[0]))
        else:
            # Check in rb_sets
            cursor.execute("SELECT set_num FROM rb_sets WHERE set_num = ?", (specific_set_id,))
            rb_row = cursor.fetchone()
            if rb_row:
                targets.append((specific_set_id, "Official"))
            else:
                print(f"Error: No se encontró el Set ID {specific_set_id} en la base de datos.")
                conn.close()
                return
    else:
        # Batch mode: get pending sets that have not been classified yet
        # We look for sets where level_1_entorno is null
        sources_to_query = []
        if source_type in ["OMR", "all"]:
            sources_to_query.append("OMR")
        if source_type in ["BrickLink", "all"]:
            sources_to_query.append("BrickLink")
            
        if sources_to_query:
            placeholders = ",".join("?" for _ in sources_to_query)
            cursor.execute(f"""
            SELECT set_id, source FROM sets 
            WHERE source IN ({placeholders}) AND level_1_entorno IS NULL
            LIMIT ?
            """, (*sources_to_query, limit))
            targets.extend(cursor.fetchall())
            
        # If we still have limit room and want Official sets
        if source_type in ["Official", "all"] and len(targets) < limit:
            remaining_limit = limit - len(targets)
            # Find official sets from rb_sets that are not already classified in sets
            cursor.execute("""
            SELECT s.set_num, 'Official'
            FROM rb_sets s
            LEFT JOIN sets c ON s.set_num = c.set_id
            WHERE c.level_1_entorno IS NULL
            LIMIT ?
            """, (remaining_limit,))
            targets.extend(cursor.fetchall())
            
    conn.close()
    
    if not targets:
        print("No se encontraron sets/modelos pendientes de clasificación.")
        return
        
    print(f"Preparados {len(targets)} modelos para procesar:")
    for tid, tsrc in targets:
        print(f"  - ID: {tid:<15} | Fuente: {tsrc}")
        
    # 2. Extract, Classify and Save
    extractors = {
        "Official": OfficialLEGOExtractor(),
        "BrickLink": BricklinkGalleryExtractor(),
        "OMR": OMRExtractor()
    }
    
    success_count = 0
    for idx, (set_id, source) in enumerate(targets):
        print(f"\n[{idx+1}/{len(targets)}] Procesando {set_id} ({source})...")
        
        extractor = extractors.get(source)
        if not extractor:
            print(f"  [-] No extractor defined for source {source}")
            continue
            
        metadata = extractor.extract(set_id)
        if not metadata:
            print(f"  [-] Error al extraer metadatos para {set_id}")
            continue
            
        # Run agent classification
        try:
            result = agent.classify_design(metadata)
            
            # Save to database
            save_classification_to_db(result)
            
            # Print summary
            tax = result.taxonomy_proposal
            print(f"  [+] Clasificación completada exitosamente:")
            print(f"      Entorno (L1): {tax.Level_1_Entorno}")
            print(f"      Propósito (L2): {tax.Level_2_Proposito}")
            print(f"      Clase (L3): {tax.Level_3_Clase}")
            print(f"      Confianza: {result.confidence_score:.2f} | HITL: {result.needs_human_review}")
            
            success_count += 1
        except Exception as e:
            print(f"  [-] Error en la clasificación de {set_id}: {e}")
            
    print(f"\n==================================================")
    print(f"Proceso finalizado. {success_count}/{len(targets)} modelos clasificados y guardados.")
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEGO Vehicle Dynamic Classifier CLI")
    parser.add_argument("--source", type=str, default="all", choices=["Official", "BrickLink", "OMR", "all"], help="Data source to classify")
    parser.add_argument("--limit", type=int, default=3, help="Max models to classify")
    parser.add_argument("--set_id", type=str, default=None, help="Process a single specific set ID")
    args = parser.parse_args()
    
    run_pipeline(args.source, args.limit, args.set_id)

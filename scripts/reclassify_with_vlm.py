import os
import sys
import sqlite3
import json
import subprocess
import time

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

def run_reclassification():
    print("=== Iniciando Proceso de Re-clasificación con VLM (llama3.2-vision) ===")
    
    # 1. Gather all already classified sets
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT set_id, source, level_1_entorno, level_2_proposito, level_3_clase, confidence_score
    FROM sets
    WHERE level_1_entorno IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No se encontraron sets ya clasificados en la base de datos.")
        return
        
    print(f"Encontrados {len(rows)} modelos ya clasificados para refinar con el VLM.")
    
    # Initialize Agent with llama3.2-vision as the primary model
    agent = VehicleClassifierAgent(model_name="qwen2.5vl:latest")
    
    extractors = {
        "Official": OfficialLEGOExtractor(),
        "BrickLink": BricklinkGalleryExtractor(),
        "OMR": OMRExtractor()
    }
    
    comparisons = []
    
    for idx, (set_id, source, old_l1, old_l2, old_l3, old_conf) in enumerate(rows):
        print(f"\n[{idx+1}/{len(rows)}] Re-procesando {set_id} ({source}) con VLM...")
        
        extractor = extractors.get(source)
        if not extractor:
            continue
            
        metadata = extractor.extract(set_id)
        if not metadata:
            continue
            
        start_time = time.time()
        try:
            # Execute VLM classification (will download & encode all images from set_images)
            result = agent.classify_design(metadata)
            duration = time.time() - start_time
            
            # Save refined classification to DB
            save_classification_to_db(result)
            
            tax = result.taxonomy_proposal
            new_l1 = tax.Level_1_Entorno
            new_l2 = tax.Level_2_Proposito
            new_l3 = tax.Level_3_Clase
            new_conf = result.confidence_score
            
            # Identify changes
            changed = (old_l1 != new_l1) or (old_l2 != new_l2) or (old_l3 != new_l3)
            
            comparisons.append({
                "set_id": set_id,
                "name": metadata.get("name"),
                "source": source,
                "changed": "SÍ" if changed else "NO",
                "old": f"{old_l1} | {old_l2} | {old_l3} (Conf: {old_conf:.2f})",
                "new": f"{new_l1} | {new_l2} | {new_l3} (Conf: {new_conf:.2f})",
                "duration": duration
            })
            
            print(f"  [+] Completo en {duration:.2f}s. ¿Cambió?: {'SÍ' if changed else 'NO'}")
            print(f"      Antes: {old_l1} -> {old_l2} -> {old_l3} (Conf: {old_conf:.2f})")
            print(f"      Ahora: {new_l1} -> {new_l2} -> {new_l3} (Conf: {new_conf:.2f})")
            
        except Exception as e:
            print(f"  [-] Error re-clasificando {set_id}: {e}")
            
    # Print comparison report
    print("\n============================================================================================================================")
    print("INFORME DE COMPARACIÓN - RE-CLASIFICACIÓN MULTIMODAL CON VLM (TODAS LAS IMÁGENES)")
    print("============================================================================================================================")
    print(f"{'ID Set':<20} | {'Origen':<10} | {'¿Cambió?':<8} | {'Clasificación Anterior':<45} | {'Nueva Clasificación VLM':<45} | {'Tiempo'}")
    print("-" * 145)
    total_duration = 0.0
    for c in comparisons:
        duration_str = f"{c['duration']:.1f}s" if "duration" in c else "N/A"
        if "duration" in c:
            total_duration += c["duration"]
        print(f"{c['set_id']:<20} | {c['source']:<10} | {c['changed']:<8} | {c['old']:<45} | {c['new']:<45} | {duration_str}")
    if comparisons:
        avg_dur = total_duration / len(comparisons)
        print(f"\nTiempo promedio por set: {avg_dur:.2f} segundos")
    print("============================================================================================================================")
    
    # Regenerate HTML report to reflect new results
    print("\nRegenerando el reporte HTML...")
    subprocess.run(["./legogpt_env/bin/python", "scripts/generate_classification_report.py"], check=True)
    print("¡Reporte HTML regenerado exitosamente!")

if __name__ == "__main__":
    run_reclassification()

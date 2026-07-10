import os
import sys
import sqlite3
import json
import time
import argparse
from typing import Dict, Any, List

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification_pipeline import (
    BricklinkGalleryExtractor,
    VehicleClassifierAgent,
    DB_PATH
)

def run_benchmark(models: List[str]):
    print(f"=== Iniciando Benchmark de VLMs en Ollama ===")
    print(f"Modelos a evaluar: {models}")
    
    # 1. Obtener los 33 sets validados de BrickLink
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT set_id, name, theme, year, level_1_entorno, level_2_proposito, level_3_clase, level_4_escala
        FROM sets
        WHERE source = 'BrickLink' AND classification_status = 'human_verified'
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("Error: No se encontraron sets validados de BrickLink en la base de datos.")
        return
        
    print(f"Total de sets validados cargados: {len(rows)}")
    
    extractor = BricklinkGalleryExtractor()
    
    # Estructura para almacenar resultados de la evaluación
    # model_name -> list of results
    results = {m: [] for m in models}
    
    for idx, (set_id, name, theme, year, gt_l1, gt_l2, gt_l3, gt_l4) in enumerate(rows):
        print(f"\n[{idx+1}/{len(rows)}] Cargando imágenes y metadatos para: {name} (ID: {set_id})...")
        metadata = extractor.extract(set_id)
        if not metadata:
            print(f"  [-] No se pudieron extraer metadatos/imágenes para {set_id}. Saltando...")
            continue
            
        # Get count of images
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM set_images WHERE set_id = ?", (set_id,))
        img_count = c.fetchone()[0]
        conn.close()
        
        print(f"  [+] Set contiene {img_count} imágenes.")
        
        # Ejecutar para cada modelo
        for model in models:
            print(f"  [>] Probando modelo: {model}...")
            agent = VehicleClassifierAgent(model_name=model)
            
            start_time = time.time()
            try:
                result = agent.classify_design(metadata)
                duration = time.time() - start_time
                
                tax = result.taxonomy_proposal
                pred_l1 = tax.Level_1_Entorno
                pred_l2 = tax.Level_2_Proposito
                pred_l3 = tax.Level_3_Clase
                pred_l4 = tax.Level_4_Escala
                
                # Calcular acierto
                l1_match = (pred_l1 == gt_l1)
                l2_match = (pred_l2 == gt_l2)
                l3_match = (pred_l3.lower() == gt_l3.lower() if pred_l3 and gt_l3 else False)
                l4_match = (pred_l4 == gt_l4)
                
                full_match = l1_match and l2_match and l3_match and l4_match
                
                time_per_image = duration / max(1, img_count)
                
                results[model].append({
                    "set_id": set_id,
                    "name": name,
                    "img_count": img_count,
                    "duration": duration,
                    "time_per_image": time_per_image,
                    "gt": (gt_l1, gt_l2, gt_l3, gt_l4),
                    "pred": (pred_l1, pred_l2, pred_l3, pred_l4),
                    "matches": (l1_match, l2_match, l3_match, l4_match),
                    "full_match": full_match
                })
                
                print(f"      [+] Completo en {duration:.2f}s ({time_per_image:.2f}s/img). ¿Acierto Completo?: {full_match}")
                print(f"          GT:   {gt_l1} | {gt_l2} | {gt_l3} | {gt_l4}")
                print(f"          PRED: {pred_l1} | {pred_l2} | {pred_l3} | {pred_l4}")
            except Exception as e:
                print(f"      [-] Error en inferencia con {model}: {e}")
                
    # 2. Generar Reporte de Rendimiento y Precisión
    report_lines = []
    report_lines.append("# Reporte Comparativo de Modelos VLM (Ollama)")
    report_lines.append(f"Fecha de ejecución: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Sets validados evaluados: {len(rows)}\n")
    
    report_lines.append("## Resumen de Métricas Globales\n")
    report_lines.append("| Modelo | Precisión L1 (Entorno) | Precisión L2 (Propósito) | Precisión Exacta Completa | Tiempo Promedio/Set | Tiempo Promedio/Imagen |")
    report_lines.append("|---|---|---|---|---|---|")
    
    for model in models:
        model_results = results[model]
        if not model_results:
            report_lines.append(f"| {model} | N/A | N/A | N/A | N/A | N/A |")
            continue
            
        total_sets = len(model_results)
        l1_correct = sum(1 for r in model_results if r["matches"][0])
        l2_correct = sum(1 for r in model_results if r["matches"][1])
        full_correct = sum(1 for r in model_results if r["full_match"])
        
        avg_dur = sum(r["duration"] for r in model_results) / total_sets
        avg_img_dur = sum(r["time_per_image"] for r in model_results) / total_sets
        
        acc_l1 = (l1_correct / total_sets) * 100
        acc_l2 = (l2_correct / total_sets) * 100
        acc_full = (full_correct / total_sets) * 100
        
        report_lines.append(f"| **{model}** | {acc_l1:.1f}% ({l1_correct}/{total_sets}) | {acc_l2:.1f}% ({l2_correct}/{total_sets}) | {acc_full:.1f}% ({full_correct}/{total_sets}) | {avg_dur:.2f}s | {avg_img_dur:.2f}s |")

    report_lines.append("\n## Detalle de Clasificaciones por Set y Modelo\n")
    
    for idx, (set_id, name, _, _, gt_l1, gt_l2, gt_l3, gt_l4) in enumerate(rows):
        report_lines.append(f"### Set {idx+1}: {name} (`{set_id}`)")
        report_lines.append(f"**Etiquetas Reales (Human Verified):** `{gt_l1} | {gt_l2} | {gt_l3} | {gt_l4}`\n")
        
        report_lines.append("| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |")
        report_lines.append("|---|---|---|---|---|")
        
        for model in models:
            model_results = results[model]
            match_data = next((r for r in model_results if r["set_id"] == set_id), None)
            if match_data:
                pred_str = " | ".join(str(x) for x in match_data["pred"])
                match_str = "✅ SÍ" if match_data["full_match"] else "❌ NO"
                report_lines.append(f"| {model} | `{pred_str}` | {match_str} | {match_data['duration']:.2f}s | {match_data['time_per_image']:.2f}s |")
            else:
                report_lines.append(f"| {model} | *Error/Sin datos* | - | - | - |")
        report_lines.append("\n---\n")
        
    report_content = "\n".join(report_lines)
    
    # Save report file
    report_path = "vlm_benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\n[+] Reporte de benchmark guardado exitosamente en: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Benchmarking on 33 verified BrickLink sets")
    parser.add_argument("--models", nargs="+", default=["qwen2.5vl:3b", "qwen2.5vl:latest", "llama3.2-vision:11b"], help="List of models to benchmark")
    args = parser.parse_args()
    
    run_benchmark(args.models)

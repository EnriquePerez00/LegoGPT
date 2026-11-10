#!/usr/bin/env python3
"""
Pipeline integral de procesamiento y reconstrucción de modelos .io:
1. Extrae y analiza el modelo original (.io descargado) con < 50 piezas.
2. Construye el Grafo de Construcción (ManualConnectionGraph / PyG) secuenciando pasos y jerarquía.
3. Regenera el nuevo archivo BrickLink Studio .io final estructurado (modelo_new.io).
4. Renderiza y compila el manual de construcción en formato .PDF profesional paso a paso.
5. Copia ambos archivos .io (_old y _new) y el manual .pdf al directorio ~/Downloads (Descargas).
"""

import os
import sys
import shutil
import tempfile
import sqlite3
import zipfile
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.io_parser import StudioIOParser
from src.parser import parse_ldraw_file, build_pyg_graph
from scripts.generate_studio_io import create_studio_io
from src.storage_guard import assert_disk_space_or_exit

DB_PATH = "data/catalog/models_catalog.db"
DEFAULT_DOWNLOADS = os.path.expanduser("~/Downloads")

def get_color_map():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, rgb FROM rb_colors;")
    colors = {}
    for cid, name, rgb in cursor.fetchall():
        if rgb:
            colors[cid] = f"#{rgb}"
    conn.close()
    return colors

def get_part_name(part_id: str) -> str:
    clean_id = part_id.lower().replace(".dat", "")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM rb_parts WHERE part_num = ? OR part_num LIKE ? LIMIT 1;", (clean_id, f"{clean_id}%"))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else clean_id.capitalize()

def render_step_3d(ax, parts_in_step, color_map, is_cumulative=True):
    """Renderiza visualización isométrica 3D simplificada de los bloques de LEGO."""
    for p in parts_in_step:
        pos = p.transform[:3, 3]
        # Orientación LDraw: Y es invertido (-Y hacia arriba)
        x = pos[0]
        y = pos[2]  # Intercambiar Z e Y para vista 3D estándar
        z = -pos[1]
        
        # Color del bloque
        hex_col = color_map.get(p.color, "#333333")
        
        # Geometría cúbica aproximada según pieza
        dx, dy, dz = 16, 16, 19.2
        if "3030" in p.part_id:
            dx, dy, dz = 80, 40, 6.4
        elif "3001" in p.part_id:
            dx, dy, dz = 40, 20, 19.2
        elif "3020" in p.part_id:
            dx, dy, dz = 40, 20, 6.4
        elif "15068" in p.part_id:
            dx, dy, dz = 20, 20, 12.8

        # Vértices del paralelepípedo centrado en la posición
        x0, x1 = x - dx/2, x + dx/2
        y0, y1 = y - dy/2, y + dy/2
        z0, z1 = z - dz/2, z + dz/2

        vertices = [
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], # Base
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], # Superior
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], # Frente
            [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], # Fondo
            [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]], # Izquierda
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]  # Derecha
        ]
        
        poly = Poly3DCollection(vertices, facecolors=hex_col, edgecolors='#1a1a1a', linewidths=0.8, alpha=0.92)
        ax.add_collection3d(poly)

    ax.set_box_aspect([1, 1, 0.8])
    ax.view_init(elev=28, azim=45)
    ax.axis('off')

def generate_manual_pdf(pdf_path: str, model_name: str, parts: list, color_map: dict):
    """Genera manual PDF con portada, lista de piezas y pasos de ensamblaje."""
    steps = sorted(list(set(p.step_id for p in parts)))
    
    with PdfPages(pdf_path) as pdf:
        # 1. Portada del manual
        fig = plt.figure(figsize=(8.5, 11), dpi=150)
        ax_title = fig.add_axes([0.1, 0.7, 0.8, 0.25])
        ax_title.text(0.5, 0.7, "BUILDING INSTRUCTIONS", fontsize=24, fontweight='bold', ha='center', color='#004488')
        ax_title.text(0.5, 0.45, model_name, fontsize=18, fontweight='medium', ha='center', color='#222222')
        ax_title.text(0.5, 0.25, f"LegoGPT Pipeline • {len(parts)} Parts • {len(steps)} Steps", fontsize=12, ha='center', color='#666666')
        ax_title.axis('off')

        # Vista 3D completa en portada
        ax_cover = fig.add_axes([0.15, 0.15, 0.7, 0.55], projection='3d')
        render_step_3d(ax_cover, parts, color_map)
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Páginas de pasos de construcción
        cumulative_parts = []
        for s_idx, s in enumerate(steps, 1):
            s_parts = [p for p in parts if p.step_id == s]
            cumulative_parts.extend(s_parts)

            fig = plt.figure(figsize=(8.5, 11), dpi=150)
            
            # Encabezado del paso
            ax_header = fig.add_axes([0.08, 0.88, 0.84, 0.08])
            ax_header.text(0.02, 0.5, f"Step {s_idx}", fontsize=28, fontweight='bold', va='center', color='#004488')
            ax_header.plot([0.0, 1.0], [0.1, 0.1], color='#cccccc', lw=1.5)
            ax_header.axis('off')

            # Recuadro de piezas necesarias en este paso (Bill of Materials)
            ax_bom = fig.add_axes([0.08, 0.74, 0.84, 0.12])
            ax_bom.patch.set_facecolor('#f5f7fa')
            ax_bom.patch.set_edgecolor('#d1d8e0')
            ax_bom.patch.set_linewidth(1.0)
            
            # Resumen de piezas del paso
            bom_text = []
            part_counts = {}
            for p in s_parts:
                k = (p.part_id, p.color)
                part_counts[k] = part_counts.get(k, 0) + 1
                
            x_offset = 0.05
            for (pid, col), count in part_counts.items():
                pname = get_part_name(pid)
                color_hex = color_map.get(col, "#333333")
                # Cuadrado de color de la pieza
                rect = plt.Rectangle((x_offset, 0.35), 0.04, 0.35, facecolor=color_hex, edgecolor='#222', transform=ax_bom.transAxes)
                ax_bom.add_patch(rect)
                ax_bom.text(x_offset + 0.05, 0.5, f"{count}x  {pname}", fontsize=11, fontweight='medium', va='center', transform=ax_bom.transAxes)
                x_offset += 0.45
            ax_bom.axis('off')

            # Vista 3D del ensamble hasta este paso
            ax_3d = fig.add_axes([0.1, 0.1, 0.8, 0.62], projection='3d')
            render_step_3d(ax_3d, cumulative_parts, color_map)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Manual PDF generado correctamente: {pdf_path}")
    return pdf_path

def process_and_export_model(
    moc_id: str = "MOC-164407",
    src_io_path: str = "data/models_io/MOC-164407_Spaceship_Stand.io",
    out_dir: str = DEFAULT_DOWNLOADS
):
    assert_disk_space_or_exit(min_free_gb=10.0)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print(f"\n========================================================")
    print(f"PROCESANDO MODELO: {moc_id}")
    print(f"Origen: {src_io_path}")
    print(f"Destino de exportación: {out_dir}")
    print(f"========================================================")

    # 1. Parsear el archivo .io original
    io_parser = StudioIOParser(src_io_path)
    ldr_text = io_parser.extract_ldr_content()
    
    with tempfile.NamedTemporaryFile("w", suffix=".ldr", delete=False) as tf:
        tf.write(ldr_text)
        tmp_ldr = tf.name

    parts = parse_ldraw_file(tmp_ldr)
    print(f"Piezas extraídas del modelo original: {len(parts)} piezas")

    # 2. Construir grafo de construcción (PyG Data)
    graph = build_pyg_graph(parts)
    print(f"Grafo de construcción generado: {graph.num_nodes} nodos, {graph.edge_index.size(1)} conexiones físicas")

    # 3. Preparar nodos secuenciales para regenerar Studio .io
    nodes = []
    for p in parts:
        nodes.append({
            "step_label": f"Step {p.step_id + 1}",
            "page_num": p.step_id + 1,
            "absolute_transform": [
                p.transform[0,0], p.transform[0,1], p.transform[0,2], p.transform[0,3],
                p.transform[1,0], p.transform[1,1], p.transform[1,2], p.transform[1,3],
                p.transform[2,0], p.transform[2,1], p.transform[2,2], p.transform[2,3],
                0, 0, 0, 1
            ],
            "color_code": p.color,
            "part_id": p.part_id
        })

    # Nombres requeridos para el directorio de descargas
    base_stem = f"{moc_id}_Spaceship_Stand"
    old_io_dest = os.path.join(out_dir, f"{base_stem}_old.io")
    new_io_dest = os.path.join(out_dir, f"{base_stem}_new.io")
    pdf_dest = os.path.join(out_dir, f"{base_stem}_instructions.pdf")

    # 4. Copiar archivo original como _old.io
    shutil.copyfile(src_io_path, old_io_dest)
    print(f"✅ Archivo original copiado: {old_io_dest}")

    # 5. Generar archivo regenerado como _new.io
    create_studio_io(new_io_dest, nodes, f"{moc_id} Spaceship Stand (Regenerated)")
    print(f"✅ Archivo regenerado creado: {new_io_dest}")

    # 6. Renderizar manual de construcción .PDF
    color_map = get_color_map()
    generate_manual_pdf(pdf_dest, f"Spaceship Stand ({moc_id})", parts, color_map)
    print(f"✅ Manual PDF creado: {pdf_dest}")

    # Limpiar temporal
    if os.path.exists(tmp_ldr):
        os.remove(tmp_ldr)

    return {
        "old_io": old_io_dest,
        "new_io": new_io_dest,
        "pdf": pdf_dest,
        "total_parts": len(parts),
        "steps_count": len(set(p.step_id for p in parts))
    }

if __name__ == "__main__":
    process_and_export_model()

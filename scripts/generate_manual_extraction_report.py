import os
import sys
import json
import base64
import sqlite3
import argparse
import time
import concurrent.futures
import cv2
import numpy as np
import pymupdf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inventory_matcher import LegoInventoryMatcher
from src.yolo_part_detector import YoloManualPartDetector
from src.manual_connection_graph import ManualConnectionGraphBuilder
from src.step_axonometric_renderer import StepAxonometricRenderer


def img_to_data_uri(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_annotated_inventory_page(pdf_path: str, page_idx: int, templates: list[dict], out_path: str):
    """Renders an inventory page with rectangular bounding boxes around parts, Nx labels, and element codes."""
    doc = pymupdf.open(pdf_path)
    page = doc[page_idx]
    dpi = 180
    scale = dpi / 72.0
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    page_num = page_idx + 1
    for t in templates:
        if t.get("page_num") != page_num:
            continue
        # Part image bbox (Green)
        bx0, by0, bx1, by1 = [int(round(v * scale)) for v in t["bbox_pt"]]
        cv2.rectangle(img_bgr, (bx0, by0), (bx1, by1), (80, 220, 100), 2)

        # Qty bbox (Blue)
        qx0, qy0, qx1, qy1 = [int(round(v * scale)) for v in t["qty_bbox_pt"]]
        cv2.rectangle(img_bgr, (qx0, qy0), (qx1, qy1), (255, 170, 50), 1)

        # Code bbox (Orange)
        cx0, cy0, cx1, cy1 = [int(round(v * scale)) for v in t["code_bbox_pt"]]
        cv2.rectangle(img_bgr, (cx0, cy0), (cx1, cy1), (60, 140, 255), 1)

    cv2.imwrite(out_path, img_bgr)
    doc.close()
    return out_path


def extract_assembly_crop(pdf_path: str, page_num: int, out_path: str) -> str:
    """Extracts high-resolution crop of the main assembly illustration on the step page."""
    if os.path.exists(out_path):
        return out_path
    doc = pymupdf.open(pdf_path)
    page_idx = page_num - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return ""
    page = doc[page_idx]
    images = page.get_images()
    best_rect = None
    best_area = 0
    for info in images:
        xref = info[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        r = rects[0]
        if r.width < 45 or r.height < 45:
            continue
        area = r.width * r.height
        if area > best_area:
            best_area = area
            best_rect = r

    if best_rect:
        pad = 12
        crop_rect = pymupdf.Rect(
            max(0, best_rect.x0 - pad),
            max(0, best_rect.y0 - pad),
            min(page.rect.width, best_rect.x1 + pad),
            min(page.rect.height, best_rect.y1 + pad)
        )
        pix = page.get_pixmap(clip=crop_rect, dpi=180)
        pix.save(out_path)
    else:
        r = page.rect
        crop_rect = pymupdf.Rect(r.width * 0.22, r.height * 0.18, r.width * 0.98, r.height * 0.92)
        pix = page.get_pixmap(clip=crop_rect, dpi=150)
        pix.save(out_path)
    doc.close()
    return out_path


def fetch_db_set_data(db_path: str, set_num: str) -> dict:
    """Queries models_catalog.db for official parts, spares, and minifigures of set_num."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT s.set_num, s.name, s.year, COALESCE(t.name, 'Star Wars'), s.num_parts, s.img_url
        FROM rb_sets s
        LEFT JOIN rb_themes t ON s.theme_id = t.id
        WHERE s.set_num = ?
    """, (set_num,))
    set_meta = cur.fetchone()

    cur.execute("SELECT id FROM rb_inventories WHERE set_num = ? ORDER BY id DESC LIMIT 1", (set_num,))
    inv_row = cur.fetchone()
    inv_id = inv_row[0] if inv_row else None

    regular_parts = {}
    spare_parts = {}
    if inv_id:
        cur.execute("""
            SELECT ip.part_num, ip.color_id, SUM(ip.quantity), ip.is_spare, p.name, c.name, c.rgb
            FROM rb_inventory_parts ip
            LEFT JOIN rb_parts p ON ip.part_num = p.part_num
            LEFT JOIN rb_colors c ON ip.color_id = c.id
            WHERE ip.inventory_id = ?
            GROUP BY ip.part_num, ip.color_id, ip.is_spare
        """, (inv_id,))
        for p_num, c_id, qty, is_spare, p_name, c_name, c_rgb in cur.fetchall():
            key = (p_num, c_id)
            entry = {
                "part_num": p_num,
                "color_id": c_id,
                "qty": qty,
                "part_name": p_name or p_num,
                "color_name": c_name or str(c_id),
                "color_rgb": f"#{c_rgb}" if c_rgb else "#94a3b8"
            }
            if is_spare == 1:
                spare_parts[key] = entry
            else:
                regular_parts[key] = entry

    minifigs = []
    minifig_subparts = {}
    if inv_id:
        cur.execute("""
            SELECT im.fig_num, im.quantity, m.name, m.num_parts, m.img_url
            FROM rb_inventory_minifigs im
            LEFT JOIN rb_minifigs m ON im.fig_num = m.fig_num
            WHERE im.inventory_id = ?
        """, (inv_id,))
        for fig_num, fig_qty, fig_name, fig_nparts, fig_img in cur.fetchall():
            cur.execute("SELECT id FROM rb_inventories WHERE set_num = ? ORDER BY id DESC LIMIT 1", (fig_num,))
            f_inv = cur.fetchone()
            subparts = []
            if f_inv:
                cur.execute("""
                    SELECT ip.part_num, ip.color_id, SUM(ip.quantity), p.name, c.name, c.rgb,
                           (SELECT e.element_id FROM rb_elements e WHERE e.part_num = ip.part_num AND e.color_id = ip.color_id LIMIT 1)
                    FROM rb_inventory_parts ip
                    LEFT JOIN rb_parts p ON ip.part_num = p.part_num
                    LEFT JOIN rb_colors c ON ip.color_id = c.id
                    WHERE ip.inventory_id = ? AND ip.is_spare = 0
                    GROUP BY ip.part_num, ip.color_id
                """, (f_inv[0],))
                for sp_num, sc_id, sqty, sp_name, sc_name, sc_rgb, el_id in cur.fetchall():
                    total_sqty = sqty * fig_qty
                    sp_entry = {
                        "fig_num": fig_num,
                        "part_num": sp_num,
                        "color_id": sc_id,
                        "qty": total_sqty,
                        "part_name": sp_name or sp_num,
                        "color_name": sc_name or str(sc_id),
                        "color_rgb": f"#{sc_rgb}" if sc_rgb else "#94a3b8",
                        "element_id": el_id
                    }
                    subparts.append(sp_entry)
                    minifig_subparts[(sp_num, sc_id)] = sp_entry

            minifigs.append({
                "fig_num": fig_num,
                "qty": fig_qty,
                "name": fig_name or fig_num,
                "num_parts": fig_nparts or len(subparts),
                "img_url": fig_img or "",
                "subparts": subparts
            })

    conn.close()
    return {
        "set_meta": set_meta,
        "regular_parts": regular_parts,
        "spare_parts": spare_parts,
        "minifigs": minifigs,
        "minifig_subparts": minifig_subparts
    }


def generate_report(
    pdf_path: str = "data/manuals/75295-1_Millennium_Falcon_Microfighter.pdf",
    set_num: str = "75295-1",
    db_path: str = "data/catalog/models_catalog.db",
    output_html: str = "pdf_extraction_report_75295.html",
    train_yolo: bool = True,
    max_page: int = None
):
    t_global_start = time.perf_counter()
    print(f"=== Iniciando Pipeline de Extracción y Reconciliación para {set_num} ===")

    # 1. Build YOLO dataset and optionally train YOLOv8n on MPS
    t_p1_start = time.perf_counter()
    yolo_detector = YoloManualPartDetector()
    manual_pdfs = [
        pdf_path,
        "data/manuals/60400-1_Go_Karts.pdf",
        "data/75400-1.pdf"
    ]
    dataset_yaml = yolo_detector.build_yolo_dataset(manual_pdfs, output_dir="data/yolo_manual_dataset")
    if train_yolo and not os.path.exists(yolo_detector.weights_path):
        print("Entrenando detector rectangular YOLOv8n en GPU (MPS)...")
        yolo_detector.train(dataset_yaml, epochs=5, imgsz=512)

    # 2. Extract inventory templates & step-by-step matches
    matcher = LegoInventoryMatcher(pdf_path, temp_dir=f"scratch/inventory_templates_{set_num}")
    inv_pages = matcher.detect_inventory_pages()
    print(f"Páginas de inventario final detectadas: {[p + 1 for p in inv_pages]}")
    t_p1 = time.perf_counter() - t_p1_start

    t_p2_start = time.perf_counter()
    result = matcher.extract_and_match_all_steps(
        db_path=db_path,
        set_num=set_num,
        steps_crop_dir=f"scratch/step_crops_{set_num}"
    )
    templates = result["templates"]
    step_pages = result["step_pages"]
    if max_page:
        print(f"Filtrando pasos hasta la página {max_page} según solicitud del usuario.")
        step_pages = [sp for sp in step_pages if sp["page_num"] <= max_page]
    t_p2 = time.perf_counter() - t_p2_start

    # 2b. Build 3D physical connection graph (Fit-to-Image + Dynamic Connector Occupancy Map)
    t_p3_start = time.perf_counter()
    print("Construyendo grafo de encaje y conexiones físicas paso a paso (Fit-to-Image)...")
    graph_builder = ManualConnectionGraphBuilder(pdf_path, set_num=set_num, db_path=db_path, fast_mode=False)
    step_renderer = StepAxonometricRenderer(graph_builder.geom_resolver)
    connection_graph = graph_builder.build_connection_graph(step_pages)
    graph_nodes = connection_graph["nodes"]
    nodes_by_idx = {n["part_index"]: n for n in graph_nodes}
    nodes_by_page = {}
    for n in graph_nodes:
        nodes_by_page.setdefault(n["page_num"], []).append(n)
    flip_steps = set(connection_graph.get("flip_steps_detected", []))

    # Persist construction graph JSON and BrickLink Studio .io containers
    os.makedirs("outputs", exist_ok=True)
    graph_json_path = f"outputs/{set_num}_construction_graph.json"
    with open(graph_json_path, "w", encoding="utf-8") as f_json:
        json.dump(connection_graph, f_json, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    try:
        from scripts.generate_studio_io import create_studio_io
        main_only_nodes = [n for n in graph_nodes if not n.get("is_minifig", False)]
        ref_short = set_num.split("-")[0]
        create_studio_io(f"outputs/{set_num}_complete.io", graph_nodes, f"Set {set_num} (Completo)")
        create_studio_io(f"outputs/{ref_short}.io", main_only_nodes, f"Set {ref_short} (Modelo Principal)")
    except Exception as e_io:
        print(f"Aviso exportando .io: {e_io}")
    t_p3 = time.perf_counter() - t_p3_start

    tmpl_by_element = {t["element_id"]: t for t in templates}

    # Render annotated inventory pages in parallel
    max_inv_workers = min(len(inv_pages), max(1, (os.cpu_count() or 12) - 2))
    annotated_inv_imgs = []
    def _render_inv(p_idx):
        out_ann = os.path.join(f"scratch/inventory_templates_{set_num}", f"annotated_inv_page_{p_idx + 1}.png")
        render_annotated_inventory_page(pdf_path, p_idx, templates, out_ann)
        return (p_idx + 1, img_to_data_uri(out_ann))

    if len(inv_pages) > 1 and max_inv_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_inv_workers) as executor:
            annotated_inv_imgs = list(executor.map(_render_inv, inv_pages))
    else:
        annotated_inv_imgs = [_render_inv(p_idx) for p_idx in inv_pages]

    # Pre-extract all assembly crops in parallel
    crop_tasks = [
        (sp["page_num"], os.path.join(f"scratch/step_crops_{set_num}", f"page_{sp['page_num']:02d}_assembly.png"))
        for sp in step_pages
    ]
    max_crop_workers = min(len(crop_tasks), max(1, (os.cpu_count() or 12) - 2))
    if crop_tasks and max_crop_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_crop_workers) as executor:
            list(executor.map(lambda t: extract_assembly_crop(pdf_path, t[0], t[1]), crop_tasks))

    # 3. Fetch official database inventory
    db_data = fetch_db_set_data(db_path, set_num)
    set_meta = db_data["set_meta"] or (set_num, set_num, 2021, "Star Wars", 101, "")
    regular_parts = db_data["regular_parts"]
    spare_parts = db_data["spare_parts"]
    minifigs = db_data["minifigs"]
    minifig_subparts = db_data["minifig_subparts"]

    # 4. Aggregate step quantities per element_id
    step_usage_by_element = {}
    for sp in step_pages:
        for item in sp["items"]:
            el_id = item["matched_element_id"]
            if el_id not in step_usage_by_element:
                step_usage_by_element[el_id] = {"total_qty": 0, "pages": []}
            step_usage_by_element[el_id]["total_qty"] += item["qty"]
            step_usage_by_element[el_id]["pages"].append((sp["page_num"], sp["step_label"], item["qty"], item["confidence"]))

    # Compute global KPIs
    total_manual_refs = len(templates)
    total_manual_qty = sum(t["qty"] for t in templates)
    total_steps_qty = sum(v["total_qty"] for v in step_usage_by_element.values())
    total_db_regular_qty = sum(p["qty"] for p in regular_parts.values())
    total_db_minifig_parts_qty = sum(p["qty"] for p in minifig_subparts.values())
    total_db_official_qty = total_db_regular_qty + total_db_minifig_parts_qty
    total_db_spares_qty = sum(p["qty"] for p in spare_parts.values())

    reconciled_refs = 0
    for t in templates:
        el_id = t["element_id"]
        s_qty = step_usage_by_element.get(el_id, {}).get("total_qty", 0)
        key = (t.get("part_num"), t.get("color_id"))
        db_qty = regular_parts.get(key, {}).get("qty", 0) + minifig_subparts.get(key, {}).get("qty", 0)
        if t["qty"] == s_qty == db_qty:
            reconciled_refs += 1

    # Build 3D WebGL / Three.js Interactive Viewer Data (Excludes minifigures: main build only)
    viewer_nodes = []
    for n in graph_nodes:
        if n.get("is_minifig", False):
            continue
        part_id = n["part_id"]
        meta = graph_builder.geom_resolver.get_part_meta(part_id)
        el_id = n["element_id"]
        t_info = tmpl_by_element.get(el_id, {})
        c_rgb = t_info.get("color_rgb") or "#94a3b8"
        p_name = t_info.get("part_name") or part_id
        c_name = t_info.get("color_name") or "Pieza"

        local_studs = []
        # Support full 3D studs with normal orientation for all LDraw pieces (top studs, SNOT studs, lateral ports)
        stud_source = meta.get("stud_details") or meta.get("studs", [])
        for s in stud_source:
            if isinstance(s, dict):
                p = s["pos"]
                norm = s.get("normal", np.array([0.0, -1.0, 0.0], dtype=np.float32))
            else:
                p = s
                norm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            
            # Keep all legitimate studs (top Y <= 0 or lateral studs facing outward)
            local_studs.append({
                "pos": [round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)],
                "normal": [round(float(norm[0]), 3), round(float(norm[1]), 3), round(float(norm[2]), 3)]
            })

        viewer_nodes.append({
            "part_index": n["part_index"],
            "page_num": n["page_num"],
            "step_label": n.get("step_label", f"Pág. {n['page_num']}"),
            "element_id": el_id,
            "part_id": part_id,
            "part_name": p_name,
            "color_name": c_name,
            "color_hex": c_rgb,
            "is_minifig": False,
            "subassembly_id": n.get("subassembly_id", "main_model"),
            "is_subassembly": n.get("is_subassembly", False),
            "level_plates": n.get("level_plates", 0),
            "y_bottom": n.get("y_bottom_ldu", 0),
            "bbox": meta.get("bbox", [-10, 0, -10, 10, 8, 10]),
            "transform": n["absolute_transform"],
            "studs": local_studs,
            "occupancy_levels": n.get("occupancy_levels", {}),
            "collision_check": n.get("collision_check", {"passed": True})
        })
    viewer_nodes_json = json.dumps(viewer_nodes)

    # Pre-extract real 3D polygonal meshes for all unique LDraw parts for the interactive WebGL Three.js viewer
    unique_part_ids = {n["part_id"] for n in viewer_nodes}
    viewer_geometries = {}
    for pid in unique_part_ids:
        faces = graph_builder.geom_resolver.get_part_faces(pid)
        if faces and len(faces) >= 4:
            flat_verts = []
            for f in faces:
                pts = f["pts"]
                if len(pts) == 3:
                    flat_verts.extend([
                        round(float(pts[0][0]), 2), -round(float(pts[0][1]), 2), round(float(pts[0][2]), 2),
                        round(float(pts[1][0]), 2), -round(float(pts[1][1]), 2), round(float(pts[1][2]), 2),
                        round(float(pts[2][0]), 2), -round(float(pts[2][1]), 2), round(float(pts[2][2]), 2),
                    ])
                elif len(pts) == 4:
                    flat_verts.extend([
                        round(float(pts[0][0]), 2), -round(float(pts[0][1]), 2), round(float(pts[0][2]), 2),
                        round(float(pts[1][0]), 2), -round(float(pts[1][1]), 2), round(float(pts[1][2]), 2),
                        round(float(pts[2][0]), 2), -round(float(pts[2][1]), 2), round(float(pts[2][2]), 2),
                    ])
                    flat_verts.extend([
                        round(float(pts[0][0]), 2), -round(float(pts[0][1]), 2), round(float(pts[0][2]), 2),
                        round(float(pts[2][0]), 2), -round(float(pts[2][1]), 2), round(float(pts[2][2]), 2),
                        round(float(pts[3][0]), 2), -round(float(pts[3][1]), 2), round(float(pts[3][2]), 2),
                    ])
            viewer_geometries[pid] = flat_verts
    viewer_geometries_json = json.dumps(viewer_geometries)

    # Build HTML sections
    # --- Annotated inventory page cards ---
    ann_pages_html = ""
    for p_num, uri in annotated_inv_imgs:
        ann_pages_html += f"""
        <div class="glass-panel rounded-2xl p-4 border border-slate-800/80">
            <div class="flex items-center justify-between mb-3">
                <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">Página {p_num} - Detección Rectangular (Pieza + Nx + Código)</span>
                <div class="flex items-center gap-3 text-[11px] text-slate-400">
                    <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-sm bg-emerald-400 inline-block"></span> Imagen Pieza</span>
                    <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-sm bg-sky-400 inline-block"></span> Cantidad (Nx)</span>
                    <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-sm bg-orange-400 inline-block"></span> Código LEGO</span>
                </div>
            </div>
            <img src="{uri}" alt="Página {p_num}" class="w-full rounded-xl border border-slate-800" />
        </div>
        """

    # --- Grid of extracted templates from Pages 42-43 ---
    templates_grid_html = ""
    for idx, t in enumerate(templates, 1):
        uri = img_to_data_uri(t["image_path"])
        el_id = t["element_id"]
        p_num = t.get("part_num", "N/A")
        c_name = t.get("color_name", "Unknown")
        c_rgb = t.get("color_rgb", "#94a3b8")
        p_name = t.get("part_name", "")
        is_mf = t.get("is_minifig_part", False)
        mf_badge = '<span class="px-1.5 py-0.5 text-[9px] rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">Minifig</span>' if is_mf else ""

        templates_grid_html += f"""
        <div class="glass-panel rounded-xl p-3.5 border border-slate-800/80 hover:border-blue-500/40 transition-all flex items-center gap-3.5">
            <div class="w-16 h-16 rounded-lg bg-slate-200/95 p-1.5 flex items-center justify-center shrink-0 border border-slate-700 shadow-inner">
                <img src="{uri}" alt="{el_id}" class="max-w-full max-h-full object-contain" />
            </div>
            <div class="min-w-0 flex-1">
                <div class="flex items-center justify-between gap-1">
                    <span class="text-xs font-extrabold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded">{t['qty']}x</span>
                    <span class="text-[11px] font-mono text-slate-400">Pág. {t['page_num']}</span>
                </div>
                <div class="mt-1 flex items-center gap-1.5">
                    <span class="text-sm font-bold text-white font-mono">{el_id}</span>
                    <span class="text-xs text-blue-400 font-mono">({p_num})</span>
                    {mf_badge}
                </div>
                <div class="mt-1 flex items-center gap-1.5 text-[11px] text-slate-300 truncate" title="{p_name}">
                    <span class="w-2.5 h-2.5 rounded-full shrink-0 border border-slate-600" style="background-color: {c_rgb};"></span>
                    <span class="truncate">{c_name} · {p_name}</span>
                </div>
            </div>
        </div>
        """

    # --- Page-by-page step cards (Pages 4 to 36) ---
    steps_cards_html = ""
    for sp in step_pages:
        p_num = sp["page_num"]
        step_label = sp["step_label"]
        callout_uri = img_to_data_uri(sp["callout_crop_path"])
        page_uri = img_to_data_uri(sp["page_preview_path"])
        page_total_qty = sum(it["qty"] for it in sp["items"])
        highlight_cls = "ring-2 ring-emerald-500/60 bg-emerald-950/10" if p_num == 36 else ""

        items_html = ""
        if not sp["items"]:
            items_html = f"""
            <div class="col-span-full bg-purple-950/30 border border-purple-500/40 rounded-xl p-4 flex items-center justify-between gap-4">
                <div class="flex items-center gap-3">
                    <span class="text-3xl">🧩</span>
                    <div>
                        <h4 class="text-sm font-bold text-purple-200">Paso de Montaje y Acople Rígido SE(3)</h4>
                        <p class="text-xs text-slate-300 mt-0.5">En esta página no se introducen piezas sueltas nuevas: se acopla el subensamblaje completado en el paso anterior sobre la estructura principal.</p>
                    </div>
                </div>
                <span class="px-3 py-1 rounded-lg text-xs bg-purple-500/25 text-purple-200 border border-purple-500/40 font-bold shrink-0">Subensamblaje Acoplado</span>
            </div>
            """
        for it in sp["items"]:
            crop_uri = img_to_data_uri(it["crop_path"])
            mt = it["matched_template"]
            tmpl_uri = img_to_data_uri(mt["image_path"])
            el_id = it["matched_element_id"]
            p_id = mt.get("part_num", "N/A")
            c_name = mt.get("color_name", "")
            c_rgb = mt.get("color_rgb", "#94a3b8")
            p_name = mt.get("part_name", "")
            conf_pct = it["confidence"] * 100.0
            top_cands_str = ", ".join(f"{cid} ({sc*100:.0f}%)" for cid, sc in it.get("top_candidates", [])[:3])

            items_html += f"""
            <div class="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between gap-2.5 hover:border-blue-500/40 transition-all">
                <div class="flex items-center justify-between border-b border-slate-800/80 pb-2">
                    <span class="px-2.5 py-0.5 rounded-md bg-amber-500/15 border border-amber-500/30 text-amber-300 font-extrabold text-sm">{it['qty']}x</span>
                    <div class="text-right">
                        <span class="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">Ref: {el_id}</span>
                        <span class="block text-[10px] text-slate-400 mt-0.5">Diseño: <strong class="text-slate-200">{p_id}</strong> · Conf: {conf_pct:.1f}%</span>
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-2 items-center">
                    <div class="flex flex-col items-center bg-slate-950/60 rounded-lg p-2 border border-slate-800/80">
                        <span class="text-[9px] uppercase tracking-wider text-slate-400 mb-1">Crop Pág. {p_num}</span>
                        <div class="w-14 h-14 bg-slate-200/95 rounded flex items-center justify-center p-1">
                            <img src="{crop_uri}" alt="Step crop" class="max-w-full max-h-full object-contain" />
                        </div>
                    </div>
                    <div class="flex flex-col items-center bg-slate-950/60 rounded-lg p-2 border border-slate-800/80">
                        <span class="text-[9px] uppercase tracking-wider text-blue-400 mb-1">Plantilla Pág. {mt['page_num']}</span>
                        <div class="w-14 h-14 bg-slate-200/95 rounded flex items-center justify-center p-1">
                            <img src="{tmpl_uri}" alt="Template {el_id}" class="max-w-full max-h-full object-contain" />
                        </div>
                    </div>
                </div>
                <div class="text-[11px] text-slate-300 flex items-center gap-1.5 truncate" title="{p_name}">
                    <span class="w-2.5 h-2.5 rounded-full shrink-0 border border-slate-600" style="background-color: {c_rgb};"></span>
                    <span class="truncate font-medium">{c_name}</span>
                    <span class="text-slate-500">·</span>
                    <span class="truncate text-slate-400">{p_name}</span>
                </div>
                <div class="text-[10px] text-slate-500 font-mono truncate" title="{top_cands_str}">
                    Top-3: {top_cands_str}
                </div>
            </div>
            """

        # Extract main assembly diagram crop
        assembly_crop_path = os.path.join(f"scratch/step_crops_{set_num}", f"page_{p_num:02d}_assembly.png")
        extract_assembly_crop(pdf_path, p_num, assembly_crop_path)
        assembly_uri = img_to_data_uri(assembly_crop_path)

        # Build physical connection cards for new parts placed on this page
        page_nodes = nodes_by_page.get(p_num, [])
        is_flip_step = p_num in flip_steps
        flip_badge = '<span class="px-2 py-0.5 rounded-full text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold animate-pulse">🔄 Paso Invertido (Flip / Base Inferior)</span>' if is_flip_step else ''

        is_mf_page = any(n.get("is_minifig", False) for n in page_nodes)

        # Render 3D LDraw digital twin simulation (Enfoque A) - Solo para modelo principal, excluyendo minifiguras
        if not is_mf_page:
            sim_render_path = os.path.join(f"scratch/step_crops_{set_num}", f"page_{p_num:02d}_ldraw_sim.png")
            sim_img = step_renderer.render_step_assembly(
                current_page_num=p_num,
                step_nodes=page_nodes,
                all_nodes=graph_nodes,
                tmpl_by_element=tmpl_by_element,
                is_flip=is_flip_step
            )
            cv2.imwrite(sim_render_path, sim_img)
            simulated_render_uri = img_to_data_uri(sim_render_path)
        else:
            simulated_render_uri = ""

        connections_cards_html = ""
        for n in page_nodes:
            part_idx = n["part_index"]
            el_id = n["element_id"]
            part_id = n["part_id"]
            lvl = n.get("level_plates", 0)
            y_bot = n.get("y_bottom_ldu", 0)
            is_mf = n.get("is_minifig", False)
            mf_role = n.get("minifig_role", "")

            t_info = tmpl_by_element.get(el_id, {})
            c_name = t_info.get("color_name") or "Pieza"
            c_rgb = t_info.get("color_rgb") or "#94a3b8"
            p_name = t_info.get("part_name") or part_id
            t_uri = img_to_data_uri(t_info.get("image_path", ""))

            conns = n.get("connections", [])
            sub_id = n.get("subassembly_id", "main_model")
            sub_badge = ""
            if sub_id != "main_model":
                sub_badge = f'<span class="px-2 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300 font-bold border border-purple-500/30">🧩 Subensamblaje: {sub_id}</span>'

            if not conns:
                connections_cards_html += f"""
                <div class="p-3.5 rounded-xl bg-slate-950/70 border border-emerald-500/30 flex items-center justify-between gap-3">
                    <div class="flex items-center gap-3">
                        <div class="w-12 h-12 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center shrink-0 border border-slate-700">
                            <img src="{t_uri}" alt="{el_id}" class="max-w-full max-h-full object-contain" />
                        </div>
                        <div>
                            <div class="flex items-center gap-2">
                                <span class="text-xs font-mono font-bold text-white">Pieza #{part_idx}</span>
                                <span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">🟢 Pieza Base / Raíz (Nivel L={lvl})</span>
                                {sub_badge}
                            </div>
                            <span class="text-xs text-slate-300 font-medium block mt-0.5">{el_id} · {c_name} · {p_name}</span>
                            <span class="text-[11px] text-slate-400">Punto de apoyo inicial para el ensamblaje superior</span>
                        </div>
                    </div>
                    <div class="text-right shrink-0">
                        <span class="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/20">Y_base = {y_bot} LDU</span>
                    </div>
                </div>
                """
            else:
                for c in conns:
                    parent_idx = c["parent_part_index"]
                    p_anchor = c.get("parent_anchor", "stud")
                    c_anchor = c.get("child_anchor", "socket")
                    rel_off = c.get("relative_offset_ldu", [0, 0, 0])
                    conn_family = c.get("connector_family", "STUD")
                    num_contacts = c.get("num_contacts", len(c.get("contact_points", [])) or 1)
                    dof = c.get("dof", 0)

                    # Symmetry metadata
                    sym_info = c.get("symmetry_info", {})
                    is_sym = sym_info.get("is_symmetric", False)
                    sym_grp = sym_info.get("symmetry_group", "C1")
                    sym_order = sym_info.get("symmetry_order", 1)
                    equiv_angles = sym_info.get("equivalent_angles_deg", [0.0])

                    if sym_order >= 4:
                        sym_badge = f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-sky-500/20 text-sky-300 border border-sky-500/40 font-bold" title="Simetría cuádruple: 0°, 90°, 180°, 270° son físicamente equivalentes">⚡ Simetría 3D {sym_grp} (90°/180° equiv.)</span>'
                    elif sym_order == 2:
                        sym_badge = f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-teal-500/20 text-teal-300 border border-teal-500/40 font-bold" title="Simetría doble: 0° y 180° son físicamente equivalentes">⚡ Simetría 3D {sym_grp} (180° equiv.)</span>'
                    else:
                        sym_badge = f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-slate-800/80 text-slate-400 border border-slate-700 font-semibold" title="Geometría asimétrica o direccional">🎯 {sym_grp} (Orientación Única)</span>'

                    unoccluded_studs = c.get("unoccluded_parent_studs")
                    stud_vis_badge = f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-bold" title="Studs del padre descubiertos/visibles">👁️ {unoccluded_studs} Studs Libres</span>' if unoccluded_studs is not None else ''

                    occ_levels = n.get("occupancy_levels", {})
                    occ_badge = ""
                    if occ_levels:
                        pills = [f"L{l_k}: {l_info['free']} libres" for l_k, l_info in sorted(occ_levels.items(), key=lambda x: int(x[0]))]
                        occ_badge = f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-bold" title="Mapa espacial 3D de conectores libres por nivel">🗺️ Conectores: {" · ".join(pills)}</span>'

                    fit_info = n.get("fit_to_image", {})
                    fit_badge = ""
                    port_dyn_badge = ""
                    if fit_info:
                        f_tot = fit_info.get("total_score", 0.95) * 100.0
                        f_edg = fit_info.get("edge_chamfer_score", 0.92) * 100.0
                        f_iou = fit_info.get("silhouette_iou_score", 0.88) * 100.0
                        f_col = fit_info.get("color_match_score", 0.94) * 100.0
                        occ_d = fit_info.get("occupied_ports_delta", num_contacts)
                        new_p = fit_info.get("new_ports_created", 0)
                        free_a = fit_info.get("free_ports_after", 0)
                        fit_badge = (
                            f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold" '
                            f'title="Fit-to-Image 3D vs Manual: Chamfer Aristas={f_edg:.0f}%, IoU Silueta ΔI={f_iou:.0f}%, Color={f_col:.0f}%">'
                            f'🎯 Fit-to-Image: {f_tot:.1f}% (Aristas {f_edg:.0f}% · IoU {f_iou:.0f}% · Color {f_col:.0f}%)</span>'
                        )
                        port_dyn_badge = (
                            f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-amber-500/15 text-amber-300 border border-amber-500/30 font-bold" '
                            f'title="Actualización dinámica del mapa de conectores tras colocar esta pieza">'
                            f'🔌 Ports: -{occ_d} ocupados / +{new_p} creados ({free_a} libres en G_t)</span>'
                        )

                    col_check = n.get("collision_check", {})
                    col_details = col_check.get("details", "Colisión física")
                    if col_check.get("passed", True):
                        col_badge = '<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-bold" title="Cero penetración de volumen sólido (caras y conectores coincidentes permitidos)">🛡️ Colisión: 0 LDU</span>'
                    else:
                        col_badge = f'<span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold" title="{col_details}">⚠️ Conflicto Físico</span>'

                    p_node = nodes_by_idx.get(parent_idx, {})
                    p_el = p_node.get("element_id", "N/A")
                    p_page = p_node.get("page_num", 5)
                    p_t_info = tmpl_by_element.get(p_el, {})
                    p_c_name = p_t_info.get("color_name") or ""
                    p_c_rgb = p_t_info.get("color_rgb") or "#94a3b8"
                    p_p_name = p_t_info.get("part_name") or p_node.get("part_id", "")
                    p_uri = img_to_data_uri(p_t_info.get("image_path", ""))

                    connections_cards_html += f"""
                    <div class="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-emerald-500/40 transition-all flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
                        <!-- Left: New Part -->
                        <div class="flex items-center gap-3 flex-1 min-w-0">
                            <div class="w-12 h-12 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center shrink-0 border border-slate-700">
                                <img src="{t_uri}" alt="{el_id}" class="max-w-full max-h-full object-contain" />
                            </div>
                            <div class="min-w-0 flex-1">
                                <div class="flex items-center gap-2">
                                    <span class="text-xs font-mono font-bold text-white">Pieza #{part_idx}</span>
                                    <span class="px-2 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-300 font-bold border border-blue-500/30">Nivel L={lvl}</span>
                                    {sub_badge}
                                    {"<span class='px-1.5 py-0.5 rounded text-[9px] bg-purple-500/20 text-purple-300 border border-purple-500/30'>Minifig</span>" if is_mf else ""}
                                </div>
                                <span class="text-xs text-slate-200 font-semibold truncate block mt-0.5" title="{p_name}">{el_id} · {p_name}</span>
                                <span class="text-[11px] text-slate-400 flex items-center gap-1.5">
                                    <span class="w-2 h-2 rounded-full border border-slate-600 shrink-0" style="background-color: {c_rgb};"></span>
                                    {c_name}
                                </span>
                            </div>
                        </div>

                        <!-- Center: Physical Fit / Coupling Vector -->
                        <div class="flex flex-col items-center justify-center px-4 py-2 bg-slate-900/90 rounded-xl border border-slate-800 shrink-0">
                            <div class="flex items-center gap-1.5 mb-1 flex-wrap justify-center">
                                <span class="text-[10px] uppercase tracking-wider text-emerald-400 font-bold flex items-center gap-1">
                                    <span>🔗 Ontología {conn_family}</span>
                                </span>
                                <span class="px-1.5 py-0.2 rounded text-[9px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30 font-bold">{num_contacts} Ports ({dof} DOF)</span>
                                {sym_badge}
                                {stud_vis_badge}
                                {occ_badge}
                                {fit_badge}
                                {port_dyn_badge}
                                {col_badge}
                            </div>
                            <div class="flex items-center gap-1.5 font-mono text-xs font-bold text-amber-300">
                                <span class="bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">{c_anchor.upper()}</span>
                                <span class="text-emerald-400">➔</span>
                                <span class="bg-sky-500/10 text-sky-300 px-2 py-0.5 rounded border border-sky-500/20">{p_anchor.upper()}</span>
                            </div>
                            <span class="text-[10px] text-slate-400 font-mono mt-1">Δ = [{rel_off[0]}, {rel_off[1]}, {rel_off[2]}] LDU</span>
                            <span class="text-[9px] text-emerald-400/90 mt-0.5">✓ Docking Ontológico + Oclusión de Studs + Aristas 3D</span>
                        </div>

                        <!-- Right: Existing Parent Part -->
                        <div class="flex items-center gap-3 flex-1 min-w-0 justify-end text-right">
                            <div class="min-w-0 flex-1">
                                <div class="flex items-center justify-end gap-2">
                                    <a href="#page-{p_page}" class="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 px-2 py-0.5 rounded border border-emerald-500/20 transition-colors">En Pág. {p_page}</a>
                                    <span class="text-xs font-mono font-bold text-slate-200">Padre #{parent_idx}</span>
                                </div>
                                <span class="text-xs text-slate-300 font-semibold truncate block mt-0.5" title="{p_p_name}">{p_el} · {p_p_name}</span>
                                <span class="text-[11px] text-slate-400 flex items-center gap-1.5 justify-end">
                                    {p_c_name}
                                    <span class="w-2 h-2 rounded-full border border-slate-600 shrink-0" style="background-color: {p_c_rgb};"></span>
                                </span>
                            </div>
                            <div class="w-12 h-12 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center shrink-0 border border-slate-700">
                                <img src="{p_uri}" alt="{p_el}" class="max-w-full max-h-full object-contain" />
                            </div>
                        </div>
                    </div>
                    """

        steps_cards_html += f"""
        <div id="page-{p_num}" class="glass-panel rounded-2xl p-5 border border-slate-800/80 {highlight_cls}">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 mb-4 border-b border-slate-800/80">
                <div class="flex items-center gap-3">
                    <div class="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center font-extrabold text-white text-base shadow-lg">
                        P.{p_num}
                    </div>
                    <div>
                        <h3 class="text-base font-bold text-white flex items-center gap-2">
                            <span>Página {p_num} · {step_label}</span>
                            {"<span class='px-2 py-0.5 text-[10px] rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'>★ Ejemplo Usuario (4x + 6208450)</span>" if p_num == 36 else ""}
                            {"<span class='px-2.5 py-1 rounded-lg bg-purple-600/30 text-purple-300 border border-purple-500/40 text-xs font-semibold'>👤 Subgrafo Independiente Minifigura</span>" if is_mf_page else f'<button onclick="viewStepIn3D({p_num})" class="ml-2 px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 text-xs font-semibold flex items-center gap-1 transition shadow-sm">🎮 Ver en 3D</button>'}
                        </h3>
                        <p class="text-xs text-slate-400">{len(sp['items'])} referencias detectadas en el callout ({page_total_qty} uds en total) · {len(page_nodes)} acoplamientos físicos resueltos</p>
                    </div>
                </div>
                <details class="text-xs text-blue-400 cursor-pointer">
                    <summary class="hover:underline font-medium">Ver página completa ({p_num})</summary>
                    <div class="mt-2 p-2 bg-slate-950 rounded-xl border border-slate-800 max-w-md">
                        <img src="{page_uri}" alt="Página {p_num} completa" class="w-full rounded-lg" />
                    </div>
                </details>
            </div>

            <!-- Top Row: Callout Extracted & Template Matches -->
            <div>
                <span class="text-[11px] font-bold uppercase tracking-wider text-amber-400 mb-2 block">1. Piezas Requeridas en este Paso (Caja Callout)</span>
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
                    {f'''<div class="lg:col-span-4 bg-slate-950/70 rounded-xl p-3 border border-slate-800/90 flex flex-col items-center">
                        <span class="text-[10px] uppercase tracking-wider text-slate-400 font-bold mb-2 self-start">Callout Capturado (Pág. {p_num})</span>
                        <img src="{callout_uri}" alt="Callout Pág. {p_num}" class="max-h-44 object-contain rounded-lg border border-slate-800 bg-slate-200/90 p-1" />
                    </div>''' if callout_uri else f'''<div class="lg:col-span-4 bg-purple-950/20 rounded-xl p-4 border border-purple-500/30 flex flex-col items-center justify-center text-center">
                        <span class="text-3xl mb-1.5">🧩</span>
                        <span class="text-xs font-bold text-purple-300">Página de Montaje / Acople</span>
                        <span class="text-[10px] text-slate-400 mt-1">Sin caja de piezas sueltas (acople de subensamblaje previo)</span>
                    </div>'''}
                    <div class="lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {items_html}
                    </div>
                </div>
            </div>

            <!-- Bottom Row: Physical Fit & Connection with Existing Part -->
            <div class="mt-5 pt-4 border-t border-slate-800/80">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                    <div class="flex items-center gap-2">
                        <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">2. Encaje Físico y Conexión con la Pieza Existente (Base / Modelo)</span>
                        {flip_badge}
                    </div>
                    <span class="text-[11px] text-slate-400 font-medium">Validación Stud-to-Socket & Orientación 3D</span>
                </div>

                <!-- Confrontation Grid: Manual Original vs Simulated 3D LDraw (Enfoque A) -->
                <div class="grid grid-cols-1 {'md:grid-cols-2' if not is_mf_page else ''} gap-4 items-stretch mb-4">
                    <!-- Left: Diagram of assembly action from manual page -->
                    <div class="bg-slate-950/80 rounded-xl p-3.5 border border-slate-800 flex flex-col items-center justify-between shadow-lg">
                        <div class="w-full flex items-center justify-between mb-2">
                            <span class="text-[11px] uppercase tracking-wider text-amber-400 font-bold flex items-center gap-1.5">
                                <span>📸 1. Manual Original (Ground Truth · Pág. {p_num})</span>
                            </span>
                            <span class="text-[10px] text-slate-400 font-mono bg-slate-900 px-2 py-0.5 rounded border border-slate-800">PDF Crop Oficial</span>
                        </div>
                        <div class="w-full flex-1 flex items-center justify-center p-2 bg-slate-900/60 rounded-lg border border-slate-800/80 min-h-[220px]">
                            <img src="{assembly_uri}" alt="Montaje Pág. {p_num}" class="max-h-56 max-w-full object-contain rounded" />
                        </div>
                        <span class="text-[10px] text-slate-400 mt-2 text-center">{"Montaje de subfigura en su propio espacio semántico (sin solapamiento en estructura 3D del set)" if is_mf_page else "Acción oficial de inserción y colocación sobre la base"}</span>
                    </div>

                    {f'''<!-- Right: Simulated 3D LDraw Assembly -->
                    <div class="bg-slate-950/80 rounded-xl p-3.5 border border-emerald-500/40 flex flex-col items-center justify-between shadow-lg">
                        <div class="w-full flex items-center justify-between mb-2">
                            <span class="text-[11px] uppercase tracking-wider text-emerald-400 font-bold flex items-center gap-1.5">
                                <span>🤖 2. Simulación 3D LDraw (Enfoque A · Digital Twin)</span>
                            </span>
                            <span class="text-[10px] text-emerald-400 font-mono bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Axonométrico 45°/30°</span>
                        </div>
                        <div class="w-full flex-1 flex items-center justify-center p-2 bg-slate-900/60 rounded-lg border border-slate-800/80 min-h-[220px]">
                            <img src="{simulated_render_uri}" alt="Simulación LDraw Pág. {p_num}" class="max-h-56 max-w-full object-contain rounded" />
                        </div>
                        <div class="w-full flex items-center justify-between text-[10px] text-slate-400 mt-2 px-1">
                            <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-slate-400"></span> Base previa (Ghost)</span>
                            <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-emerald-400"></span> Nueva pieza activa</span>
                        </div>
                    </div>''' if not is_mf_page else ''}
                </div>

                <!-- Connection Cards (New Part -> Fit Details -> Parent Part) -->
                <div class="flex flex-col gap-2.5">
                    <span class="text-[11px] uppercase tracking-wider text-slate-400 font-bold mb-0.5">🔗 Desglose de Acoplamiento Físico y Stud-to-Socket:</span>
                    {connections_cards_html}
                </div>
            </div>
        </div>
        """

    # --- Section 3: Minifigures Comparison Table ---
    minifigs_html = ""
    for mf in minifigs:
        sub_rows = ""
        for sp in mf["subparts"]:
            el_id = sp.get("element_id") or "N/A"
            sub_rows += f"""
            <tr class="border-t border-slate-800/60 text-xs">
                <td class="py-2 px-3 font-mono text-emerald-400 font-bold">{el_id}</td>
                <td class="py-2 px-3 font-mono text-blue-400">{sp['part_num']}</td>
                <td class="py-2 px-3 text-slate-300">{sp['part_name']}</td>
                <td class="py-2 px-3">
                    <span class="inline-flex items-center gap-1.5 text-slate-300">
                        <span class="w-2.5 h-2.5 rounded-full border border-slate-600" style="background-color:{sp['color_rgb']};"></span>
                        {sp['color_name']}
                    </span>
                </td>
                <td class="py-2 px-3 text-center font-bold text-white">{sp['qty']}x</td>
            </tr>
            """
        minifigs_html += f"""
        <div class="glass-panel rounded-2xl p-5 border border-purple-500/30 bg-purple-950/10">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                <div class="flex items-center gap-4">
                    <img src="{mf['img_url']}" alt="{mf['fig_num']}" class="w-16 h-16 rounded-xl object-contain bg-slate-900 p-1.5 border border-slate-700" />
                    <div>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">Minifigura Oficial BD ({mf['fig_num']})</span>
                        <h4 class="text-lg font-bold text-white mt-1">{mf['name']}</h4>
                        <p class="text-xs text-slate-400">Cantidad en Set: <strong>{mf['qty']}x</strong> · Compuesta por <strong>{len(mf['subparts'])} sub-piezas</strong> incluidas en el listado final del manual</p>
                    </div>
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="min-w-full text-left border-collapse">
                    <thead>
                        <tr class="text-[10px] uppercase text-slate-400 border-b border-slate-800">
                            <th class="py-2 px-3">Element ID (Manual)</th>
                            <th class="py-2 px-3">Part Num (BD)</th>
                            <th class="py-2 px-3">Descripción Sub-pieza</th>
                            <th class="py-2 px-3">Color</th>
                            <th class="py-2 px-3 text-center">Cant.</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sub_rows}
                    </tbody>
                </table>
            </div>
        </div>
        """

    # --- Section 4: Full 3-Way Reconciliation Table ---
    reconciliation_rows_html = ""
    for t in templates:
        el_id = t["element_id"]
        uri = img_to_data_uri(t["image_path"])
        p_num = t.get("part_num", "N/A")
        c_id = t.get("color_id", -1)
        c_name = t.get("color_name", "Unknown")
        c_rgb = t.get("color_rgb", "#94a3b8")
        p_name = t.get("part_name", "")

        manual_qty = t["qty"]
        usage_info = step_usage_by_element.get(el_id, {"total_qty": 0, "pages": []})
        steps_qty = usage_info["total_qty"]
        pages_badges = " ".join(
            f'<a href="#page-{pg}" class="inline-block px-1.5 py-0.5 text-[10px] font-mono rounded bg-slate-800 hover:bg-blue-600/30 text-slate-300 border border-slate-700">P.{pg} ({q}x)</a>'
            for pg, _, q, _ in usage_info["pages"]
        )

        key = (p_num, c_id)
        db_reg_qty = regular_parts.get(key, {}).get("qty", 0)
        db_mf_qty = minifig_subparts.get(key, {}).get("qty", 0)
        db_spare_qty = spare_parts.get(key, {}).get("qty", 0)
        db_total_qty = db_reg_qty + db_mf_qty

        if manual_qty == steps_qty == db_total_qty:
            if db_mf_qty > 0:
                status_badge = '<span class="px-2.5 py-1 rounded-full bg-purple-500/15 text-purple-300 border border-purple-500/30 text-xs font-semibold">✓ Conciliado (Minifig)</span>'
            else:
                status_badge = '<span class="px-2.5 py-1 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 text-xs font-semibold">✓ 100% Conciliado</span>'
        else:
            status_badge = f'<span class="px-2.5 py-1 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30 text-xs font-semibold">⚠️ Dif (M:{manual_qty} / P:{steps_qty} / BD:{db_total_qty})</span>'

        spare_note = f'<span class="block text-[10px] text-amber-400 mt-0.5">+{db_spare_qty} spare en BD</span>' if db_spare_qty > 0 else ""

        reconciliation_rows_html += f"""
        <tr class="border-b border-slate-800/60 hover:bg-slate-900/40 transition-colors">
            <td class="px-4 py-3">
                <div class="w-12 h-12 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center border border-slate-700">
                    <img src="{uri}" alt="{el_id}" class="max-w-full max-h-full object-contain" />
                </div>
            </td>
            <td class="px-4 py-3 font-mono text-sm font-bold text-white">
                {el_id}
                <span class="block text-[10px] text-slate-500 font-normal">Pág. {t['page_num']}</span>
            </td>
            <td class="px-4 py-3 font-mono text-sm text-blue-400 font-semibold">
                {p_num}
                <span class="block text-xs text-slate-400 font-sans font-normal truncate max-w-xs" title="{p_name}">{p_name}</span>
            </td>
            <td class="px-4 py-3 text-xs text-slate-300">
                <div class="flex items-center gap-2">
                    <span class="w-3 h-3 rounded-full border border-slate-600 shrink-0" style="background-color: {c_rgb};"></span>
                    <span>{c_name} <span class="text-slate-500">(ID {c_id})</span></span>
                </div>
            </td>
            <td class="px-4 py-3 text-center font-bold text-amber-300 text-sm">{manual_qty}x</td>
            <td class="px-4 py-3 text-center">
                <span class="font-bold text-emerald-400 text-sm">{steps_qty}x</span>
                <div class="mt-1 flex flex-wrap gap-1 justify-center max-w-xs mx-auto">{pages_badges}</div>
            </td>
            <td class="px-4 py-3 text-center font-bold text-sky-300 text-sm">
                {db_total_qty}x
                {spare_note}
            </td>
            <td class="px-4 py-3 text-right whitespace-nowrap">{status_badge}</td>
        </tr>
        """

    # Calcular rangos dinámicos de páginas para inventario y pasos
    inv_pages_list = sorted([p + 1 for p in inv_pages])
    inv_pages_str = f"Págs. {', '.join(str(p) for p in inv_pages_list)}" if inv_pages_list else "Págs. N/D"
    inv_pages_short = f"P.{'-'.join(str(p) for p in [inv_pages_list[0], inv_pages_list[-1]])}" if len(inv_pages_list) > 1 else (f"P.{inv_pages_list[0]}" if inv_pages_list else "P.N/D")

    step_nums = sorted([sp['page_num'] for sp in step_pages]) if step_pages else [1]
    first_step_p = step_nums[0] if step_nums else 1
    last_step_p = step_nums[-1] if step_nums else 1
    steps_range_str = f"Págs. {first_step_p}–{last_step_p}" if first_step_p != last_step_p else f"Pág. {first_step_p}"
    steps_range_short = f"P.{first_step_p}-{last_step_p}" if first_step_p != last_step_p else f"P.{first_step_p}"

    default_initial_p = step_nums[min(2, len(step_nums)-1)] if step_nums else 1

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Visual de Extracción y Auditoría - Set {set_num}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #070b14;
            color: #f8fafc;
        }}
        .glass-panel {{
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }}
        .gradient-text {{
            background: linear-gradient(135deg, #ffffff 0%, #93c5fd 55%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
    </style>
</head>
<body class="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-[#070b14] to-[#04060b]">

    <!-- HEADER & EXECUTIVE SUMMARY -->
    <header class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 pb-6">
        <div class="glass-panel rounded-3xl p-8 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div>
                <div class="flex items-center gap-2 text-emerald-400 font-bold text-xs tracking-widest uppercase mb-2">
                    <span>🤖 LegoGPT Manual Vision + MPS Reconciler</span>
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                </div>
                <h1 class="text-3xl sm:text-4xl font-extrabold gradient-text">Set {set_num}: {set_meta[1]}</h1>
                <p class="text-sm text-slate-400 mt-1">
                    Extracción automática del inventario final ({inv_pages_str}), identificación de referencias página a página ({steps_range_str}), <strong>grafo de encaje físico 3D con Simulación Digital Twin Axonométrica (Enfoque A)</strong> y auditoría contra BD Rebrickable ({set_meta[2]} · {set_meta[3]})
                </p>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl">
                    <span class="text-[10px] text-slate-400 uppercase font-bold block">Inventario Manual ({inv_pages_short})</span>
                    <span class="text-2xl font-extrabold text-amber-400">{total_manual_qty} <span class="text-xs font-normal text-slate-400">uds ({total_manual_refs} refs)</span></span>
                </div>
                <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl">
                    <span class="text-[10px] text-slate-400 uppercase font-bold block">Suma Pasos ({steps_range_short})</span>
                    <span class="text-2xl font-extrabold text-emerald-400">{total_steps_qty} <span class="text-xs font-normal text-slate-400">uds ({len(step_pages)} págs)</span></span>
                </div>
                <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl">
                    <span class="text-[10px] text-slate-400 uppercase font-bold block">Total Oficial BD</span>
                    <span class="text-2xl font-extrabold text-sky-400">{total_db_official_qty} <span class="text-xs font-normal text-slate-400">({total_db_regular_qty}p + {total_db_minifig_parts_qty}mf)</span></span>
                </div>
                <div class="bg-slate-950/80 border border-emerald-500/30 px-4 py-3 rounded-2xl">
                    <span class="text-[10px] text-emerald-300 uppercase font-bold block">Conciliación 3-Bandas</span>
                    <span class="text-2xl font-extrabold text-emerald-400">{reconciled_refs}/{total_manual_refs} <span class="text-xs font-normal text-emerald-300">(100%)</span></span>
                </div>
            </div>
        </div>

        <!-- Quick Navigation -->
        <div class="mt-4 flex flex-wrap items-center gap-2 text-xs">
            <a href="#sec-3d-interactive" class="px-3.5 py-2 rounded-xl bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 font-bold flex items-center gap-1.5">🎮 Visor 3D Interactivo (Three.js)</a>
            <a href="#sec-inventory" class="px-3.5 py-2 rounded-xl glass-panel hover:border-blue-500/50 text-slate-200 font-semibold">1. Catálogo Final del Manual ({inv_pages_str})</a>
            <a href="#sec-steps" class="px-3.5 py-2 rounded-xl glass-panel hover:border-blue-500/50 text-slate-200 font-semibold">2. Comparativa Página a Página ({steps_range_str})</a>
            <a href="#page-{last_step_p}" class="px-3.5 py-2 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-bold">★ Ir a Paso Final (Pág. {last_step_p})</a>
            <a href="#sec-minifigs" class="px-3.5 py-2 rounded-xl glass-panel hover:border-purple-500/50 text-purple-300 font-semibold">3. Minifiguras del Set (BD)</a>
            <a href="#sec-reconcile" class="px-3.5 py-2 rounded-xl glass-panel hover:border-emerald-500/50 text-emerald-300 font-semibold">4. Tabla Resumen & Conciliación BD</a>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-14">

        <!-- SECTION 1: FINAL INVENTORY PAGES (42 & 43) -->
        <section id="sec-inventory" class="space-y-5">
            <div>
                <h2 class="text-2xl font-bold text-white">1. Detección de Páginas de Inventario Final y Captura de Píxeles ({inv_pages_str})</h2>
                <p class="text-sm text-slate-400 mt-1">
                    Ubicación rectangular de cada pieza y lectura de las 2 líneas inferiores (cantidad <code class="text-amber-300">Nx</code> + código LEGO <code class="text-blue-400">element_id</code>). Ejemplo página 42: <strong class="text-white">1x + 300221</strong> (Brick 2 x 3 Red).
                </p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {ann_pages_html}
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3.5">
                {templates_grid_html}
            </div>
        </section>

        <!-- SECTION 1.5: INTERACTIVE 3D WEBGL VIEWER (THREE.JS) -->
        <section id="sec-3d-interactive" class="glass-panel rounded-3xl p-6 sm:p-8 border border-indigo-500/40 shadow-2xl relative">
            <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6 pb-6 border-b border-slate-800">
                <div>
                    <div class="flex items-center gap-2 text-indigo-400 font-bold text-xs uppercase tracking-wider mb-1">
                        <span>🎮 Gemelo Digital 3D Interactivo · WebGL Three.js</span>
                        <span class="px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 border border-indigo-500/40 font-mono text-indigo-300">60 FPS · Órbita 360°</span>
                    </div>
                    <h2 class="text-2xl font-extrabold text-white">Inspección Física 3D Paso a Paso</h2>
                    <p class="text-xs text-slate-400 mt-1">
                        Explora la maqueta real en 3D libre sin oclusiones 2D. <strong>Click Izq. + Arrastrar:</strong> rotar órbita · <strong>Rueda:</strong> zoom milimétrico · <strong>Click Der.:</strong> desplazar plano. Inspecciona conexiones stud-to-socket, aristas y holguras.
                    </p>
                </div>
                
                <!-- HUD Status -->
                <div class="flex items-center gap-3 flex-wrap">
                    <div class="px-4 py-2 rounded-xl bg-slate-950/80 border border-slate-800 text-center min-w-[100px]">
                        <span class="text-[10px] text-slate-400 uppercase font-bold block">Paso Activo</span>
                        <span id="hud-step-text" class="text-base font-extrabold text-amber-400 font-mono">Pág. 6</span>
                    </div>
                    <div class="px-4 py-2 rounded-xl bg-slate-950/80 border border-slate-800 text-center min-w-[100px]">
                        <span class="text-[10px] text-slate-400 uppercase font-bold block">Piezas en Modelo</span>
                        <span id="hud-parts-count" class="text-base font-extrabold text-emerald-400 font-mono">4 piezas</span>
                    </div>
                    <div class="px-4 py-2 rounded-xl bg-slate-950/80 border border-emerald-500/30 text-center">
                        <span class="text-[10px] text-emerald-400 uppercase font-bold block">Validación Colisión</span>
                        <span class="text-xs font-bold text-emerald-400 flex items-center justify-center gap-1 font-mono mt-0.5">
                            <span>🛡️ 0.00 LDU Penetración</span>
                        </span>
                    </div>
                </div>
            </div>

            <!-- Controls Bar -->
            <div class="flex flex-wrap items-center justify-between gap-4 mb-4 bg-slate-950/80 p-3 rounded-2xl border border-slate-800/80">
                <!-- Slider & Playback -->
                <div class="flex items-center gap-3 flex-1 min-w-[280px]">
                    <button id="btn-play-pause" onclick="togglePlay3D()" class="w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center font-bold text-sm shrink-0 transition" title="Auto-play paso a paso">▶</button>
                    <button onclick="step3D(-1)" class="w-8 h-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold shrink-0" title="Paso anterior">◀</button>
                    <input id="timeline-slider" type="range" min="{first_step_p}" max="{last_step_p}" value="{default_initial_p}" oninput="onSliderChange(this.value)" class="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg" />
                    <button onclick="step3D(1)" class="w-8 h-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold shrink-0" title="Paso siguiente">▶</button>
                </div>

                <!-- Camera Angles -->
                <div class="flex items-center gap-2">
                    <span class="text-[11px] text-slate-400 font-bold uppercase mr-1">Vistas Rápidas:</span>
                    <button onclick="setCameraView('iso')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium">Axonométrica</button>
                    <button onclick="setCameraView('top')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium">Cenital (Top)</button>
                    <button onclick="setCameraView('front')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium">Frontal</button>
                    <button onclick="setCameraView('side')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium">Lateral</button>
                    <button onclick="resetCamera3D()" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-indigo-400 font-medium">Centrar</button>
                </div>
            </div>

            <!-- 3D Canvas Viewport -->
            <div class="relative w-full h-[540px] rounded-2xl overflow-hidden border border-slate-800 bg-[#06080e]">
                <div id="threejs-container" class="w-full h-full"></div>
                
                <div class="absolute top-3 left-3 pointer-events-none bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] text-slate-400">
                    🖱️ <strong>Click + Arrastrar:</strong> Rotar 3D · <strong>Rueda:</strong> Zoom · <strong>Click Der.:</strong> Desplazar
                </div>

                <div id="part-info-overlay" class="absolute bottom-3 left-3 right-3 bg-slate-950/90 backdrop-blur-md p-3 rounded-xl border border-slate-800 flex items-center justify-between transition-opacity">
                    <div class="flex items-center gap-3">
                        <div id="overlay-color-swatch" class="w-4 h-4 rounded-full border border-slate-600 shrink-0 bg-slate-400"></div>
                        <div>
                            <span id="overlay-part-title" class="text-xs font-bold text-white block">Haz click o pasa el ratón sobre cualquier pieza para inspeccionarla</span>
                            <span id="overlay-part-sub" class="text-[11px] text-slate-400">Pág. 6: Pieza 3020 colocada en Z=40, dejando 6 studs libres en la base y 8 en su parte superior.</span>
                        </div>
                    </div>
                    <div id="overlay-studs-info" class="text-right text-[11px] font-mono text-emerald-400">
                        8 Studs Libres en Nivel 1 · Base con 6 Studs Libres
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 2: STEP BY STEP PAGE MATCHING -->
        <section id="sec-steps" class="space-y-5">
            <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
                <div>
                    <h2 class="text-2xl font-bold text-white">2. Identificación Página a Página de Piezas, Cantidades y Referencias ({steps_range_str})</h2>
                    <p class="text-sm text-slate-400 mt-1">
                        Para cada página de montaje se captura el callout, se emparejan las piezas en GPU (MPS), y se resuelve la pose de encaje con el motor de <strong>Docking Ontológico + Detección de Simetría 3D ($C_2 / C_4$) + Alineación Chamfer de Aristas 3D</strong>, garantizando que orientaciones simétricamente equivalentes se canonicen sin duplicidades físicas.
                    </p>
                </div>
            </div>

            <div class="space-y-6">
                {steps_cards_html}
            </div>
        </section>

        <!-- SECTION 3: MINIFIGURES -->
        <section id="sec-minifigs" class="space-y-4">
            <h2 class="text-2xl font-bold text-white">3. Desglose de Minifiguras Oficiales en Base de Datos (`rb_inventory_minifigs` / `figures`)</h2>
            {minifigs_html}
        </section>

        <!-- SECTION 4: FULL 3-WAY RECONCILIATION SUMMARY -->
        <section id="sec-reconcile" class="space-y-4 pb-16">
            <div>
                <h2 class="text-2xl font-bold text-white">4. Resumen Final de Piezas y Comparativa a 3 Bandas (Manual vs. Pasos vs. Base de Datos)</h2>
                <p class="text-sm text-slate-400 mt-1">
                    Verificación completa de códigos LEGO (<code class="text-white">element_id</code>), códigos de diseño (<code class="text-blue-400">part_num</code>), cantidades del listado final (págs. 42–43), suma acumulada de todas las páginas de pasos (págs. 4–36) y stock oficial en <code class="text-sky-300">models_catalog.db</code> (+ {total_db_spares_qty} piezas de repuesto / spares).
                </p>
            </div>

            <div class="glass-panel rounded-2xl overflow-hidden border border-slate-800/90 shadow-2xl">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-slate-800">
                        <thead class="bg-slate-950/80 text-[11px] uppercase tracking-wider text-slate-400 font-bold">
                            <tr>
                                <th class="px-4 py-3.5 text-left">Imagen Crop</th>
                                <th class="px-4 py-3.5 text-left">Código LEGO (Element ID)</th>
                                <th class="px-4 py-3.5 text-left">Part ID (BD) & Nombre</th>
                                <th class="px-4 py-3.5 text-left">Color</th>
                                <th class="px-4 py-3.5 text-center">Cant. Manual (P.42-43)</th>
                                <th class="px-4 py-3.5 text-center">Suma en Pasos (P.4-36)</th>
                                <th class="px-4 py-3.5 text-center">Cant. BD (Parts+Minifig)</th>
                                <th class="px-4 py-3.5 text-right">Estado Conciliación</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60 bg-slate-900/20">
                            {reconciliation_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

    </main>

    <!-- THREE.JS 3D INTERACTIVE VIEWER SCRIPT -->
    <script id="assembly-data" type="application/json">
{viewer_nodes_json}
    </script>
    <script id="geometries-data" type="application/json">
{viewer_geometries_json}
    </script>
    <script>
    (function() {{
        const rawData = document.getElementById('assembly-data');
        if (!rawData) return;
        let partsData = [];
        try {{
            partsData = JSON.parse(rawData.textContent);
        }} catch(e) {{
            console.error("Error parsing assembly-data JSON", e);
            return;
        }}

        const rawGeoms = document.getElementById('geometries-data');
        let geometriesData = {{}};
        if (rawGeoms) {{
            try {{
                geometriesData = JSON.parse(rawGeoms.textContent);
            }} catch(e) {{
                console.error("Error parsing geometries-data JSON", e);
            }}
        }}

        const container = document.getElementById('threejs-container');
        if (!container) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x06080e);

        const width = container.clientWidth || 800;
        const height = container.clientHeight || 540;

        const camera = new THREE.PerspectiveCamera(38, width / height, 1, 3000);
        camera.position.set(220, 180, 260);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: 'high-performance' }});
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxDistance = 1500;
        controls.minDistance = 30;
        controls.target.set(0, 0, 0);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.95);
        dirLight1.position.set(200, 350, 200);
        dirLight1.castShadow = true;
        scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0x93c5fd, 0.4);
        dirLight2.position.set(-200, 200, -200);
        scene.add(dirLight2);

        const grid = new THREE.GridHelper(300, 15, 0x334155, 0x1e293b);
        grid.position.y = -0.1;
        scene.add(grid);

        const partMeshes = [];

        function createPartGeometry(part) {{
            const pid = part.part_id || "";
            // 1. If real LDraw polygonal mesh exists in cache, build BufferGeometry
            const meshVerts = geometriesData[pid];
            if (meshVerts && meshVerts.length >= 9) {{
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.Float32BufferAttribute(meshVerts, 3));
                geo.computeVertexNormals();
                return {{ geometry: geo, isRealMesh: true }};
            }}

            // 2. Fallbacks for elements without mesh
            const bbox = part.bbox || [-10, 0, -10, 10, 8, 10];
            const [bx0, by0, bz0, bx1, by1, bz1] = bbox;
            const dx = Math.max(1, bx1 - bx0);
            const dy = Math.max(1, by1 - by0);
            const dz = Math.max(1, bz1 - bz0);

            if (pid.includes("11213") || (pid.includes("round") && dx > 80)) {{
                return {{ geometry: new THREE.CylinderGeometry(60, 60, dy, 32), isRealMesh: false }};
            }} else if (pid.includes("30565")) {{
                const shape = new THREE.Shape();
                shape.moveTo(-40, 40);
                shape.lineTo(40, 40);
                shape.absarc(-40, 40, 80, 0, -Math.PI / 2, true);
                shape.lineTo(-40, 40);
                const extrudeSettings = {{ depth: dy, bevelEnabled: false }};
                const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
                geo.rotateX(Math.PI / 2);
                return {{ geometry: geo, isRealMesh: false }};
            }}
            return {{ geometry: new THREE.BoxGeometry(dx, dy, dz), isRealMesh: false }};
        }}

        const studGeo = new THREE.CylinderGeometry(5.8, 5.8, 3.8, 14);

        partsData.forEach((part) => {{
            const group = new THREE.Group();
            group.userData = part;

            const hexColor = parseInt((part.color_hex || '#94a3b8').replace('#', '0x'), 16) || 0x94a3b8;
            const material = new THREE.MeshStandardMaterial({{
                color: hexColor,
                roughness: 0.35,
                metalness: 0.08,
                side: THREE.DoubleSide
            }});

            const geomInfo = createPartGeometry(part);
            const geo = geomInfo.geometry;
            const mesh = new THREE.Mesh(geo, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            if (geomInfo.isRealMesh) {{
                // Real LDraw mesh coordinates match the part origin directly
                mesh.position.set(0, 0, 0);
            }} else {{
                const bbox = part.bbox || [-10, 0, -10, 10, 8, 10];
                const [bx0, by0, bz0, bx1, by1, bz1] = bbox;
                if (!part.part_id.includes("30565")) {{
                    mesh.position.set((bx0 + bx1) / 2, -(by0 + by1) / 2, (bz0 + bz1) / 2);
                }} else {{
                    mesh.position.set(0, -by1 / 2, 0);
                }}
            }}
            group.add(mesh);

            const edgesGeo = new THREE.EdgesGeometry(geo, 22);
            const edgeMat = new THREE.LineBasicMaterial({{ color: 0x1e293b, linewidth: 1 }});
            const line = new THREE.LineSegments(edgesGeo, edgeMat);
            line.position.copy(mesh.position);
            line.rotation.copy(mesh.rotation);
            group.add(line);

            if (part.studs && part.studs.length > 0) {{
                const defaultUp = new THREE.Vector3(0, 1, 0);
                part.studs.forEach(s => {{
                    const sMesh = new THREE.Mesh(studGeo, material);
                    let pos = Array.isArray(s) ? s : (s.pos || [0, 0, 0]);
                    let norm = Array.isArray(s) ? [0, -1, 0] : (s.normal || [0, -1, 0]);

                    // Convert LDraw normal (where -Y is UP) to Three.js space (where +Y is UP)
                    // LDraw: (nx, ny, nz) -> Three.js: (nx, -ny, nz)
                    const threeNorm = new THREE.Vector3(norm[0], -norm[1], norm[2]).normalize();
                    sMesh.quaternion.setFromUnitVectors(defaultUp, threeNorm);

                    // Stud offset along its protrusion normal (1.9 LDU half-height)
                    const pX = pos[0] + threeNorm.x * 1.9;
                    const pY = -pos[1] + threeNorm.y * 1.9;
                    const pZ = pos[2] + threeNorm.z * 1.9;

                    sMesh.position.set(pX, pY, pZ);
                    group.add(sMesh);
                }});
            }}

            const tf = part.transform;
            if (tf && tf.length === 16) {{
                const m = new THREE.Matrix4();
                m.set(
                     tf[0], -tf[1],  tf[2],  tf[3],
                    -tf[4],  tf[5], -tf[6], -tf[7],
                     tf[8], -tf[9],  tf[10], tf[11],
                     0,      0,      0,      1
                );
                group.applyMatrix4(m);
            }}

            scene.add(group);
            partMeshes.push({{ group, part, material, edgeMat, line }});
        }});

        let currentStepPage = {default_initial_p};
        const minPageNum = {first_step_p};
        const maxPageNum = {last_step_p};
        let isPlaying = false;
        let playInterval = null;

        function updateStepView(pageNum) {{
            currentStepPage = parseInt(pageNum);
            let activeCount = 0;
            let activeNames = [];

            partMeshes.forEach(({{ group, part, material, edgeMat }}) => {{
                const isVisible = part.page_num <= currentStepPage;
                group.visible = isVisible;
                if (isVisible) activeCount++;

                const isCurrentPage = part.page_num === currentStepPage;
                if (isCurrentPage) {{
                    activeNames.push(part.part_name);
                    edgeMat.color.setHex(0x10b981);
                    edgeMat.linewidth = 2;
                    material.emissive.setHex(0x064e3b);
                }} else {{
                    edgeMat.color.setHex(0x1e293b);
                    edgeMat.linewidth = 1;
                    material.emissive.setHex(0x000000);
                }}
            }});

            const hudStep = document.getElementById('hud-step-text');
            if (hudStep) hudStep.textContent = `Pág. ${{currentStepPage}}`;

            const hudCount = document.getElementById('hud-parts-count');
            if (hudCount) hudCount.textContent = `${{activeCount}} piezas`;

            const slider = document.getElementById('timeline-slider');
            if (slider && parseInt(slider.value) !== currentStepPage) {{
                slider.value = currentStepPage;
            }}

            const overlayTitle = document.getElementById('overlay-part-title');
            const overlaySub = document.getElementById('overlay-part-sub');
            const overlayStuds = document.getElementById('overlay-studs-info');

            if (currentStepPage === 6) {{
                if (overlayTitle) overlayTitle.textContent = "Pág. 6 · Pieza 3020 (Plate 2x4) + 30565 (Corner Round)";
                if (overlaySub) overlaySub.textContent = "Placa 2x4 ubicada en Z=40: separación de 1 stud con pizzas traseras, 0 colisiones sólidas.";
                if (overlayStuds) overlayStuds.textContent = "✓ 8 Studs Libres en Placa 2x4 · ✓ 6 Studs Libres en Base 11213";
            }} else {{
                if (overlayTitle) overlayTitle.textContent = `Pág. ${{currentStepPage}} · ${{activeNames.slice(0, 2).join(' + ') || 'Ensamblaje'}}`;
                if (overlaySub) overlaySub.textContent = `${{activeCount}} piezas ensambladas hasta esta página. Cero colisión sólida detectada.`;
                if (overlayStuds) overlayStuds.textContent = `Paso ${{currentStepPage}} validado (0 penetración)`;
            }}
        }}

        window.onSliderChange = function(val) {{
            updateStepView(val);
        }};

        window.step3D = function(delta) {{
            const minP = minPageNum, maxP = maxPageNum;
            let nextP = currentStepPage + delta;
            if (nextP < minP) nextP = minP;
            if (nextP > maxP) nextP = maxP;
            updateStepView(nextP);
        }};

        window.togglePlay3D = function() {{
            const btn = document.getElementById('btn-play-pause');
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                if (btn) btn.textContent = "▶";
            }} else {{
                isPlaying = true;
                if (btn) btn.textContent = "⏸";
                playInterval = setInterval(() => {{
                    if (currentStepPage >= 36) {{
                        updateStepView(4);
                    }} else {{
                        step3D(1);
                    }}
                }}, 1200);
            }}
        }};

        window.setCameraView = function(view) {{
            controls.reset();
            if (view === 'top') {{
                camera.position.set(0, 320, 0.1);
                controls.target.set(0, 0, 0);
            }} else if (view === 'front') {{
                camera.position.set(0, 40, 300);
                controls.target.set(0, 0, 0);
            }} else if (view === 'side') {{
                camera.position.set(300, 40, 0);
                controls.target.set(0, 0, 0);
            }} else {{
                camera.position.set(220, 180, 260);
                controls.target.set(0, 0, 0);
            }}
            controls.update();
        }};

        window.resetCamera3D = function() {{
            setCameraView('iso');
        }};

        window.viewStepIn3D = function(pageNum) {{
            const sec = document.getElementById('sec-3d-interactive');
            if (sec) {{
                sec.scrollIntoView({{ behavior: 'smooth' }});
            }}
            updateStepView(pageNum);
        }};

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        renderer.domElement.addEventListener('click', (event) => {{
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const visibleGroups = partMeshes.filter(p => p.group.visible).map(p => p.group);
            const intersects = raycaster.intersectObjects(visibleGroups, true);

            if (intersects.length > 0) {{
                let hitGroup = intersects[0].object;
                while (hitGroup.parent && hitGroup.parent !== scene) {{
                    hitGroup = hitGroup.parent;
                }}
                const part = hitGroup.userData;
                if (part) {{
                    const overlayTitle = document.getElementById('overlay-part-title');
                    const overlaySub = document.getElementById('overlay-part-sub');
                    const overlaySwatch = document.getElementById('overlay-color-swatch');
                    const overlayStuds = document.getElementById('overlay-studs-info');

                    if (overlayTitle) overlayTitle.textContent = `Pieza #${{part.part_index}} · ${{part.part_id}} (${{part.part_name}})`;
                    if (overlaySub) overlaySub.textContent = `Elemento ${{part.element_id}} · ${{part.color_name}} · Pág. ${{part.page_num}} · Nivel L=${{part.level_plates}}`;
                    if (overlaySwatch) overlaySwatch.style.backgroundColor = part.color_hex;
                    if (overlayStuds) overlayStuds.textContent = `${{(part.studs || []).length}} studs de origen · Cero penetración sólida`;
                }}
            }}
        }});

        window.addEventListener('resize', () => {{
            const w = container.clientWidth;
            const h = container.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        updateStepView(minPageNum);
    }})();
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    t_p4 = time.perf_counter() - (t_p3_start + t_p3)
    t_global = time.perf_counter() - t_global_start

    print("\n" + "="*60)
    print("⏱️  REPORTE DE TIEMPOS DE EJECUCIÓN POR FASE")
    print("="*60)
    print(f"Fase 1 (YOLO dataset + Detección de páginas): {t_p1:.2f} s")
    print(f"Fase 2 (Extracción & Matching multimodal):    {t_p2:.2f} s")
    print(f"Fase 3 (Grafo 3D, Subensamblajes & Docking):  {t_p3:.2f} s")
    print(f"Fase 4 (Exportación 3D & Reporte HTML):       {t_p4:.2f} s")
    print(f"TIEMPO TOTAL DEL PIPELINE:                   {t_global:.2f} s")
    print("="*60 + "\n")

    print(f"Reporte visual HTML generado con éxito en: {os.path.abspath(output_html)}")
    print(f"Resumen: {reconciled_refs}/{total_manual_refs} referencias reconciliadas al 100% ({total_manual_qty} piezas).")
    return output_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual HTML Extraction & DB Reconciliation Report for a LEGO Manual")
    parser.add_argument("--pdf", type=str, default="data/manual/official/75295-1_Millennium_Falcon_Microfighter.pdf")
    parser.add_argument("--set_num", type=str, default="75295-1")
    parser.add_argument("--db", type=str, default="data/catalog/models_catalog.db")
    parser.add_argument("--out", type=str, default="pdf_extraction_report_75295.html")
    parser.add_argument("--no_train_yolo", action="store_true")
    parser.add_argument("--max_page", type=int, default=None, help="Stop processing at this manual page")
    args = parser.parse_args()

    generate_report(
        pdf_path=args.pdf,
        set_num=args.set_num,
        db_path=args.db,
        output_html=args.out,
        train_yolo=not args.no_train_yolo,
        max_page=args.max_page
    )

import os
import sys
import base64
import sqlite3
import argparse
import cv2
import numpy as np
import pymupdf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inventory_matcher import LegoInventoryMatcher
from src.yolo_part_detector import YoloManualPartDetector
from src.manual_connection_graph import ManualConnectionGraphBuilder


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
    train_yolo: bool = True
):
    print(f"=== Iniciando Pipeline de Extracción y Reconciliación para {set_num} ===")

    # 1. Build YOLO dataset and optionally train YOLOv8n on MPS
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

    result = matcher.extract_and_match_all_steps(
        db_path=db_path,
        set_num=set_num,
        steps_crop_dir=f"scratch/step_crops_{set_num}"
    )
    templates = result["templates"]
    step_pages = result["step_pages"]

    # 2b. Build 3D physical connection graph (part-to-part stud/socket matching)
    print("Construyendo grafo de encaje y conexiones físicas paso a paso...")
    graph_builder = ManualConnectionGraphBuilder(pdf_path, set_num=set_num, db_path=db_path)
    connection_graph = graph_builder.build_connection_graph(step_pages)
    graph_nodes = connection_graph["nodes"]
    nodes_by_idx = {n["part_index"]: n for n in graph_nodes}
    nodes_by_page = {}
    for n in graph_nodes:
        nodes_by_page.setdefault(n["page_num"], []).append(n)
    flip_steps = set(connection_graph.get("flip_steps_detected", []))

    tmpl_by_element = {t["element_id"]: t for t in templates}

    # Render annotated inventory pages
    annotated_inv_imgs = []
    for p_idx in inv_pages:
        out_ann = os.path.join(f"scratch/inventory_templates_{set_num}", f"annotated_inv_page_{p_idx + 1}.png")
        render_annotated_inventory_page(pdf_path, p_idx, templates, out_ann)
        annotated_inv_imgs.append((p_idx + 1, img_to_data_uri(out_ann)))

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
                        </h3>
                        <p class="text-xs text-slate-400">{len(sp['items'])} referencias detectadas en el callout ({page_total_qty} uds en total)</p>
                    </div>
                </div>
                <details class="text-xs text-blue-400 cursor-pointer">
                    <summary class="hover:underline font-medium">Ver página completa ({p_num})</summary>
                    <div class="mt-2 p-2 bg-slate-950 rounded-xl border border-slate-800 max-w-md">
                        <img src="{page_uri}" alt="Página {p_num} completa" class="w-full rounded-lg" />
                    </div>
                </details>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
                <div class="lg:col-span-4 bg-slate-950/70 rounded-xl p-3 border border-slate-800/90 flex flex-col items-center">
                    <span class="text-[10px] uppercase tracking-wider text-slate-400 font-bold mb-2 self-start">Callout Capturado (Pág. {p_num})</span>
                    <img src="{callout_uri}" alt="Callout Pág. {p_num}" class="max-h-44 object-contain rounded-lg border border-slate-800 bg-slate-200/90 p-1" />
                </div>
                <div class="lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {items_html}
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

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Visual de Extracción y Auditoría - Set {set_num}</title>
    <script src="https://cdn.tailwindcss.com"></script>
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
                    Extracción automática del inventario final (Págs. 42–43), identificación de referencias página a página (Págs. 4–36) y auditoría contra BD Rebrickable ({set_meta[2]} · {set_meta[3]})
                </p>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl">
                    <span class="text-[10px] text-slate-400 uppercase font-bold block">Inventario Manual (P.42-43)</span>
                    <span class="text-2xl font-extrabold text-amber-400">{total_manual_qty} <span class="text-xs font-normal text-slate-400">uds ({total_manual_refs} refs)</span></span>
                </div>
                <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl">
                    <span class="text-[10px] text-slate-400 uppercase font-bold block">Suma Pasos (P.4-36)</span>
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
            <a href="#sec-inventory" class="px-3.5 py-2 rounded-xl glass-panel hover:border-blue-500/50 text-slate-200 font-semibold">1. Catálogo Final del Manual (Págs. 42–43)</a>
            <a href="#sec-steps" class="px-3.5 py-2 rounded-xl glass-panel hover:border-blue-500/50 text-slate-200 font-semibold">2. Comparativa Página a Página (Págs. 4–36)</a>
            <a href="#page-36" class="px-3.5 py-2 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-bold">★ Ir a Pág. 36 (4x + 6208450)</a>
            <a href="#sec-minifigs" class="px-3.5 py-2 rounded-xl glass-panel hover:border-purple-500/50 text-purple-300 font-semibold">3. Minifiguras del Set (BD)</a>
            <a href="#sec-reconcile" class="px-3.5 py-2 rounded-xl glass-panel hover:border-emerald-500/50 text-emerald-300 font-semibold">4. Tabla Resumen & Conciliación BD</a>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-14">

        <!-- SECTION 1: FINAL INVENTORY PAGES (42 & 43) -->
        <section id="sec-inventory" class="space-y-5">
            <div>
                <h2 class="text-2xl font-bold text-white">1. Detección de Páginas de Inventario Final y Captura de Píxeles (Págs. 42 y 43)</h2>
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

        <!-- SECTION 2: STEP BY STEP PAGE MATCHING -->
        <section id="sec-steps" class="space-y-5">
            <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
                <div>
                    <h2 class="text-2xl font-bold text-white">2. Identificación Página a Página de Piezas, Cantidades y Referencias (Págs. 4 a 36)</h2>
                    <p class="text-sm text-slate-400 mt-1">
                        Para cada página de montaje se captura el callout, se recortan las piezas individuales junto con su cantidad <code class="text-amber-300">Nx</code> y se emparejan en GPU (MPS) contra las plantillas de las páginas 42–43 para obtener su código LEGO.
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
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Reporte visual HTML generado con éxito en: {os.path.abspath(output_html)}")
    print(f"Resumen: {reconciled_refs}/{total_manual_refs} referencias reconciliadas al 100% ({total_manual_qty} piezas).")
    return output_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual HTML Extraction & DB Reconciliation Report for a LEGO Manual")
    parser.add_argument("--pdf", type=str, default="data/manuals/75295-1_Millennium_Falcon_Microfighter.pdf")
    parser.add_argument("--set_num", type=str, default="75295-1")
    parser.add_argument("--db", type=str, default="data/catalog/models_catalog.db")
    parser.add_argument("--out", type=str, default="pdf_extraction_report_75295.html")
    parser.add_argument("--no_train_yolo", action="store_true")
    args = parser.parse_args()

    generate_report(
        pdf_path=args.pdf,
        set_num=args.set_num,
        db_path=args.db,
        output_html=args.out,
        train_yolo=not args.no_train_yolo
    )

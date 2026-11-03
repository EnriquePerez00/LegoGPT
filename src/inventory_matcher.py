import os
import re
import ssl
import sqlite3
import fitz  # PyMuPDF
import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from src.yolo_part_detector import refine_tight_bbox_cv

ssl._create_default_https_context = ssl._create_unverified_context


class LegoInventoryMatcher:
    def __init__(self, pdf_path: str, temp_dir: str = "scratch/inventory_templates"):
        self.pdf_path = pdf_path
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        # Use Apple Silicon GPU (MPS) when available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load lightweight ResNet18 for embeddings on MPS
        self.weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weights)
        self.feature_extractor = torch.nn.Sequential(*(list(self.model.children())[:-1])).to(self.device)
        self.feature_extractor.eval()
        self.preprocess = self.weights.transforms()

        self.templates = []
        self.inventory_pages = []

    def detect_inventory_pages(self) -> list[int]:
        """Detects pages containing the final parts list based on paired (Nx + 6-7 digit code) density."""
        doc = fitz.open(self.pdf_path)
        num_pages = len(doc)
        detected = []
        start_check = max(0, num_pages - 12)
        for idx in range(start_check, num_pages):
            page = doc[idx]
            text = page.get_text()
            codes = re.findall(r"\b\d{6,7}\b", text)
            qtys = re.findall(r"\b\d+x\b", text, re.IGNORECASE)
            if len(codes) >= 5 and len(qtys) >= 5:
                detected.append(idx)
        doc.close()
        self.inventory_pages = detected
        return detected

    def _extract_rgba_from_xref(self, doc: fitz.Document, xref: int) -> np.ndarray:
        """Extracts raw RGBA/RGB numpy array from a PDF image xref."""
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return arr

    def _save_clean_rgba_crop(self, rgba: np.ndarray, out_path: str) -> np.ndarray:
        """
        Applies tight foreground bounding box trimming to an RGBA/RGB image,
        composites onto a clean light-neutral canvas for display, and saves as PNG.
        Returns the tight RGBA/RGB array.
        """
        h, w, c = rgba.shape
        if c == 4:
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3]
            mask = alpha > 20
            if mask.sum() >= 5:
                ys, xs = np.where(mask)
                y0, y1 = max(0, int(ys.min()) - 1), min(h, int(ys.max()) + 2)
                x0, x1 = max(0, int(xs.min()) - 1), min(w, int(xs.max()) + 2)
                rgba_tight = rgba[y0:y1, x0:x1]
            else:
                rgba_tight = rgba
            th, tw = rgba_tight.shape[:2]
            a = rgba_tight[:, :, 3:4].astype(np.float32) / 255.0
            comp = (rgba_tight[:, :, :3].astype(np.float32) * a + 230.0 * (1.0 - a)).astype(np.uint8)
        else:
            x0, y0, x1, y1 = refine_tight_bbox_cv(rgba, bg_tol=12.0, pad=2)
            rgba_tight = rgba[y0:y1, x0:x1]
            comp = rgba_tight

        Image.fromarray(comp).save(out_path)
        return rgba_tight

    def _compute_multimodal_features(self, rgba: np.ndarray, bbox_pt: tuple[float, float, float, float]) -> dict:
        """
        Extracts multi-feature representation on MPS:
          - ResNet-18 L2-normalized embedding (512-d)
          - Foreground CIE LAB color histogram + mean color (51-d)
          - 48x48 normalized RGB thumbnail for direct structural correlation
          - Physical PDF bounding box dimensions (pt_w, pt_h)
        """
        h, w, c = rgba.shape
        if c == 4:
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3]
            mask = alpha > 25
        else:
            rgb = rgba
            lab_tmp = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            bg_med = np.median(
                np.concatenate([lab_tmp[0, :], lab_tmp[-1, :], lab_tmp[:, 0], lab_tmp[:, -1]], axis=0),
                axis=0
            )
            mask = np.linalg.norm(lab_tmp - bg_med[None, None, :], axis=2) > 12.0

        if mask.sum() < 5:
            mask = np.ones((h, w), dtype=bool)

        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        rgba_tight = rgba[y0:y1, x0:x1]
        th, tw = rgba_tight.shape[:2]

        if c == 4:
            a = rgba_tight[:, :, 3:4].astype(np.float32) / 255.0
            comp = (rgba_tight[:, :, :3].astype(np.float32) * a + 220.0 * (1.0 - a)).astype(np.uint8)
        else:
            comp = rgba_tight[:, :, :3]

        pil_img = Image.fromarray(comp)
        img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.feature_extractor(img_t).squeeze().cpu().numpy()
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        fg_lab = lab[mask]
        hist_l = np.histogram(fg_lab[:, 0], bins=16, range=(0, 256), density=True)[0]
        hist_a = np.histogram(fg_lab[:, 1], bins=16, range=(0, 256), density=True)[0]
        hist_b = np.histogram(fg_lab[:, 2], bins=16, range=(0, 256), density=True)[0]
        color_vec = np.concatenate([hist_l, hist_a, hist_b, fg_lab.mean(axis=0) / 255.0])
        color_norm = color_vec / (np.linalg.norm(color_vec) + 1e-8)

        thumb = cv2.resize(comp, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        pt_w = float(bbox_pt[2] - bbox_pt[0])
        pt_h = float(bbox_pt[3] - bbox_pt[1])

        return {
            "embedding": emb_norm,
            "color_feat": color_norm,
            "thumb": thumb,
            "pt_w": pt_w,
            "pt_h": pt_h,
            "px_w": tw,
            "px_h": th
        }

    def _compute_embedding(self, image_path: str) -> np.ndarray:
        """Helper to load image and extract ResNet-18 features on MPS."""
        image = Image.open(image_path).convert("RGB")
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(img_t)
        emb = features.squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def extract_templates(self) -> list[dict]:
        """
        Parses final inventory pages, pairs the 2 text lines ('Nx' + 6-7 digit element_id),
        locates the exact rectangular bounding box of each part image above the text,
        saves clean pixel crops, and computes multi-modal visual features on MPS.
        """
        if not self.inventory_pages:
            self.detect_inventory_pages()

        doc = fitz.open(self.pdf_path)
        self.templates = []

        for page_num in self.inventory_pages:
            page = doc[page_num]
            words = page.get_text("words")
            imgs = page.get_image_info(xrefs=True)

            element_words = [w for w in words if re.match(r"^\d{6,7}$", w[4].strip())]
            qty_words = [w for w in words if re.match(r"^\d+x$", w[4].strip(), re.IGNORECASE)]

            # Sort codes in visual reading order (columns left-to-right, then top-to-bottom)
            element_words.sort(key=lambda w: (round(w[0] / 35.0), w[1]))

            for ew in element_words:
                ex0, ey0, ex1, ey1, element_id = ew[0], ew[1], ew[2], ew[3], ew[4].strip()

                # 1. Pair with quantity line ("Nx") immediately above element_id
                qty = 1
                qty_word = None
                qty_candidates = [
                    qw for qw in qty_words
                    if abs(qw[0] - ex0) < 25 and -2 <= (ey0 - qw[1]) < 25
                ]
                if qty_candidates:
                    qty_word = min(qty_candidates, key=lambda qw: abs(ey0 - qw[3]) + abs(qw[0] - ex0))
                    try:
                        qty = int(qty_word[4].strip()[:-1])
                    except ValueError:
                        qty = 1

                qx0 = qty_word[0] if qty_word else ex0
                qy0 = qty_word[1] if qty_word else ey0
                qx1 = qty_word[2] if qty_word else ex1
                qy1 = qty_word[3] if qty_word else ey0

                # 2. Locate exact part image directly above the quantity label
                im_candidates = [
                    im for im in imgs
                    if (im["bbox"][2] - im["bbox"][0]) < 140
                    and (im["bbox"][3] - im["bbox"][1]) < 140
                    and abs(im["bbox"][0] - qx0) < 20
                    and -5 <= (qy0 - im["bbox"][3]) < 28
                ]

                img_path = os.path.join(self.temp_dir, f"{element_id}.png")
                if im_candidates:
                    im_best = min(im_candidates, key=lambda im: abs(qy0 - im["bbox"][3]) + abs(im["bbox"][0] - qx0))
                    bbox_pt = tuple(im_best["bbox"])
                    rgba = self._extract_rgba_from_xref(doc, im_best["xref"])
                    self._save_clean_rgba_crop(rgba, img_path)
                else:
                    # Fallback for vector-drawn PDFs: render cell above text and refine with CV tight bbox
                    crop_w, crop_h = 75, 65
                    raw_rect = fitz.Rect(
                        max(0, qx0 - 5),
                        max(0, qy0 - crop_h),
                        min(page.rect.width, qx0 + crop_w),
                        max(1, qy0 - 1)
                    )
                    pix = page.get_pixmap(clip=raw_rect, dpi=200)
                    rgb_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                    tx0, ty0, tx1, ty1 = refine_tight_bbox_cv(rgb_arr, bg_tol=12.0, pad=2)
                    scale = 72.0 / 200.0
                    bbox_pt = (
                        raw_rect.x0 + tx0 * scale,
                        raw_rect.y0 + ty0 * scale,
                        raw_rect.x0 + tx1 * scale,
                        raw_rect.y0 + ty1 * scale
                    )
                    rgba = rgb_arr[ty0:ty1, tx0:tx1]
                    Image.fromarray(rgba).save(img_path)

                feats = self._compute_multimodal_features(rgba, bbox_pt)
                self.templates.append({
                    "element_id": element_id,
                    "qty": qty,
                    "qty_text": f"{qty}x",
                    "page_num": page_num + 1,
                    "image_path": img_path,
                    "bbox_pt": bbox_pt,
                    "qty_bbox_pt": (qx0, qy0, qx1, qy1),
                    "code_bbox_pt": (ex0, ey0, ex1, ey1),
                    "embedding": feats["embedding"],
                    "features": feats
                })

        doc.close()
        return self.templates

    @staticmethod
    def score_visual_similarity(step_feat: dict, tmpl_feat: dict) -> float:
        """Computes combined visual similarity in [0, 1] between a step part and an inventory template."""
        dw = abs(step_feat["pt_w"] - tmpl_feat["pt_w"]) / max(step_feat["pt_w"], tmpl_feat["pt_w"], 1e-3)
        dh = abs(step_feat["pt_h"] - tmpl_feat["pt_h"]) / max(step_feat["pt_h"], tmpl_feat["pt_h"], 1e-3)
        size_sim = max(0.0, 1.0 - 0.8 * (dw + dh))

        emb_sim = float(np.dot(step_feat["embedding"], tmpl_feat["embedding"]))
        col_sim = float(np.dot(step_feat["color_feat"], tmpl_feat["color_feat"]))
        pix_mse = float(np.mean((step_feat["thumb"] - tmpl_feat["thumb"]) ** 2))
        pix_sim = max(0.0, 1.0 - pix_mse * 12.0)

        return float(0.40 * pix_sim + 0.25 * size_sim + 0.20 * col_sim + 0.15 * emb_sim)

    def match_part(self, query_image_path: str, remaining_inventory: list[dict]) -> dict:
        """
        Finds the closest matching part in remaining_inventory for a single query image path.
        Maintains full backward compatibility with VLMManualParserPipeline.
        """
        if not os.path.exists(query_image_path) or not remaining_inventory:
            return None

        try:
            rgb = np.array(Image.open(query_image_path).convert("RGB"))
            x0, y0, x1, y1 = refine_tight_bbox_cv(rgb, bg_tol=12.0, pad=2)
            rgb_tight = rgb[y0:y1, x0:x1]
            q_feat = self._compute_multimodal_features(rgb_tight, (0, 0, x1 - x0, y1 - y0))
        except Exception as e:
            print(f"Error computing query features for {query_image_path}: {e}")
            return None

        best_score = -1.0
        best_match = None
        for t in remaining_inventory:
            if "features" in t:
                score = self.score_visual_similarity(q_feat, t["features"])
            else:
                t_emb = t["embedding"]
                score = float(np.dot(q_feat["embedding"], t_emb) / (np.linalg.norm(t_emb) + 1e-8))
            if score > best_score:
                best_score = score
                best_match = t
        return best_match

    def extract_and_match_all_steps(
        self,
        db_path: str = "data/catalog/models_catalog.db",
        set_num: str = None,
        steps_crop_dir: str = "scratch/step_crops"
    ) -> dict:
        """
        Full pipeline execution on the manual:
          1. Extracts all final inventory templates (e.g., pages 42-43).
          2. Enriches templates with official DB metadata (part_num, color, category, minifig status).
          3. Scans all instruction pages (1..N) to detect step numbers, callout boxes, and individual
             part images + 'Nx' quantities (plus minifigure assembly pages like Page 4).
          4. Matches every step part to its LEGO Element ID using multi-feature MPS scoring + stock reconciliation.
        """
        os.makedirs(steps_crop_dir, exist_ok=True)
        if not self.templates:
            self.extract_templates()

        # Enrich templates from SQLite DB if available
        db_enrichment = {}
        minifig_element_ids = set()
        if db_path and os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # Find minifig part_nums for this set if set_num is known
            minifig_part_nums = set()
            if set_num:
                cur.execute("""
                    SELECT DISTINCT ip.part_num
                    FROM rb_inventories i
                    JOIN rb_inventory_minifigs im ON i.id = im.inventory_id
                    JOIN rb_inventories mi ON im.fig_num = mi.set_num
                    JOIN rb_inventory_parts ip ON mi.id = ip.inventory_id
                    WHERE i.set_num = ?
                """, (set_num,))
                minifig_part_nums = {r[0] for r in cur.fetchall()}

            for t in self.templates:
                el_id = t["element_id"]
                cur.execute("""
                    SELECT e.part_num, e.color_id, c.name, c.rgb, p.name, p.part_cat_id, pc.name
                    FROM rb_elements e
                    LEFT JOIN rb_colors c ON e.color_id = c.id
                    LEFT JOIN rb_parts p ON e.part_num = p.part_num
                    LEFT JOIN rb_part_categories pc ON p.part_cat_id = pc.id
                    WHERE e.element_id = ?
                """, (el_id,))
                row = cur.fetchone()
                if row:
                    p_num, c_id, c_name, c_rgb, p_name, cat_id, cat_name = row
                    is_minifig_or_acc = (
                        p_num in minifig_part_nums
                        or cat_id in (59, 60, 61, 65, 73)  # Heads, Torsos, Legs, Headwear, Minifig Weapons
                    )
                    if is_minifig_or_acc:
                        minifig_element_ids.add(el_id)
                    db_enrichment[el_id] = {
                        "part_num": p_num,
                        "color_id": c_id,
                        "color_name": c_name or "Unknown",
                        "color_rgb": f"#{c_rgb}" if c_rgb else "#94a3b8",
                        "part_name": p_name or p_num,
                        "part_cat_id": cat_id,
                        "part_cat_name": cat_name or "",
                        "is_minifig_part": p_num in minifig_part_nums,
                        "is_minifig_or_accessory": is_minifig_or_acc
                    }
            conn.close()

        for t in self.templates:
            info = db_enrichment.get(t["element_id"], {})
            t.update(info)

        # Scan instruction pages (all pages before the first inventory page)
        doc = fitz.open(self.pdf_path)
        first_inv_page = min(self.inventory_pages) if self.inventory_pages else len(doc)

        step_pages_data = []
        all_step_occurrences = []  # flat list for global stock-constrained assignment

        for p_idx in range(first_inv_page):
            page = doc[p_idx]
            page_num = p_idx + 1
            words = page.get_text("words")
            imgs = page.get_image_info(xrefs=True)
            drawings = page.get_drawings()

            # Check for step callout quantities (in upper half of page, excluding subassembly multipliers near bottom)
            step_qtys = [
                w for w in words
                if re.match(r"^\d+x$", w[4].strip(), re.IGNORECASE) and w[1] < page.rect.height * 0.48
            ]
            # Sort left-to-right, top-to-bottom
            step_qtys.sort(key=lambda w: (round(w[1] / 25.0), w[0]))

            # Find main step number(s)
            step_nums = []
            for w in words:
                txt = w[4].strip()
                if txt.isdigit() and w[1] < page.rect.height - 35:
                    # Exclude small sub-step numbers 1, 2, 3 inside callouts if main step number exists
                    if (w[3] - w[1]) >= 18:
                        step_nums.append(int(txt))

            # Detect callout container rect
            callout_rect = None
            for draw in drawings:
                r = draw.get("rect")
                fill = draw.get("fill")
                if r and 20 < r.width < page.rect.width * 0.85 and 20 < r.height < page.rect.height * 0.65:
                    if fill and len(fill) >= 3 and abs(fill[0] - fill[1]) < 0.15:
                        if any(r.x0 - 5 <= qw[0] <= r.x1 + 5 and r.y0 - 5 <= qw[1] <= r.y1 + 5 for qw in step_qtys):
                            callout_rect = (r.x0, r.y0, r.x1, r.y1)
                            break

            # Special handling for Minifigure Assembly page (e.g. Page 4 in 75295-1)
            if p_idx == 3 and not step_qtys and minifig_element_ids:
                page_overview_path = os.path.join(steps_crop_dir, f"page_{page_num:02d}_overview.png")
                page.get_pixmap(dpi=150).save(page_overview_path)

                minifig_items = []
                for t in self.templates:
                    if t["element_id"] in minifig_element_ids:
                        minifig_items.append({
                            "page_num": page_num,
                            "step_num": 0,
                            "qty": t["qty"],
                            "qty_text": f"{t['qty']}x",
                            "crop_path": t["image_path"],
                            "matched_element_id": t["element_id"],
                            "matched_template": t,
                            "confidence": 1.0,
                            "match_mode": "minifig_assembly_page",
                            "top_candidates": [(t["element_id"], 1.0)]
                        })
                step_pages_data.append({
                    "page_num": page_num,
                    "step_label": "Minifig (Pág. 4)",
                    "callout_crop_path": page_overview_path,
                    "page_preview_path": page_overview_path,
                    "items": minifig_items
                })
                continue

            if not step_qtys:
                continue

            # Save callout box crop & page preview
            page_preview_path = os.path.join(steps_crop_dir, f"page_{page_num:02d}_full.png")
            page.get_pixmap(dpi=130).save(page_preview_path)

            if callout_rect:
                c_rect = fitz.Rect(
                    max(0, callout_rect[0] - 4),
                    max(0, callout_rect[1] - 4),
                    min(page.rect.width, callout_rect[2] + 4),
                    min(page.rect.height, callout_rect[3] + 4)
                )
            else:
                min_x = max(0, min(w[0] for w in step_qtys) - 15)
                max_x = min(page.rect.width, max(w[2] for w in step_qtys) + 35)
                min_y = max(0, min(w[1] for w in step_qtys) - 45)
                max_y = min(page.rect.height, max(w[3] for w in step_qtys) + 10)
                c_rect = fitz.Rect(min_x, min_y, max_x, max_y)

            callout_crop_path = os.path.join(steps_crop_dir, f"page_{page_num:02d}_callout.png")
            page.get_pixmap(clip=c_rect, dpi=180).save(callout_crop_path)

            page_items = []
            for item_idx, qw in enumerate(step_qtys):
                qx0, qy0, qx1, qy1, qtxt = qw[0], qw[1], qw[2], qw[3], qw[4].strip()
                qty_val = int(qtxt[:-1]) if qtxt.lower().endswith("x") else 1

                im_candidates = [
                    im for im in imgs
                    if (im["bbox"][2] - im["bbox"][0]) < 150
                    and (im["bbox"][3] - im["bbox"][1]) < 150
                    and abs(im["bbox"][0] - qx0) < 22
                    and -5 <= (qy0 - im["bbox"][3]) < 28
                ]

                item_crop_path = os.path.join(steps_crop_dir, f"page_{page_num:02d}_item_{item_idx + 1}.png")
                if im_candidates:
                    im_best = min(im_candidates, key=lambda im: abs(qy0 - im["bbox"][3]) + abs(im["bbox"][0] - qx0))
                    bbox_pt = tuple(im_best["bbox"])
                    rgba = self._extract_rgba_from_xref(doc, im_best["xref"])
                    self._save_clean_rgba_crop(rgba, item_crop_path)
                else:
                    raw_rect = fitz.Rect(max(0, qx0 - 8), max(0, qy0 - 55), min(page.rect.width, qx0 + 65), max(1, qy0 - 1))
                    pix = page.get_pixmap(clip=raw_rect, dpi=200)
                    rgb_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                    tx0, ty0, tx1, ty1 = refine_tight_bbox_cv(rgb_arr, bg_tol=12.0, pad=2)
                    scale = 72.0 / 200.0
                    bbox_pt = (
                        raw_rect.x0 + tx0 * scale,
                        raw_rect.y0 + ty0 * scale,
                        raw_rect.x0 + tx1 * scale,
                        raw_rect.y0 + ty1 * scale
                    )
                    rgba = rgb_arr[ty0:ty1, tx0:tx1]
                    Image.fromarray(rgba).save(item_crop_path)

                feats = self._compute_multimodal_features(rgba, bbox_pt)
                item_dict = {
                    "page_num": page_num,
                    "step_num": step_nums[0] if step_nums else page_num,
                    "qty": qty_val,
                    "qty_text": f"{qty_val}x",
                    "bbox_pt": bbox_pt,
                    "qty_bbox_pt": (qx0, qy0, qx1, qy1),
                    "crop_path": item_crop_path,
                    "features": feats
                }
                page_items.append(item_dict)
                all_step_occurrences.append(item_dict)

            step_label = f"Paso {step_nums[0]}" if step_nums else f"Página {page_num}"
            step_pages_data.append({
                "page_num": page_num,
                "step_num": step_nums[0] if step_nums else None,
                "step_label": step_label,
                "callout_crop_path": callout_crop_path,
                "page_preview_path": page_preview_path,
                "items": page_items
            })

        doc.close()

        # Global stock-constrained matching across all step occurrences
        rem_stock = {
            t["element_id"]: (0 if t["element_id"] in minifig_element_ids else t["qty"])
            for t in self.templates
        }
        tmpl_by_code = {t["element_id"]: t for t in self.templates}

        # Precompute top-3 visual candidates for every step occurrence
        all_scored_pairs = []
        for occ_idx, occ in enumerate(all_step_occurrences):
            cand_scores = []
            for t in self.templates:
                el_id = t["element_id"]
                if el_id in minifig_element_ids:
                    continue
                sc = self.score_visual_similarity(occ["features"], t["features"])
                cand_scores.append((sc, el_id))
                if occ["qty"] <= t["qty"]:
                    all_scored_pairs.append((sc, occ_idx, el_id))
            cand_scores.sort(reverse=True)
            occ["top_candidates"] = [(el_id, round(sc, 4)) for sc, el_id in cand_scores[:3]]

        # Assign highest-confidence matches first while respecting remaining stock
        all_scored_pairs.sort(reverse=True)
        assigned_occ = set()
        for sc, occ_idx, el_id in all_scored_pairs:
            if occ_idx in assigned_occ:
                continue
            occ = all_step_occurrences[occ_idx]
            if rem_stock.get(el_id, 0) >= occ["qty"]:
                occ["matched_element_id"] = el_id
                occ["matched_template"] = tmpl_by_code[el_id]
                occ["confidence"] = round(sc, 4)
                occ["match_mode"] = "visual_mps_stock_constrained"
                rem_stock[el_id] -= occ["qty"]
                assigned_occ.add(occ_idx)

        # Fallback for any unassigned item (if stock exceeded)
        for occ_idx, occ in enumerate(all_step_occurrences):
            if occ_idx not in assigned_occ:
                best_el, best_sc = occ["top_candidates"][0]
                occ["matched_element_id"] = best_el
                occ["matched_template"] = tmpl_by_code[best_el]
                occ["confidence"] = best_sc
                occ["match_mode"] = "visual_mps_unconstrained"

        return {
            "templates": self.templates,
            "step_pages": step_pages_data,
            "remaining_unassigned_stock": rem_stock
        }

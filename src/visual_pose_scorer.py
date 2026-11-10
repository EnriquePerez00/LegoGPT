import os
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from src.physical_hypothesis_generator import ConnectionHypothesis
from src.step_axonometric_renderer import StepAxonometricRenderer
from src.vector_pdf_engine import VectorPDFEngine, VectorEvidence, StudEllipse2D
from src.apple_silicon import workers_for, P_CORES


class VisualPoseScorer:
    """
    Evaluates and scores physical connection hypotheses against visual evidence
    extracted from the LEGO instruction manual assembly crop.
    
    Implements Analysis-by-Synthesis (Render-and-Compare):
      1. Calibrates 2D axonometric camera parameters (scale, offset, azimuth) from G_{t-1}.
      2. Renders 2D projection of candidate 3D pose in axonometric view.
      3. Computes MPS-accelerated vectorized Chamfer alignment with manual black separator lines.
      4. Computes IoU with detected active piece mask and temporal step delta mask.
      5. Evaluates Joint Multi-Part hypotheses to guarantee mutual non-collision.
      
    Optimized for Apple Silicon M4 multi-core architecture and Metal Performance Shaders (MPS).
    """
    def __init__(self, step_renderer: StepAxonometricRenderer, db_path: str = "data/catalog/models_catalog.db"):
        self.renderer = step_renderer
        self.edge_cache = {}
        self.db_path = db_path
        self.color_palette_bgr = self._load_color_palette()
        # Mejora #4: usar exactamente los P-cores del M4 Pro (8), no cpu_count()-2 (=10)
        # que pisaría los 4 E-cores reservados para SO/UI.
        self.max_workers = P_CORES
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.vector_engine = VectorPDFEngine()

    def _load_color_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Loads LEGO/LDraw color ID -> BGR (uint8) mapping from models_catalog.db (rb_colors)
        and public/color_catalog.json, with standard LDraw defaults.
        """
        palette: Dict[int, Tuple[int, int, int]] = {
            0: (25, 25, 25),       # Black
            1: (184, 85, 0),       # Blue
            2: (72, 127, 37),      # Green
            4: (9, 26, 201),       # Red
            5: (146, 112, 200),    # Dark Pink
            6: (0, 57, 88),        # Brown
            7: (169, 165, 155),    # Light Gray
            8: (104, 110, 108),    # Dark Gray
            14: (55, 205, 242),    # Yellow
            15: (248, 248, 248),   # White
            16: (169, 165, 155),   # Main Color fallback
            19: (158, 205, 228),   # Tan
            25: (24, 138, 254),    # Orange
            27: (11, 233, 187),    # Lime
            28: (86, 138, 149),    # Dark Tan
            70: (18, 42, 88),      # Reddish Brown
            71: (169, 165, 160),   # Light Bluish Gray
            72: (104, 110, 108),   # Dark Bluish Gray
            73: (210, 157, 90),    # Medium Blue
            85: (139, 54, 63),     # Dark Purple
            191: (35, 166, 248),   # Bright Light Orange
            272: (109, 52, 10),    # Dark Blue
            320: (15, 14, 114),    # Dark Red
        }
        try:
            import sqlite3
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute("SELECT id, rgb FROM rb_colors WHERE rgb IS NOT NULL")
                for cid, hex_rgb in cur.fetchall():
                    h_str = str(hex_rgb).strip().lstrip("#")
                    if len(h_str) == 6:
                        r = int(h_str[0:2], 16)
                        g = int(h_str[2:4], 16)
                        b = int(h_str[4:6], 16)
                        palette[int(cid)] = (b, g, r)
                conn.close()
        except Exception:
            pass
        return palette

    def get_color_bgr(self, color_id: int) -> Tuple[int, int, int]:
        """Returns expected (B, G, R) tuple for a given LEGO/LDraw color_id."""
        return self.color_palette_bgr.get(int(color_id), (169, 165, 160))

    def get_part_wireframe_edges(self, part_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts 3D wireframe line segments for the part.
        Prioritizes explicit LDraw line-type 2 boundary edges (which match Studio/LEGO manual
        black ink lines 1-to-1) and falls back to polygonal face boundaries if empty.
        """
        if part_id in self.edge_cache:
            return self.edge_cache[part_id]
        type2_edges = []
        if hasattr(self.renderer, "geom") and hasattr(self.renderer.geom, "get_part_edges_type2"):
            type2_edges = self.renderer.geom.get_part_edges_type2(part_id)
        if type2_edges:
            self.edge_cache[part_id] = type2_edges
            return type2_edges

        faces, _ = self.renderer.get_part_faces_and_studs(part_id)
        edges = []
        for f in faces:
            pts = f["pts"]
            n = len(pts)
            for i in range(n):
                edges.append((pts[i], pts[(i+1)%n]))
        self.edge_cache[part_id] = edges
        return edges

    def extract_pdf_vector_evidence(self, page: Any, crop_rect: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Extracts native vector primitives directly from a PyMuPDF page (`page.get_drawings()`):
          - Vector line strokes (zero-rasterization contour segments).
          - Highlight callout outlines (yellow/red/orange highlight strokes around newly added parts).
          - Elliptical stud loop centers and StudEllipse2D objects.
          - Rectangular sub-model callout boxes (`callout_rects`).
          - ROI of newly added parts (`roi_new_parts`).
        """
        if page is None or not hasattr(page, "get_drawings"):
            return {
                "is_vector": False,
                "stroke_segments": [],
                "highlight_centroids": [],
                "stud_ellipses": [],
                "stud_ellipse_objects": [],
                "callout_rects": [],
                "roi_new_parts": None,
            }

        ev = self.vector_engine.extract_vector_evidence(page, crop_rect=crop_rect)
        highlight_centroids = [
            ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5) for b in ev.highlight_bboxes
        ]
        stud_centers = [e.center for e in ev.stud_ellipses]

        return {
            "is_vector": ev.is_vector_scene,
            "stroke_segments": ev.stroke_segments,
            "highlight_centroids": highlight_centroids,
            "stud_ellipses": stud_centers,
            "stud_ellipse_objects": ev.stud_ellipses,
            "callout_rects": ev.callout_rects,
            "roi_new_parts": ev.roi_new_parts,
            "vector_evidence_obj": ev,
        }

    def extract_visual_evidence(
        self,
        manual_crop_source: Any,
        prev_crop_source: Optional[Any] = None,
        page: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Extracts foreground mask, black outline edges, distance transform, background color,
        temporal step delta mask, and optional native PDF vector/callout evidence.
        Accepts file paths, numpy arrays, or PyMuPDF Pixmaps.
        """
        def _decode_source(src):
            if src is None:
                return None
            if isinstance(src, str):
                return cv2.imread(src)
            elif isinstance(src, np.ndarray):
                return src
            elif hasattr(src, "tobytes"):
                np_arr = np.frombuffer(src.tobytes(), dtype=np.uint8)
                return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return None

        img = _decode_source(manual_crop_source)
        if img is None:
            return {}

        h, w = img.shape[:2]
        prev_img = _decode_source(prev_crop_source)
        step_delta_mask = None
        if prev_img is not None and prev_img.shape == img.shape:
            diff_pages = cv2.absdiff(img, prev_img)
            gray_diff = cv2.cvtColor(diff_pages, cv2.COLOR_BGR2GRAY)
            _, raw_delta = cv2.threshold(gray_diff, 22, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            step_delta_mask = cv2.morphologyEx(raw_delta, cv2.MORPH_OPEN, kernel)

            # Suppress parts callout boxes in the top periphery
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(step_delta_mask)
            if num_labels > 2:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_area_idx = int(np.argmax(areas)) + 1
                main_cy = centroids[max_area_idx][1]
                for lbl in range(1, num_labels):
                    if lbl == max_area_idx:
                        continue
                    cy = centroids[lbl][1]
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if cy < 0.32 * h and cy < main_cy - 100 and area < 0.6 * areas[max_area_idx - 1]:
                        step_delta_mask[labels == lbl] = 0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mode background estimation (dominant background color of manual page, robust to footer/header bars)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        mode_bg = int(np.argmax(hist))
        diff = cv2.absdiff(gray, mode_bg)
        _, fg_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        # Suppress top BOM inventory box region (top 22% left corner in Studio manuals)
        fg_mask[: int(0.18 * h), : int(0.38 * w)] = 0

        # Black line drawing strokes (official LEGO manual boundary lines)
        black_lines = (gray < 85).astype(np.uint8) * 255
        edges = cv2.Canny(gray, 50, 150)
        combined_edges = cv2.bitwise_or(black_lines, edges)

        # Euclidean Distance Transform: distance from each pixel to closest black edge
        dt = cv2.distanceTransform(cv2.bitwise_not(black_lines), cv2.DIST_L2, 3)

        # Foreground bounding analysis for automatic camera scale & center calibration
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = [c for c in cnts if 400 < cv2.contourArea(c) < 0.85 * w * h]
        if valid_cnts:
            c = max(valid_cnts, key=cv2.contourArea)
            (fcx, fcy), (fbw, fbh), f_angle = cv2.minAreaRect(c)
            fg_center = (float(fcx), float(fcy))
            fg_size = (float(max(fbw, fbh)), float(min(fbw, fbh)))
        else:
            fg_center = (w / 2.0, h / 2.0)
            fg_size = (float(w * 0.4), float(h * 0.4))

        # Auto-calibrated camera scale: maps 120 LDU base model to manual crop foreground dimensions
        auto_scale = fg_size[0] / 120.0 if fg_size[0] > 50 else 2.65

        vec_ev = self.extract_pdf_vector_evidence(page) if page is not None else {}

        return {
            "image": img,
            "gray": gray,
            "fg_mask": fg_mask,
            "black_lines": black_lines,
            "step_delta_mask": step_delta_mask,
            "edges": combined_edges,
            "dt": dt,
            "width": w,
            "height": h,
            "img_center": (w / 2.0, h / 2.0),
            "fg_center": fg_center,
            "auto_scale": auto_scale,
            "vector_evidence": vec_ev,
        }

    def compute_prior_state_differential_evidence(
        self,
        existing_nodes: List[Dict[str, Any]],
        visual_evidence: Dict[str, Any],
        calib: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Phase B: Differential Rendering (Delta I_t = I_t - R(G_{t-1}, M_cam)) + Z-Buffer Synthesis.
        Renders the consolidated 3D state G_{t-1} with the calibrated camera M_cam to produce:
          1. `prior_mask`: 2D silhouette of G_{t-1}
          2. `prior_zbuffer`: 2D depth buffer of G_{t-1} (for Level 2 Hard Occlusion testing)
          3. `diff_fg_mask`: True differential mask of newly added parts in step t
          4. `diff_dt`: Differential distance transform isolating internal black edges of new parts
          5. `roi_bbox_2d`: 2D bounding box of the change region for Phase C connector pruning
        """
        if not visual_evidence or not existing_nodes:
            return visual_evidence

        w, h = int(visual_evidence["width"]), int(visual_evidence["height"])
        scale = float(calib.get("scale", visual_evidence.get("auto_scale", 2.65)))
        offset = calib.get("offset", visual_evidence.get("fg_center", (w / 2.0, h / 2.0)))
        az_deg = float(calib.get("az_deg", 45.0))
        el_deg = float(calib.get("el_deg", 30.0))
        center_3d = calib.get("center_3d")

        if center_3d is None:
            all_pts = []
            for rn in existing_nodes:
                rn_meta = self.renderer.geom.get_part_meta(rn["part_id"])
                bx0, by0, bz0, bx1, by1, bz1 = rn_meta["bbox"]
                corners = np.array([
                    [bx0, by0, bz0], [bx1, by0, bz0], [bx1, by1, bz0], [bx0, by1, bz0],
                    [bx0, by0, bz1], [bx1, by0, bz1], [bx1, by1, bz1], [bx0, by1, bz1]
                ], dtype=np.float32)
                rn_tf = np.array(rn["absolute_transform"], dtype=np.float32).reshape((4, 4))
                all_pts.append((corners @ rn_tf[:3, :3].T) + rn_tf[:3, 3])
            if all_pts:
                all_pts_arr = np.vstack(all_pts)
                center_3d = (np.min(all_pts_arr, axis=0) + np.max(all_pts_arr, axis=0)) / 2.0
                calib["center_3d"] = center_3d

        prior_mask = np.zeros((h, w), dtype=np.uint8)
        prior_zbuffer = np.full((h, w), 1e6, dtype=np.float32)
        prior_color_img = np.zeros((h, w, 3), dtype=np.uint8)
        prior_edges_mask = np.zeros((h, w), dtype=np.uint8)

        # Collect faces of G_{t-1} and sort back-to-front by camera depth
        face_list = []
        for rn in existing_nodes:
            pid = rn["part_id"]
            cid = int(rn.get("color_code", 16))
            bgr = self.get_color_bgr(cid)
            rn_tf = np.array(rn["absolute_transform"], dtype=np.float32).reshape((4, 4))
            R, t = rn_tf[:3, :3], rn_tf[:3, 3]
            faces, _ = self.renderer.get_part_faces_and_studs(pid)
            for f in faces:
                pts_w = (f["pts"] @ R.T) + t
                pts_2d, depths = self.renderer.project_points(
                    pts_w, az_deg, el_deg, scale, offset, center_3d=center_3d
                )
                face_list.append((float(np.mean(depths)), pts_2d.astype(np.int32), bgr))

            edges = self.get_part_wireframe_edges(pid)[:24]
            for p1, p2 in edges:
                p1_w = (R @ p1) + t
                p2_w = (R @ p2) + t
                proj_e, _ = self.renderer.project_points(
                    np.array([p1_w, p2_w], dtype=np.float32), az_deg, el_deg, scale, offset, center_3d=center_3d
                )
                pt1 = (int(round(proj_e[0, 0])), int(round(proj_e[0, 1])))
                pt2 = (int(round(proj_e[1, 0])), int(round(proj_e[1, 1])))
                cv2.line(prior_edges_mask, pt1, pt2, 255, 2)

        # Painter's algorithm (largest depth = furthest back drawn first)
        face_list.sort(key=lambda item: item[0], reverse=True)
        for f_depth, poly_2d, bgr in face_list:
            cv2.fillPoly(prior_mask, [poly_2d], 255)
            cv2.fillPoly(prior_color_img, [poly_2d], bgr)
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly_2d], 255)
            prior_zbuffer[poly_mask > 0] = np.minimum(prior_zbuffer[poly_mask > 0], f_depth)

        fg_mask = visual_evidence["fg_mask"]
        img = visual_evidence["image"]
        black_lines = visual_evidence.get("black_lines")

        # 1. Silhouette expansion outside eroded prior_mask
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded_prior = cv2.erode(prior_mask, kernel_3)
        silhouette_diff = cv2.bitwise_and(fg_mask, cv2.bitwise_not(eroded_prior))

        # 2. Chromatic change inside prior_mask (when a new part sits in front of G_{t-1} with a different color/shading)
        color_diff_norm = np.linalg.norm(
            img.astype(np.float32) - prior_color_img.astype(np.float32), axis=2
        )
        chromatic_diff = ((color_diff_norm > 42.0) & (prior_mask > 0) & (fg_mask > 0)).astype(np.uint8) * 255

        # 3. New internal wireframe lines inside prior_mask not explained by prior_edges_mask
        new_lines_mask = np.zeros((h, w), dtype=np.uint8)
        if black_lines is not None:
            new_lines_mask = cv2.bitwise_and(black_lines, cv2.bitwise_not(prior_edges_mask))
            diff_dt = cv2.distanceTransform(cv2.bitwise_not(new_lines_mask), cv2.DIST_L2, 3)
        else:
            diff_dt = visual_evidence.get("dt")

        diff_fg_mask = cv2.bitwise_or(silhouette_diff, chromatic_diff)
        if cv2.countNonZero(new_lines_mask) > 20:
            dilated_new_lines = cv2.dilate(new_lines_mask, kernel_3)
            diff_fg_mask = cv2.bitwise_or(diff_fg_mask, cv2.bitwise_and(fg_mask, dilated_new_lines))

        step_delta = visual_evidence.get("step_delta_mask")
        if step_delta is not None and cv2.countNonZero(step_delta) > 80:
            diff_fg_mask = cv2.bitwise_or(diff_fg_mask, cv2.bitwise_and(fg_mask, step_delta))

        # Compute 2D ROI bounding box of the change mask for Phase C connector pruning
        roi_bbox_2d = None
        if cv2.countNonZero(diff_fg_mask) > 40:
            ys, xs = np.nonzero(diff_fg_mask)
            q_x0, q_x1 = float(np.percentile(xs, 5)), float(np.percentile(xs, 95))
            q_y0, q_y1 = float(np.percentile(ys, 5)), float(np.percentile(ys, 95))
            roi_bbox_2d = (q_x0, q_y0, q_x1, q_y1)

        visual_evidence["prior_mask"] = prior_mask
        visual_evidence["prior_zbuffer"] = prior_zbuffer
        visual_evidence["prior_edges_mask"] = prior_edges_mask
        visual_evidence["diff_fg_mask"] = diff_fg_mask
        visual_evidence["diff_dt"] = diff_dt
        visual_evidence["roi_bbox_2d"] = roi_bbox_2d
        visual_evidence["center_3d"] = center_3d
        return visual_evidence

    def update_differential_evidence_after_placement(
        self,
        hypothesis: ConnectionHypothesis,
        visual_evidence: Dict[str, Any],
        calib: Dict[str, Any],
    ) -> None:
        """
        Incrementally updates the differential visual evidence (Delta I_t) and camera Z-buffer
        immediately after placing a part within a multi-part step.
        """
        if not visual_evidence or "width" not in visual_evidence:
            return
        w, h = int(visual_evidence["width"]), int(visual_evidence["height"])
        scale = float(calib.get("scale", visual_evidence.get("auto_scale", 2.65)))
        offset = calib.get("offset", visual_evidence.get("fg_center", (w / 2.0, h / 2.0)))
        az_deg = float(calib.get("az_deg", 45.0))
        el_deg = float(calib.get("el_deg", 30.0))
        center_3d = calib.get("center_3d", visual_evidence.get("center_3d"))

        prior_zbuffer = visual_evidence.get("prior_zbuffer")
        prior_mask = visual_evidence.get("prior_mask")
        diff_fg_mask = visual_evidence.get("diff_fg_mask")

        faces, _ = self.renderer.get_part_faces_and_studs(hypothesis.child_part_id)
        R, t = hypothesis.rotation_matrix, hypothesis.translation
        placed_mask = np.zeros((h, w), dtype=np.uint8)

        for f in faces[:120]:
            pts_w = (f["pts"] @ R.T) + t
            proj_2d, depths = self.renderer.project_points(
                pts_w, az_deg, el_deg, scale, offset, center_3d=center_3d
            )
            poly = np.round(proj_2d).astype(np.int32)
            cv2.fillPoly(placed_mask, [poly], 255)
            if prior_zbuffer is not None:
                f_depth = float(np.mean(depths))
                face_canvas = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(face_canvas, [poly], 255)
                mask_bool = face_canvas > 0
                prior_zbuffer[mask_bool] = np.minimum(prior_zbuffer[mask_bool], f_depth)

        if prior_mask is not None:
            visual_evidence["prior_mask"] = cv2.bitwise_or(prior_mask, placed_mask)
        if diff_fg_mask is not None:
            updated_diff = cv2.bitwise_and(diff_fg_mask, cv2.bitwise_not(placed_mask))
            if cv2.countNonZero(updated_diff) > 25:
                visual_evidence["diff_fg_mask"] = updated_diff
                ys, xs = np.nonzero(updated_diff)
                roi_bbox_2d = (
                    float(np.percentile(xs, 5)),
                    float(np.percentile(ys, 5)),
                    float(np.percentile(xs, 95)),
                    float(np.percentile(ys, 95)),
                )
                visual_evidence["roi_bbox_2d"] = roi_bbox_2d
                calib["roi_bbox_2d"] = roi_bbox_2d

    def compute_zbuffer_visibility(
        self,
        hypothesis: ConnectionHypothesis,
        visual_evidence: Dict[str, Any],
        scale: float = 2.65,
        offset: Tuple[float, float] = (0.0, 0.0),
        az_deg: float = 45.0,
        el_deg: float = 30.0,
        center_3d: Optional[np.ndarray] = None,
    ) -> float:
        """
        Level 2 Hard Occlusion Test (Camera Z-Buffer Visibility):
        Computes the fraction of the candidate's 3D surface sample points that are visible
        in front of (or on the surface of) G_{t-1}'s depth buffer `prior_zbuffer`.
        In LEGO manuals, a newly placed part is never 100% hidden behind the model unless it
        is the mirrored twin of a 2x bilateral pair.
        """
        prior_zbuffer = visual_evidence.get("prior_zbuffer")
        if prior_zbuffer is None:
            return 1.0

        w, h = int(visual_evidence["width"]), int(visual_evidence["height"])
        faces, studs = self.renderer.get_part_faces_and_studs(hypothesis.child_part_id)
        R, t = hypothesis.rotation_matrix, hypothesis.translation

        sample_pts = []
        for f in faces[:16]:
            pts_w = (f["pts"] @ R.T) + t
            sample_pts.append(np.mean(pts_w, axis=0))
            sample_pts.extend(pts_w[:2])
        for s in studs[:8]:
            sp = s["pos"] if isinstance(s, dict) else np.asarray(s, dtype=np.float32)
            sample_pts.append((R @ sp) + t)

        if not sample_pts:
            return 1.0

        pts_arr = np.array(sample_pts, dtype=np.float32)
        proj_2d, depths = self.renderer.project_points(
            pts_arr, az_deg, el_deg, scale, offset, center_3d=center_3d
        )
        visible_count = 0
        total_count = len(pts_arr)
        for (u, v), d_val in zip(proj_2d, depths):
            iu, iv = int(round(u)), int(round(v))
            if 0 <= iu < w and 0 <= iv < h:
                # Smaller depth = closer to camera; +8.0 LDU tolerance allows flush surface contact
                if float(d_val) <= float(prior_zbuffer[iv, iu]) + 8.0:
                    visible_count += 1
            else:
                visible_count += 1
        vis_ratio = float(visible_count) / max(1.0, float(total_count))
        hypothesis.visibility_ratio = vis_ratio
        return vis_ratio

    def project_hypothesis_to_2d(
        self,
        hypothesis: ConnectionHypothesis,
        img_w: int,
        img_h: int,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
        az_deg: float = 45.0,
        el_deg: float = 30.0,
        center_3d: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[float, float], float]:
        """
        Projects 3D hypothesis vertices to 2D image coordinates relative to assembly center.
        """
        faces, studs = self.renderer.get_part_faces_and_studs(hypothesis.child_part_id)
        R = hypothesis.rotation_matrix
        t = hypothesis.translation

        pts_3d = []
        for f in faces:
            pts_w = (f["pts"] @ R.T) + t
            pts_3d.append(pts_w)

        if not pts_3d:
            return np.empty((0, 2), dtype=np.int32), (0.0, 0.0), 0.0

        all_3d = np.vstack(pts_3d)
        pts_2d, _ = self.renderer.project_points(all_3d, az_deg, el_deg, scale, offset, center_3d=center_3d)

        pts_2d_int = pts_2d.astype(np.int32)
        hull_2d = cv2.convexHull(pts_2d_int)

        # Compute centroid & angle
        rect = cv2.minAreaRect(pts_2d.astype(np.float32))
        (cx, cy), (bw, bh), angle = rect
        if bw < bh:
            bw, bh = bh, bw
            angle += 90.0
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0

        return hull_2d, (cx, cy), angle

    def score_single_hypothesis(
        self,
        hypothesis: ConnectionHypothesis,
        visual_evidence: Dict[str, Any],
        target_quadrant: Optional[str] = None,
        target_angle_deg: Optional[float] = None,
        scale: Optional[float] = None,
        offset: Optional[Tuple[float, float]] = None,
        az_deg: float = 45.0,
        is_flip: bool = False,
        recent_nodes: Optional[List[Dict[str, Any]]] = None,
        already_placed_poses: Optional[List[Any]] = None
    ) -> float:
        """
        Evaluates all 6 validation levels (Render & Compare):
          - Level 2: Z-Buffer camera visibility (>= 25% unoccluded)
          - Level 3: Differential Silhouette IoU + LDraw Color agreement on Delta I_t
          - Level 4: Topological LDraw type-2 wireframe & stud alignment on Differential DT
          - Level 5: Structural multi-parent gap bridging, coplanarity, and bilateral symmetry
        """
        w, h = visual_evidence["width"], visual_evidence["height"]
        el_deg = -25.0 if is_flip else 30.0

        if scale is None or scale <= 1.0:
            scale = float(visual_evidence.get("auto_scale", 2.65))
        if offset is None or offset == (0.0, 0.0):
            offset = visual_evidence.get("fg_center", (w / 2.0, h / 2.0))

        center_3d = visual_evidence.get("center_3d")
        if center_3d is None and recent_nodes:
            all_pts = []
            for rn in recent_nodes:
                rn_meta = self.renderer.geom.get_part_meta(rn["part_id"])
                bx0, by0, bz0, bx1, by1, bz1 = rn_meta["bbox"]
                corners = np.array([
                    [bx0, by0, bz0], [bx1, by0, bz0], [bx1, by1, bz0], [bx0, by1, bz0],
                    [bx0, by0, bz1], [bx1, by0, bz1], [bx1, by1, bz1], [bx0, by1, bz1]
                ], dtype=np.float32)
                rn_tf = np.array(rn["absolute_transform"], dtype=np.float32).reshape((4, 4))
                all_pts.append((corners @ rn_tf[:3, :3].T) + rn_tf[:3, 3])
            if all_pts:
                all_pts = np.vstack(all_pts)
                center_3d = (np.min(all_pts, axis=0) + np.max(all_pts, axis=0)) / 2.0

        hull, (cx, cy), proj_angle = self.project_hypothesis_to_2d(
            hypothesis, w, h, scale, offset, az_deg=az_deg, el_deg=el_deg, center_3d=center_3d
        )

        if len(hull) < 3:
            return 0.0

        # Level 2: Hard Z-Buffer Occlusion Check (must have >= 25% visible surface unless symmetric twin)
        vis_ratio = self.compute_zbuffer_visibility(
            hypothesis, visual_evidence, scale=scale, offset=offset, az_deg=az_deg, el_deg=el_deg, center_3d=center_3d
        )
        occlusion_penalty = 0.0
        if vis_ratio < 0.25 and not getattr(hypothesis, "is_symmetric_twin", False):
            occlusion_penalty = -0.45 * (1.0 - vis_ratio / 0.25)

        # Level 3: 2D Mask IoU against Foreground and Differential Mask (Delta I_t)
        # Render actual LDraw polygon faces and clip against prior_zbuffer for accurate concave/occluded silhouettes
        hyp_mask = np.zeros((h, w), dtype=np.uint8)
        prior_zbuffer = visual_evidence.get("prior_zbuffer")
        faces, _ = self.renderer.get_part_faces_and_studs(hypothesis.child_part_id)
        R_h, t_h = hypothesis.rotation_matrix, hypothesis.translation
        if faces:
            for f in faces[:96]:
                pts_w = (f["pts"] @ R_h.T) + t_h
                proj_2d, depths = self.renderer.project_points(
                    pts_w, az_deg, el_deg, scale, offset, center_3d=center_3d
                )
                poly = np.round(proj_2d).astype(np.int32)
                if prior_zbuffer is not None:
                    f_depth = float(np.mean(depths))
                    cu = int(np.clip(round(float(np.mean(proj_2d[:, 0]))), 0, w - 1))
                    cv = int(np.clip(round(float(np.mean(proj_2d[:, 1]))), 0, h - 1))
                    if f_depth <= float(prior_zbuffer[cv, cu]) + 8.0:
                        cv2.fillPoly(hyp_mask, [poly], 255)
                else:
                    cv2.fillPoly(hyp_mask, [poly], 255)
        if cv2.countNonZero(hyp_mask) < 10:
            cv2.fillPoly(hyp_mask, [hull], 255)

        fg_mask = visual_evidence["fg_mask"]
        intersection = cv2.countNonZero(cv2.bitwise_and(hyp_mask, fg_mask))
        hyp_area = max(1.0, float(cv2.countNonZero(hyp_mask)))
        overlap_ratio = intersection / hyp_area

        # Differential IoU on Delta I_t = I_t \ R(G_{t-1}, M_cam)
        diff_fg_mask = visual_evidence.get("diff_fg_mask")
        diff_overlap = overlap_ratio
        if diff_fg_mask is not None and cv2.countNonZero(diff_fg_mask) > 35:
            diff_inter = cv2.countNonZero(cv2.bitwise_and(hyp_mask, diff_fg_mask))
            diff_overlap = float(diff_inter) / hyp_area

        # Local Focal IoU in bounding box of candidate projection
        bx, by, bw, bh = cv2.boundingRect(hull)
        pad = 8
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        local_fg = fg_mask[y0:y1, x0:x1]
        local_hyp = hyp_mask[y0:y1, x0:x1]
        if local_hyp.size > 0 and local_fg.shape == local_hyp.shape:
            local_fg = np.ascontiguousarray(local_fg, dtype=np.uint8)
            local_hyp = np.ascontiguousarray(local_hyp, dtype=np.uint8)
            local_inter = cv2.countNonZero(cv2.bitwise_and(local_hyp, local_fg))
            local_union = cv2.countNonZero(cv2.bitwise_or(local_hyp, local_fg))
            local_iou = local_inter / max(1.0, float(local_union))
        else:
            local_iou = 0.0

        # 2. Quadrant Check
        img_cx, img_cy = visual_evidence["img_center"]
        quad = "front_left" if (cy > img_cy and cx < img_cx) else \
               "front_right" if (cy > img_cy and cx >= img_cx) else \
               "rear_left" if (cy <= img_cy and cx < img_cx) else "rear_right"

        quad_bonus = 1.0 if (target_quadrant is None or quad == target_quadrant) else 0.1

        # 3. Orientation Alignment Score
        if target_angle_deg is not None:
            angle_diff = abs(proj_angle - target_angle_deg)
            while angle_diff > 90.0:
                angle_diff = abs(angle_diff - 180.0)
            orient_score = math.cos(math.radians(angle_diff)) ** 2
        else:
            orient_score = 0.5

        # Level 4: Topological Chamfer Distance Alignment of LDraw Type-2 Wireframe Edges (Full + Differential DT)
        dt = visual_evidence.get("dt")
        diff_dt = visual_evidence.get("diff_dt")
        edges_3d = self.get_part_wireframe_edges(hypothesis.child_part_id)
        R, t = hypothesis.rotation_matrix, hypothesis.translation

        if dt is not None and edges_3d:
            edge_scores = []
            diff_edge_scores = []
            for p1, p2 in edges_3d:
                p1_w = (R @ p1) + t
                p2_w = (R @ p2) + t
                proj, _ = self.renderer.project_points(
                    np.array([p1_w, p2_w], dtype=np.float32), az_deg, el_deg, scale, offset, center_3d=center_3d
                )
                u1, v1 = proj[0]
                u2, v2 = proj[1]
                length = math.hypot(u2 - u1, v2 - v1)
                steps = max(2, int(round(length / 6.0)))
                for s in range(steps + 1):
                    alpha = s / float(steps)
                    u = int(round(u1 + alpha * (u2 - u1)))
                    v = int(round(v1 + alpha * (v2 - v1)))
                    if 0 <= u < w and 0 <= v < h:
                        d = dt[v, u]
                        edge_scores.append(math.exp(-(d**2) / (2.0 * (2.2**2))))
                        if diff_dt is not None:
                            dd = diff_dt[v, u]
                            diff_edge_scores.append(math.exp(-(dd**2) / (2.0 * (2.8**2))))
                    else:
                        edge_scores.append(0.0)
                        if diff_dt is not None:
                            diff_edge_scores.append(0.0)
            base_chamfer = float(np.mean(edge_scores)) if edge_scores else 0.0
            diff_chamfer = float(np.mean(diff_edge_scores)) if diff_edge_scores else base_chamfer
            chamfer_edge_score = 0.55 * base_chamfer + 0.45 * diff_chamfer
        else:
            chamfer_edge_score = 0.5

        # 5. Stud Occlusion and Visible Stud Field Verification
        stud_field_score = self.compute_stud_occlusion_score(
            hypothesis, visual_evidence, scale, offset, el_deg, center_3d=center_3d
        )

        # 6. Radial Center & Skeletal Axis Alignment
        radial_score = self.compute_radial_axis_alignment_score(hypothesis)

        # Level 5: Heuristics — Stud Connection Density, Multi-Parent Gap Bridging, Coplanarity & Symmetry
        stud_count = len(hypothesis.matched_studs)
        stud_density_score = min(1.0, float(stud_count) / 4.0)
        bridged_parents = getattr(hypothesis, "bridged_parents", 1)
        bridge_bonus = min(0.18, 0.09 * float(max(0, bridged_parents - 1)))

        neighbor_score = 0.5
        coplanar_bonus = 0.0
        if recent_nodes:
            recent_dists = []
            for rn in recent_nodes[-8:]:
                rn_tf = np.array(rn["absolute_transform"], dtype=np.float32).reshape((4, 4))
                rn_pos = rn_tf[:3, 3]
                d = float(np.linalg.norm(hypothesis.translation - rn_pos))
                recent_dists.append(d)
                if abs(hypothesis.translation[1] - rn_pos[1]) < 1.0:
                    coplanar_bonus = 0.05
            if recent_dists:
                min_d = min(recent_dists)
                neighbor_score = float(math.exp(-(max(0.0, min_d - 20.0)**2) / (2.0 * (60.0**2))))

        # Bilateral symmetry bonus when a previous part of the same step is already placed
        symmetry_step_bonus = 0.0
        if already_placed_poses:
            for ap in already_placed_poses:
                ap_pos = ap if isinstance(ap, np.ndarray) else np.array(ap.get("pos", ap.get("translation", [0, 0, 0])), dtype=np.float32)
                if np.linalg.norm(hypothesis.translation - ap_pos) < 6.0:
                    neighbor_score = -5.0
                    break
                # Check mirror reflection across X=0 or Z=0 on the same Y level
                if abs(hypothesis.translation[1] - ap_pos[1]) < 1.5:
                    if (abs(hypothesis.translation[0] + ap_pos[0]) < 2.0 and abs(hypothesis.translation[2] - ap_pos[2]) < 2.0) or \
                       (abs(hypothesis.translation[2] + ap_pos[2]) < 2.0 and abs(hypothesis.translation[0] - ap_pos[0]) < 2.0):
                        symmetry_step_bonus = 0.14

        # Level 3 Color Consistency Score (matches candidate's expected LEGO LDraw color against crop pixels)
        color_score = self.compute_color_match_score(hypothesis, visual_evidence, hyp_mask)

        total_score = (
            0.26 * chamfer_edge_score +
            0.22 * diff_overlap +
            0.18 * color_score +
            0.12 * neighbor_score +
            0.08 * local_iou +
            0.08 * stud_field_score +
            0.06 * stud_density_score +
            bridge_bonus +
            coplanar_bonus +
            symmetry_step_bonus +
            occlusion_penalty
        )
        if radial_score < 0.9:
            total_score *= radial_score

        hypothesis.color_score = float(color_score)
        hypothesis.iou_score = float(diff_overlap)
        hypothesis.orientation_score = float(orient_score)
        hypothesis.edge_score = float(chamfer_edge_score)
        hypothesis.stud_occlusion_score = float(stud_field_score)
        hypothesis.neighbor_score = float(neighbor_score)
        hypothesis.total_score = float(total_score)

        return total_score

    def compute_color_match_score(
        self,
        hypothesis: ConnectionHypothesis,
        visual_evidence: Dict[str, Any],
        hyp_mask: np.ndarray,
    ) -> float:
        """
        Evaluates perceptual color agreement between the candidate LEGO part's color_id
        and the non-edge foreground pixels inside its projected 2D footprint in the PDF crop.
        """
        img = visual_evidence.get("image")
        if img is None or hyp_mask is None:
            return 0.5

        color_id = getattr(hypothesis, "color_id", 16)
        if color_id == 16:
            return 0.5

        expected_bgr = np.array(self.get_color_bgr(color_id), dtype=np.float32)

        # Sample interior pixels inside hyp_mask that are not black outline ink
        gray = visual_evidence.get("gray")
        fg_mask = visual_evidence.get("fg_mask")
        valid_mask = hyp_mask > 0
        if fg_mask is not None:
            valid_mask = valid_mask & (fg_mask > 0)
        if gray is not None and color_id != 0:
            # Exclude dark wireframe ink lines unless the part itself is black (color_id == 0)
            valid_mask = valid_mask & (gray >= 65)

        pixels = img[valid_mask]
        if len(pixels) < 8:
            return 0.5

        median_bgr = np.median(pixels.astype(np.float32), axis=0)
        color_dist = float(np.linalg.norm(median_bgr - expected_bgr))
        # Gaussian chromatic similarity kernel (sigma = 55 RGB units to allow for shading/lighting)
        return float(math.exp(-(color_dist ** 2) / (2.0 * (55.0 ** 2))))

    def compute_radial_axis_alignment_score(
        self,
        hypothesis: ConnectionHypothesis
    ) -> float:
        """
        Calculates geometric skeletal alignment between child part and parent part:
        If parent is radially symmetric (e.g. C4 circular plate) and child is a sector/wedge/corner
        (has sharp focal apex/corner), rewards placements where the apex converges at the parent center.
        """
        parent_node = getattr(self, "parent_nodes_cache", {}).get(hypothesis.parent_part_index)
        if not parent_node:
            return 1.0

        p_meta = self.renderer.geom.get_part_meta(parent_node["part_id"])
        c_meta = self.renderer.geom.get_part_meta(hypothesis.child_part_id)

        # Check if parent is radially symmetric (C4, round, width == depth)
        p_bbox = p_meta["bbox"]
        p_w = p_bbox[3] - p_bbox[0]
        p_d = p_bbox[5] - p_bbox[2]
        is_radial_parent = (
            p_meta.get("symmetry_group") == "C4" or
            (abs(p_w - p_d) <= 2.0 and ("round" in parent_node["part_id"].lower() or "11213" in parent_node["part_id"]))
        )

        if not is_radial_parent:
            return 1.0

        # Parent center in world coordinates
        p_tf = np.array(parent_node["absolute_transform"], dtype=np.float32).reshape((4, 4))
        p_cx = (p_bbox[0] + p_bbox[3]) / 2.0
        p_cz = (p_bbox[2] + p_bbox[5]) / 2.0
        p_center_w = (p_tf[:3, :3] @ np.array([p_cx, 0.0, p_cz], dtype=np.float32)) + p_tf[:3, 3]

        # Check child sharp corners (focal points of wedges/sectors)
        # Note: ONLY applies to sector pieces that genuinely have their focal point at the central apex (e.g. 30565 quarter round).
        # General wedge/wing plates (such as 54384, 54383, slopes) extend outwards rather than converging at parent center.
        is_sector_corner = "30565" in hypothesis.child_part_id or "corner" in hypothesis.child_part_id
        if not is_sector_corner:
            return 1.0

        sharp_corners = [np.array([-40.0, 0.0, 40.0], dtype=np.float32)]

        R = hypothesis.rotation_matrix
        t = hypothesis.translation

        min_d_center = 999.0
        for sc in sharp_corners:
            sc_w = (R @ sc) + t
            d = math.hypot(sc_w[0] - p_center_w[0], sc_w[2] - p_center_w[2])
            if d < min_d_center:
                min_d_center = d

        # Exponential decay: score is 1.0 if apex is at center (d=0), falls off if shifted away
        return float(math.exp(-(min_d_center**2) / (2.0 * (12.0**2))))

    def compute_stud_occlusion_score(
        self,
        hypothesis: ConnectionHypothesis,
        visual_evidence: Dict[str, Any],
        scale: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
        el_deg: float = 30.0,
        center_3d: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculates alignment of visible top studs (both child's new studs and parent's unoccluded studs)
        with real stud drawing strokes in the manual diagram.
        """
        dt = visual_evidence.get("dt")
        if dt is None:
            return 0.5

        w, h = visual_evidence["width"], visual_evidence["height"]
        R, t = hypothesis.rotation_matrix, hypothesis.translation

        # 1. Child Top Studs
        child_meta = self.renderer.geom.get_part_meta(hypothesis.child_part_id)
        child_top_studs = [
            s for s in child_meta.get("studs", [])
            if (s[1] if isinstance(s, np.ndarray) else s.get("pos", [0, 0, 0])[1]) <= 0.0
        ]

        child_scores = []
        if child_top_studs:
            c_stud_pts = np.array([s if isinstance(s, np.ndarray) else s["pos"] for s in child_top_studs], dtype=np.float32)
            w_c_studs = (c_stud_pts @ R.T) + t
            proj_c, _ = self.renderer.project_points(w_c_studs, 45.0, el_deg, scale, offset, center_3d=center_3d)
            for u, v in proj_c:
                iu, iv = int(round(u)), int(round(v))
                if 0 <= iu < w and 0 <= iv < h:
                    d = dt[iv, iu]
                    child_scores.append(math.exp(-(d**2) / (2.0 * (4.0**2))))
                else:
                    child_scores.append(0.0)

        # 2. Parent Top Studs (Occluded vs Visible)
        parent_scores = []
        parent_node = getattr(self, "parent_nodes_cache", {}).get(hypothesis.parent_part_index)
        if parent_node:
            p_meta = self.renderer.geom.get_part_meta(parent_node["part_id"])
            p_tf = np.array(parent_node["absolute_transform"], dtype=np.float32).reshape((4, 4))
            p_top_studs = [
                s for s in p_meta.get("studs", [])
                if (s[1] if isinstance(s, np.ndarray) else s.get("pos", [0, 0, 0])[1]) <= 0.0
            ]
            if p_top_studs:
                p_stud_pts = np.array([s if isinstance(s, np.ndarray) else s["pos"] for s in p_top_studs], dtype=np.float32)
                w_p_studs = (p_stud_pts @ p_tf[:3, :3].T) + p_tf[:3, 3]

                # Filter visible studs (not occluded by child)
                for pt_w in w_p_studs:
                    pt_local = R.T @ (pt_w - t)
                    cb = child_meta["bbox"]
                    is_occluded = (cb[0] - 2.0 <= pt_local[0] <= cb[3] + 2.0 and cb[2] - 2.0 <= pt_local[2] <= cb[5] + 2.0)
                    if not is_occluded:
                        proj_p, _ = self.renderer.project_points(pt_w.reshape(1, 3), 45.0, el_deg, scale, offset)
                        iu, iv = int(round(proj_p[0, 0])), int(round(proj_p[0, 1]))
                        if 0 <= iu < w and 0 <= iv < h:
                            d = dt[iv, iu]
                            parent_scores.append(math.exp(-(d**2) / (2.0 * (4.0**2))))

        s_c = float(np.mean(child_scores)) if child_scores else 0.5
        s_p = float(np.mean(parent_scores)) if parent_scores else 0.5
        return 0.6 * s_c + 0.4 * s_p

    def calibrate_camera_from_prior_state(
        self,
        existing_nodes: List[Dict[str, Any]],
        visual_evidence: Dict[str, Any],
        is_flip: bool = False,
    ) -> Dict[str, Any]:
        """
        Phase A: Camera Calibration (Aligning consolidated 3D state G_{t-1} against image I_t)
         followed immediately by Phase B: Differential Rendering (Delta I_t = I_t - R(G_{t-1}, M_cam)).
        """
        w = visual_evidence.get("width", 400)
        h = visual_evidence.get("height", 400)
        base_scale = float(visual_evidence.get("auto_scale", 2.65))
        base_offset = visual_evidence.get("fg_center", (w / 2.0, h / 2.0))
        el_deg = -25.0 if is_flip else 30.0
        dt = visual_evidence.get("dt")

        # Deterministic Fast Path: Affine Stud-to-Ellipse RANSAC (< 2 ms)
        vec_ev = visual_evidence.get("vector_evidence", {})
        stud_ellipses_objs = vec_ev.get("stud_ellipse_objects", [])
        if stud_ellipses_objs and len(existing_nodes) >= 1:
            all_studs_3d = []
            for rn in existing_nodes:
                pid = rn["part_id"]
                rn_tf = np.array(rn["absolute_transform"], dtype=np.float32).reshape((4, 4))
                R, t = rn_tf[:3, :3], rn_tf[:3, 3]
                meta = self.renderer.geom.get_part_meta(pid)
                for s in meta.get("studs", []):
                    s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
                    s_w = (s_pos @ R.T) + t
                    all_studs_3d.append(s_w)

            if len(all_studs_3d) >= 3:
                studs_arr = np.array(all_studs_3d, dtype=np.float32)
                ransac_res = self.vector_engine.calibrate_affine_stud_ransac(
                    studs_3d=studs_arr,
                    detected_ellipses=stud_ellipses_objs,
                    initial_scale_guess=base_scale,
                    target_center_2d=base_offset,
                    elevation_deg=abs(el_deg),
                )
                if ransac_res.success and ransac_res.residual_error < 2.5:
                    return {
                        "scale": ransac_res.scale,
                        "offset": ransac_res.offset,
                        "az_deg": ransac_res.az_deg,
                        "el_deg": ransac_res.el_deg,
                        "center_3d": np.mean(studs_arr, axis=0),
                        "alignment_score": max(0.85, 1.0 - ransac_res.residual_error / 5.0),
                        "projection_matrix": ransac_res.projection_matrix,
                        "is_vector_calibrated": True,
                    }

        if not existing_nodes or dt is None:
            return {
                "scale": base_scale,
                "offset": base_offset,
                "az_deg": 45.0,
                "el_deg": el_deg,
                "center_3d": None,
                "alignment_score": 0.5,
            }

        # Collect 3D bounding box of full G_{t-1} and wireframe sample points from placed nodes
        sample_pts_3d = []
        all_bbox_pts = []
        for rn in existing_nodes:
            pid = rn["part_id"]
            rn_tf = np.array(rn["absolute_transform"], dtype=np.float32).reshape((4, 4))
            R, t = rn_tf[:3, :3], rn_tf[:3, 3]
            meta = self.renderer.geom.get_part_meta(pid)
            bx0, by0, bz0, bx1, by1, bz1 = meta["bbox"]
            corners = np.array([
                [bx0, by0, bz0], [bx1, by0, bz0], [bx1, by1, bz0], [bx0, by1, bz0],
                [bx0, by0, bz1], [bx1, by0, bz1], [bx1, by1, bz1], [bx0, by1, bz1]
            ], dtype=np.float32)
            w_corners = (corners @ R.T) + t
            all_bbox_pts.append(w_corners)

        for rn in existing_nodes[-12:]:
            pid = rn["part_id"]
            rn_tf = np.array(rn["absolute_transform"], dtype=np.float32).reshape((4, 4))
            R, t = rn_tf[:3, :3], rn_tf[:3, 3]
            edges = self.get_part_wireframe_edges(pid)[:16]
            for p1, p2 in edges:
                sample_pts_3d.append((R @ p1) + t)
                sample_pts_3d.append((R @ ((p1 + p2) * 0.5)) + t)

        if not sample_pts_3d:
            return {
                "scale": base_scale,
                "offset": base_offset,
                "az_deg": 45.0,
                "el_deg": el_deg,
                "center_3d": None,
                "alignment_score": 0.5,
            }

        pts_arr = np.array(sample_pts_3d, dtype=np.float32)
        bbox_arr = np.vstack(all_bbox_pts)
        center_3d = (np.min(bbox_arr, axis=0) + np.max(bbox_arr, axis=0)) / 2.0

        best_params = {
            "scale": base_scale,
            "offset": base_offset,
            "az_deg": 45.0,
            "el_deg": el_deg,
            "center_3d": center_3d,
            "alignment_score": -1.0,
        }

        for az_cand in (45.0, 135.0, 225.0, 315.0):
            for scale_mult in (0.85, 0.92, 1.0, 1.08, 1.16):
                cand_s = base_scale * scale_mult
                for dx in (-24.0, -12.0, 0.0, 12.0, 24.0):
                    for dy in (-24.0, -12.0, 0.0, 12.0, 24.0):
                        cand_off = (base_offset[0] + dx, base_offset[1] + dy)
                        proj_2d, _ = self.renderer.project_points(
                            pts_arr, az_cand, el_deg, cand_s, cand_off, center_3d=center_3d
                        )
                        us = np.clip(np.round(proj_2d[:, 0]).astype(np.int32), 0, w - 1)
                        vs = np.clip(np.round(proj_2d[:, 1]).astype(np.int32), 0, h - 1)
                        dists = dt[vs, us]
                        az_prior = 1.0 if az_cand == 45.0 else 0.94
                        score = float(np.mean(np.exp(-(dists ** 2) / (2.0 * (3.0 ** 2))))) * az_prior
                        if score > best_params["alignment_score"]:
                            best_params = {
                                "scale": cand_s,
                                "offset": cand_off,
                                "az_deg": az_cand,
                                "el_deg": el_deg,
                                "center_3d": center_3d,
                                "alignment_score": score,
                            }

        # Execute Phase B immediately: synthesize R(G_{t-1}, M_cam), Z-buffer, and Delta I_t
        self.compute_prior_state_differential_evidence(existing_nodes, visual_evidence, best_params)
        best_params["roi_bbox_2d"] = visual_evidence.get("roi_bbox_2d")
        return best_params

    def score_color_batch_mps(
        self,
        hypotheses: List[ConnectionHypothesis],
        visual_evidence: Dict[str, Any],
        scale: float = 2.65,
        offset: Tuple[float, float] = (0.0, 0.0),
        az_deg: float = 45.0,
        el_deg: float = 30.0,
        center_3d: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Vectorized batch color consistency scoring on Apple Silicon GPU (MPS):
        Samples crop RGB values around projected 3D surface points of each candidate hypothesis
        and compares them against the expected LEGO color BGR vector.
        """
        if not hypotheses:
            return []
        img = visual_evidence.get("image")
        if img is None:
            return [0.5] * len(hypotheses)

        w, h = int(visual_evidence["width"]), int(visual_evidence["height"])
        if center_3d is None:
            center_3d = visual_evidence.get("center_3d")
        c3d = np.zeros(3, dtype=np.float32) if center_3d is None else np.asarray(center_3d, dtype=np.float32)

        M = 16
        batch_pts = np.zeros((len(hypotheses), M, 3), dtype=np.float32)
        expected_bgr = np.zeros((len(hypotheses), 3), dtype=np.float32)

        for idx, hyp in enumerate(hypotheses):
            cid = getattr(hyp, "color_id", 16)
            expected_bgr[idx] = np.array(self.get_color_bgr(cid), dtype=np.float32)
            faces, _ = self.renderer.get_part_faces_and_studs(hyp.child_part_id)
            if not faces:
                pts_local = np.zeros((M, 3), dtype=np.float32)
            else:
                face_centers = [np.mean(f["pts"], axis=0) for f in faces[:M]]
                while len(face_centers) < M:
                    face_centers.append(face_centers[-1])
                pts_local = np.array(face_centers[:M], dtype=np.float32)
            batch_pts[idx] = (pts_local @ hyp.rotation_matrix.T) + hyp.translation - c3d

        az = math.radians(az_deg)
        el = math.radians(el_deg)
        ca, sa = math.cos(az), math.sin(az)
        ce, se = math.cos(el), math.sin(el)
        P_2x3 = np.array([
            [ca * scale, 0.0, -sa * scale],
            [sa * se * scale, ce * scale, ca * se * scale],
        ], dtype=np.float32)

        with torch.no_grad():
            pts_t = torch.tensor(batch_pts, dtype=torch.float32, device=self.device)  # [B, M, 3]
            P_t = torch.tensor(P_2x3.T, dtype=torch.float32, device=self.device)      # [3, 2]
            off_t = torch.tensor([offset[0], offset[1]], dtype=torch.float32, device=self.device)

            uv = torch.matmul(pts_t, P_t) + off_t  # [B, M, 2]
            norm_u = torch.clamp((uv[..., 0] / max(1.0, float(w - 1))) * 2.0 - 1.0, -1.0, 1.0)
            norm_v = torch.clamp((uv[..., 1] / max(1.0, float(h - 1))) * 2.0 - 1.0, -1.0, 1.0)
            grid = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(0)  # [1, B, M, 2]

            # Image tensor [1, 3, H, W] in BGR float32
            img_t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device=self.device).unsqueeze(0)
            sampled_bgr = F.grid_sample(img_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled_bgr = sampled_bgr.squeeze(0).permute(1, 2, 0)
            exp_t = torch.tensor(expected_bgr, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B, 1, 3]
            color_dists = torch.norm(sampled_bgr - exp_t, dim=-1)  # [B, M]
            scores_t = torch.exp(-(color_dists ** 2) / (2.0 * (60.0 ** 2))).mean(dim=-1)
            return [float(v) for v in scores_t.cpu().numpy()]

    def score_chamfer_batch_mps(
        self,
        hypotheses: List[ConnectionHypothesis],
        visual_evidence: Dict[str, Any],
        scale: float = 2.65,
        offset: Tuple[float, float] = (0.0, 0.0),
        az_deg: float = 45.0,
        el_deg: float = 30.0,
        center_3d: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Vectorized batch Chamfer edge scoring on Apple Silicon GPU (MPS) using PyTorch
        tensor operations (torch.matmul and F.grid_sample) over both full and differential
        2D Distance Transforms (`dt` and `diff_dt`).
        """
        if not hypotheses:
            return []
        dt = visual_evidence.get("dt")
        if dt is None:
            return [0.5] * len(hypotheses)
        diff_dt = visual_evidence.get("diff_dt")

        w, h = int(visual_evidence["width"]), int(visual_evidence["height"])
        if center_3d is None:
            center_3d = visual_evidence.get("center_3d")
        c3d = np.zeros(3, dtype=np.float32) if center_3d is None else np.asarray(center_3d, dtype=np.float32)

        # Build fixed-size 3D edge sample cloud per hypothesis (M=32 points per hypothesis)
        M = 32
        batch_pts = np.zeros((len(hypotheses), M, 3), dtype=np.float32)
        for idx, hyp in enumerate(hypotheses):
            edges = self.get_part_wireframe_edges(hyp.child_part_id)
            if not edges:
                pts_local = np.zeros((M, 3), dtype=np.float32)
            else:
                sampled = []
                for p1, p2 in edges:
                    sampled.append(p1)
                    sampled.append(0.5 * (p1 + p2))
                    sampled.append(p2)
                    if len(sampled) >= M:
                        break
                while len(sampled) < M:
                    sampled.append(sampled[-1])
                pts_local = np.array(sampled[:M], dtype=np.float32)
            pts_w = (pts_local @ hyp.rotation_matrix.T) + hyp.translation - c3d
            batch_pts[idx] = pts_w

        # Axonometric projection matrix on MPS
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        ca, sa = math.cos(az), math.sin(az)
        ce, se = math.cos(el), math.sin(el)
        P_2x3 = np.array([
            [ca * scale, 0.0, -sa * scale],
            [sa * se * scale, ce * scale, ca * se * scale],
        ], dtype=np.float32)

        with torch.no_grad():
            pts_t = torch.tensor(batch_pts, dtype=torch.float32, device=self.device)  # [B, M, 3]
            P_t = torch.tensor(P_2x3.T, dtype=torch.float32, device=self.device)      # [3, 2]
            off_t = torch.tensor([offset[0], offset[1]], dtype=torch.float32, device=self.device)

            uv = torch.matmul(pts_t, P_t) + off_t  # [B, M, 2]
            norm_u = torch.clamp((uv[..., 0] / max(1.0, float(w - 1))) * 2.0 - 1.0, -1.0, 1.0)
            norm_v = torch.clamp((uv[..., 1] / max(1.0, float(h - 1))) * 2.0 - 1.0, -1.0, 1.0)
            grid = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(0)  # [1, B, M, 2]

            dt_t = torch.tensor(dt, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            sampled_dt = F.grid_sample(dt_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(0).squeeze(0)
            base_scores = torch.exp(-(sampled_dt ** 2) / (2.0 * (2.2 ** 2))).mean(dim=-1)

            if diff_dt is not None:
                ddt_t = torch.tensor(diff_dt, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                sampled_ddt = F.grid_sample(ddt_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(0).squeeze(0)
                diff_scores = torch.exp(-(sampled_ddt ** 2) / (2.0 * (2.8 ** 2))).mean(dim=-1)
                scores_t = 0.55 * base_scores + 0.45 * diff_scores
            else:
                scores_t = base_scores

            return [float(v) for v in scores_t.cpu().numpy()]

    def filter_hypotheses_by_2d_roi(
        self,
        hypotheses: List[ConnectionHypothesis],
        visual_evidence: Dict[str, Any],
        scale: float = 2.65,
        offset: Tuple[float, float] = (0.0, 0.0),
        el_deg: float = 30.0,
        center_3d: Optional[np.ndarray] = None,
        max_keep: int = 40,
    ) -> List[ConnectionHypothesis]:
        """
        Fast 2D spatial ROI pre-filter (MEPNet-style centroid prior):
        Projects each hypothesis 3D center into 2D and prioritizes candidates near the
        step's active delta mask or foreground region.
        """
        if len(hypotheses) <= max_keep or not visual_evidence:
            return hypotheses

        delta_mask = visual_evidence.get("step_delta_mask")
        target_2d = visual_evidence.get("fg_center", offset)
        if delta_mask is not None and cv2.countNonZero(delta_mask) > 60:
            ys, xs = np.nonzero(delta_mask)
            target_2d = (float(np.mean(xs)), float(np.mean(ys)))

        centers_3d = np.array([h.translation for h in hypotheses], dtype=np.float32)
        proj_2d, _ = self.renderer.project_points(
            centers_3d, 45.0, el_deg, scale, offset, center_3d=center_3d
        )
        dists = np.linalg.norm(proj_2d - np.array(target_2d, dtype=np.float32), axis=1)
        order = np.argsort(dists)[:max_keep]
        return [hypotheses[int(i)] for i in order]

    def rank_hypotheses(
        self,
        hypotheses: List[ConnectionHypothesis],
        manual_crop_path: str,
        target_quadrant: Optional[str] = None,
        target_angle_deg: Optional[float] = None,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
        is_flip: bool = False,
        parent_nodes: Optional[List[Dict[str, Any]]] = None,
        recent_nodes: Optional[List[Dict[str, Any]]] = None,
        already_placed_poses: Optional[List[Any]] = None,
        inside_pool: bool = False,          # ← Mejora #2: evitar oversubscription
    ) -> List[ConnectionHypothesis]:
        """
        Ranks all candidate hypotheses against visual evidence, returning sorted by score descending.

        Args:
            inside_pool: Pasar True cuando este método se invoca desde dentro de un pool externo
                         (e.g. _eval_single_branch en beam search). Fuerza workers=1 internamente
                         para evitar oversubscription en los 8 P-cores del M4 Pro.
        """
        if not hypotheses:
            return []

        evidence = self.extract_visual_evidence(manual_crop_path)
        if not evidence:
            return hypotheses

        # Calibrate camera from prior 3D state G_{t-1} when default scale/offset are passed
        if parent_nodes and scale <= 1.0 and offset == (0.0, 0.0):
            calib = self.calibrate_camera_from_prior_state(parent_nodes, evidence, is_flip=is_flip)
            eff_scale = calib["scale"]
            eff_offset = calib["offset"]
        else:
            eff_scale = evidence.get("auto_scale", 2.65) if scale <= 1.0 else scale
            eff_offset = evidence.get("fg_center", (0.0, 0.0)) if offset == (0.0, 0.0) else offset

        if parent_nodes:
            self.parent_nodes_cache = {p["part_index"]: p for p in parent_nodes}
        else:
            self.parent_nodes_cache = {}

        # FAST-PATH OPTIMIZATION:
        # Pre-filter candidate hypotheses to Top-K (max 15) using lightweight centroid distance
        # and stud occlusion alignment before expensive full mesh render and Chamfer DT evaluation.
        if len(hypotheses) > 15:
            hypotheses = self.prune_hypotheses_by_2d_distance(
                hypotheses, evidence, eff_scale, eff_offset, el_deg=-25.0 if is_flip else 30.0, max_keep=15
            )

        # Parallel scoring across CPU cores
        # Mejora #2: si ya estamos en un pool externo (beam search), forzar secuencial
        effective_workers = workers_for(len(hypotheses), inside_pool=inside_pool)

        def eval_h(h):
            self.score_single_hypothesis(
                h, evidence, target_quadrant, target_angle_deg, eff_scale, eff_offset, is_flip,
                recent_nodes=recent_nodes, already_placed_poses=already_placed_poses
            )
            return h

        if effective_workers <= 1:
            scored = [eval_h(h) for h in hypotheses]
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                scored = list(executor.map(eval_h, hypotheses))

        scored.sort(key=lambda h: h.total_score, reverse=True)
        return scored

    def find_best_joint_hypotheses(
        self,
        part_candidates: List[Tuple[str, str, List[ConnectionHypothesis], Dict[str, Any]]],
        manual_crop_path: str,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
        is_flip: bool = False,
        parent_nodes: Optional[List[Dict[str, Any]]] = None
    ) -> List[ConnectionHypothesis]:
        """
        Solves Joint Multi-Part Matching for multiple pieces placed in the same step.
        Guarantees:
          1. Mutual non-collision between newly placed parts.
          2. No overlapping studs between newly placed parts.
          3. Maximization of joint visual likelihood score.
        """
        evidence = self.extract_visual_evidence(manual_crop_path)
        if not evidence:
            return [cands[0] for _, _, cands, _ in part_candidates if cands]

        eff_scale = evidence.get("auto_scale", 2.65) if scale <= 1.0 else scale
        eff_offset = evidence.get("fg_center", (0.0, 0.0)) if offset == (0.0, 0.0) else offset

        # Score candidates individually first
        ranked_lists = []
        for part_id, el_id, cands, meta in part_candidates:
            quad = meta.get("target_quadrant")
            ang = meta.get("target_angle_deg")
            ranked = self.rank_hypotheses(
                cands, manual_crop_path, target_quadrant=quad, target_angle_deg=ang,
                scale=eff_scale, offset=eff_offset, is_flip=is_flip, parent_nodes=parent_nodes
            )
            # Take top 12 for joint combination search
            ranked_lists.append(ranked[:12])

        if len(ranked_lists) == 1:
            return [ranked_lists[0][0]] if ranked_lists[0] else []

        # Joint search across tuples
        best_tuple = None
        best_joint_score = -1.0

        if len(ranked_lists) == 2:
            for h1 in ranked_lists[0]:
                for h2 in ranked_lists[1]:
                    # Check mutual bounding box collision
                    b1, b2 = h1.world_bbox, h2.world_bbox
                    # Margin check
                    if not (b1[3] <= b2[0] or b2[3] <= b1[0] or
                            b1[4] <= b2[1] or b2[4] <= b1[1] or
                            b1[5] <= b2[2] or b2[5] <= b1[2]):
                        # Overlap collision detected -> illegal joint state!
                        continue

                    # Check mutual stud exclusivity
                    stud_collision = False
                    for s1 in h1.matched_studs:
                        for s2 in h2.matched_studs:
                            if np.linalg.norm(np.array(s1) - np.array(s2)) < 3.0:
                                stud_collision = True
                                break
                        if stud_collision:
                            break

                    if stud_collision:
                        continue

                    joint_score = h1.total_score + h2.total_score
                    if joint_score > best_joint_score:
                        best_joint_score = joint_score
                        best_tuple = [h1, h2]

        if best_tuple:
            return best_tuple

        # Fallback to independent top choices if joint search found no pair
        return [r[0] for r in ranked_lists if r]

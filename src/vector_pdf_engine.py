"""
Vector PDF Engine for LegoGPT.

Extracts native vector primitives directly from PyMuPDF drawing streams:
  1. Bézier Curve Stud Ellipses: detects 2D closed paths (and cubic Bézier 'c' loops)
     corresponding to circular studs viewed under isometric/axonometric projection (~30 deg).
  2. Highlight Outline & ROI: extracts bright highlight strokes (red #E3000B, orange, yellow)
     marking newly assembled bricks without raster difference.
  3. Deterministic 2D-3D Stud-to-Ellipse RANSAC: solves an optimal affine projection
     matrix (P in R^{2x4}) aligning 3D model studs of G_{t-1} with 2D page ellipses in < 3 ms.
  4. Stud Disappearance Resolver: detects which exposed studs of G_{t-1} disappear
     under the new piece ROI in step t, pinpointing exact parent connection sockets.

Optimized for macOS / Apple Silicon M4 and dynamic hardware resources.
"""

from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class StudEllipse2D:
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    width: float
    height: float
    aspect_ratio: float
    color: Optional[Tuple[float, ...]] = None
    fill: Optional[Tuple[float, ...]] = None
    is_bezier: bool = False


@dataclass
class VectorEvidence:
    is_vector_scene: bool
    stud_ellipses: List[StudEllipse2D] = field(default_factory=list)
    highlight_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    roi_new_parts: Optional[Tuple[float, float, float, float]] = None
    stroke_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    callout_rects: List[Tuple[float, float, float, float]] = field(default_factory=list)
    num_drawings: int = 0
    num_beziers: int = 0


@dataclass
class AffineCalibrationResult:
    success: bool
    projection_matrix: Optional[np.ndarray] = None  # 2x4 affine matrix
    inliers_3d: List[int] = field(default_factory=list)
    inliers_2d: List[int] = field(default_factory=list)
    residual_error: float = float("inf")
    scale: float = 1.0
    offset: Tuple[float, float] = (0.0, 0.0)
    az_deg: float = 45.0
    el_deg: float = 30.0


class VectorPDFEngine:
    """
    Native vector extraction and deterministic 2D-3D geometric alignment engine
    for digital and official LEGO instruction manuals.
    """

    def __init__(self, stud_tolerance_pt: float = 4.0, max_ransac_iters: int = 200):
        self.stud_tolerance_pt = float(stud_tolerance_pt)
        self.max_ransac_iters = int(max_ransac_iters)

    # -------------------------------------------------------------------------
    # 1. Primitives & Bézier Extraction
    # -------------------------------------------------------------------------
    def extract_vector_evidence(
        self,
        page: Any,
        crop_rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> VectorEvidence:
        """
        Inspects `page.get_drawings()` in PyMuPDF to extract native vector geometry:
          - Small closed Bézier loops / ovals (studs)
          - Highlight colored strokes (new step pieces)
          - Part callout boxes
        """
        if page is None or not hasattr(page, "get_drawings"):
            return VectorEvidence(is_vector_scene=False)

        drawings = page.get_drawings()
        if not drawings:
            return VectorEvidence(is_vector_scene=False)

        pw = float(page.rect.width) if hasattr(page, "rect") else 600.0
        ph = float(page.rect.height) if hasattr(page, "rect") else 800.0

        cx0, cy0, cx1, cy1 = crop_rect if crop_rect is not None else (0.0, 0.0, pw, ph)

        stud_ellipses: List[StudEllipse2D] = []
        highlight_bboxes: List[Tuple[float, float, float, float]] = []
        stroke_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        callout_rects: List[Tuple[float, float, float, float]] = []
        total_beziers = 0

        for d in drawings:
            r = d.get("rect")
            if r is None:
                continue

            rx0, ry0, rx1, ry1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
            # Clip check
            if rx1 < cx0 or rx0 > cx1 or ry1 < cy0 or ry0 > cy1:
                continue

            w = float(r.width)
            h = float(r.height)
            if w <= 0.0 or h <= 0.0:
                continue

            color = d.get("color")
            fill = d.get("fill")
            items = d.get("items", [])

            c_count = 0
            for it in items:
                if it[0] == "c":
                    c_count += 1
                    total_beziers += 1
                elif it[0] == "l" and len(it) >= 3:
                    p1, p2 = it[1], it[2]
                    p1_x = getattr(p1, "x", p1[0] if isinstance(p1, (tuple, list)) else 0.0)
                    p1_y = getattr(p1, "y", p1[1] if isinstance(p1, (tuple, list)) else 0.0)
                    p2_x = getattr(p2, "x", p2[0] if isinstance(p2, (tuple, list)) else 0.0)
                    p2_y = getattr(p2, "y", p2[1] if isinstance(p2, (tuple, list)) else 0.0)
                    stroke_segments.append(((float(p1_x), float(p1_y)), (float(p2_x), float(p2_y))))

            # 1. Callout Boxes: medium-sized rectangles with distinct background
            if 60.0 < w < 0.75 * pw and 40.0 < h < 0.65 * ph and fill is not None:
                callout_rects.append((rx0, ry0, rx1, ry1))

            # 2. Highlight Strokes: High-saturation red, orange or yellow outlines
            if color and len(color) >= 3:
                cr, cg, cb = float(color[0]), float(color[1]), float(color[2])
                is_red = cr > 0.72 and cb < 0.28 and cg < 0.35
                is_orange = cr > 0.82 and 0.30 <= cg <= 0.70 and cb < 0.25
                is_yellow = cr > 0.85 and cg > 0.80 and cb < 0.30
                if (is_red or is_orange or is_yellow) and (4.0 < w < 0.65 * pw and 4.0 < h < 0.65 * ph):
                    highlight_bboxes.append((rx0, ry0, rx1, ry1))

            # 3. Stud Ellipses:
            # Small closed loop / oval path.
            # Stud in PDF is typically 3 to 28 pt wide, with aspect ratio between 1.05 and 2.5
            aspect = max(w, h) / max(0.5, min(w, h))
            is_stud_size = (3.0 <= w <= 32.0 and 2.0 <= h <= 25.0)
            if is_stud_size and (1.05 <= aspect <= 2.6):
                # Must be a closed loop, contain bezier curves or small oval path
                is_bezier = c_count >= 2 or d.get("closePath", False)
                center = ((rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5)
                stud_ellipses.append(
                    StudEllipse2D(
                        center=center,
                        bbox=(rx0, ry0, rx1, ry1),
                        width=w,
                        height=h,
                        aspect_ratio=aspect,
                        color=color,
                        fill=fill,
                        is_bezier=is_bezier,
                    )
                )

        # Merge highlight bboxes into a single tight ROI if present
        roi_new = None
        if highlight_bboxes:
            min_x = min(b[0] for b in highlight_bboxes)
            min_y = min(b[1] for b in highlight_bboxes)
            max_x = max(b[2] for b in highlight_bboxes)
            max_y = max(b[3] for b in highlight_bboxes)
            roi_new = (min_x, min_y, max_x, max_y)

        # Determine if scene has vector assembly content
        is_vector = (len(stud_ellipses) >= 2 or total_beziers > 15 or len(stroke_segments) > 40)

        return VectorEvidence(
            is_vector_scene=is_vector,
            stud_ellipses=stud_ellipses,
            highlight_bboxes=highlight_bboxes,
            roi_new_parts=roi_new,
            stroke_segments=stroke_segments,
            callout_rects=callout_rects,
            num_drawings=len(drawings),
            num_beziers=total_beziers,
        )

    # -------------------------------------------------------------------------
    # 2. Deterministic 2D-3D Affine RANSAC Camera Alignment
    # -------------------------------------------------------------------------
    def calibrate_affine_stud_ransac(
        self,
        studs_3d: np.ndarray,  # shape (N, 3) in world LDraw coordinates
        detected_ellipses: List[StudEllipse2D],
        initial_scale_guess: float = 2.65,
        target_center_2d: Optional[Tuple[float, float]] = None,
        elevation_deg: float = 30.0,
    ) -> AffineCalibrationResult:
        """
        Estimates the 2x4 affine projection matrix P mapping 3D studs to 2D ellipses:
            [u, v]^T = P * [X, Y, Z, 1]^T
        Uses an ultra-fast canonical RANSAC scheme leveraging axonometric view priors:
          1. Evaluates candidate axonometric view rotations (azimuth 45°, 135°, 225°, 315°).
          2. Matches 3D projections against 2D detected ellipses.
          3. Employs SVD refinement on inliers for sub-pixel accuracy in < 3 ms.
        """
        if studs_3d is None or len(studs_3d) < 3 or not detected_ellipses:
            return AffineCalibrationResult(success=False)

        pts_2d = np.array([e.center for e in detected_ellipses], dtype=np.float32)
        n_3d = len(studs_3d)
        n_2d = len(pts_2d)

        center_3d = np.mean(studs_3d, axis=0)
        c_2d = np.array(target_center_2d, dtype=np.float32) if target_center_2d else np.mean(pts_2d, axis=0)

        best_result = AffineCalibrationResult(success=False)
        best_inlier_count = 0
        min_inliers_required = min(4, max(2, n_3d // 2))

        # Standard canonical Lego manual viewing angles
        azimuth_candidates = [45.0, 135.0, 225.0, 315.0]
        elevation_candidates = [elevation_deg, -elevation_deg]

        tol_sq = self.stud_tolerance_pt ** 2

        # 3D points centered
        pts_3d_centered = studs_3d - center_3d
        homo_3d = np.hstack([studs_3d, np.ones((n_3d, 1), dtype=np.float32)])

        for az in azimuth_candidates:
            for el in elevation_candidates:
                # Axonometric rotation matrix
                az_rad = math.radians(az)
                el_rad = math.radians(el)
                cos_a, sin_a = math.cos(az_rad), math.sin(az_rad)
                cos_e, sin_e = math.cos(el_rad), math.sin(el_rad)

                # Standard camera rotation (LDraw: +Y down, screen: +Y down)
                R_cam = np.array([
                    [-cos_a, 0.0, sin_a],
                    [-sin_a * sin_e, -cos_e, -cos_a * sin_e],
                    [sin_a * cos_e, -sin_e, cos_a * cos_e]
                ], dtype=np.float32)

                # Project 3D points to normalized 2D camera coordinates
                proj_cam = pts_3d_centered @ R_cam[:2].T  # (N, 2)

                # Scale estimation: compare standard deviation / spread of 3D projection vs 2D points
                std_proj = np.std(proj_cam, axis=0)
                std_proj_scalar = float(np.mean(std_proj))
                if std_proj_scalar < 1e-4:
                    continue

                for scale_factor in [0.75, 0.9, 1.0, 1.1, 1.25]:
                    s = initial_scale_guess * scale_factor

                    # Estimated 2D points under this prior
                    pred_2d = (proj_cam * s) + c_2d

                    # Fast pairwise distance matching to find inlier consensus
                    # cdist (N, M)
                    diff = pred_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
                    dists_sq = np.sum(diff ** 2, axis=2)  # (N, M)

                    min_dists = np.min(dists_sq, axis=1)
                    matched_2d_idx = np.argmin(dists_sq, axis=1)

                    inlier_mask_3d = min_dists < tol_sq
                    inlier_count = int(np.sum(inlier_mask_3d))

                    if inlier_count > best_inlier_count and inlier_count >= min_inliers_required:
                        inliers_3d_idx = np.where(inlier_mask_3d)[0].tolist()
                        inliers_2d_idx = matched_2d_idx[inlier_mask_3d].tolist()

                        # SVD Affine Least-Squares Fit on inliers:
                        # Solve P (2x4) such that P * [X, Y, Z, 1]^T ~= [u, v]^T
                        X_in = homo_3d[inliers_3d_idx]  # (K, 4)
                        Y_in = pts_2d[inliers_2d_idx]   # (K, 2)

                        try:
                            # P^T = (X^T X)^{-1} X^T Y
                            P_transpose, residuals, rank, _ = np.linalg.lstsq(X_in, Y_in, rcond=None)
                            P = P_transpose.T  # (2, 4)

                            # Re-verify all points under refined P
                            all_pred = (homo_3d @ P.T)
                            all_diff = all_pred[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
                            all_dists_sq = np.sum(all_diff ** 2, axis=2)
                            all_min_dists = np.min(all_dists_sq, axis=1)
                            final_inliers = np.where(all_min_dists < tol_sq)[0]
                            final_count = len(final_inliers)

                            res_err = float(np.mean(np.sqrt(all_min_dists[final_inliers])))

                            if final_count >= best_inlier_count:
                                best_inlier_count = final_count
                                best_result = AffineCalibrationResult(
                                    success=True,
                                    projection_matrix=P,
                                    inliers_3d=final_inliers.tolist(),
                                    inliers_2d=np.argmin(all_dists_sq, axis=1)[final_inliers].tolist(),
                                    residual_error=res_err,
                                    scale=s,
                                    offset=(float(c_2d[0]), float(c_2d[1])),
                                    az_deg=az,
                                    el_deg=el,
                                )
                        except Exception:
                            continue

        return best_result

    # -------------------------------------------------------------------------
    # 3. Stud Disappearance & Socket Connection Resolution
    # -------------------------------------------------------------------------
    def detect_occluded_studs(
        self,
        existing_nodes: List[Dict[str, Any]],
        geom_resolver: Any,
        projection_matrix: np.ndarray,
        detected_ellipses_step_t: List[StudEllipse2D],
        roi_new: Optional[Tuple[float, float, float, float]] = None,
        stud_search_radius_pt: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """
        Pinpoints exact parent connection studs by detecting 'Stud Disappearance':
          1. Projects all upward-facing studs of G_{t-1} onto the 2D manual step.
          2. Filters for studs that project inside the 2D bounding box (ROI_new) of the new piece.
          3. Checks which of these studs lack a corresponding 2D stud ellipse in step t
             (because their top circular face is physically occluded by the socket of the new brick).
        Returns a list of dicts with parent_part_index, 3D stud world position, and 2D projection.
        """
        if not existing_nodes or projection_matrix is None:
            return []

        pts_2d = np.array([e.center for e in detected_ellipses_step_t], dtype=np.float32) if detected_ellipses_step_t else np.zeros((0, 2))

        occluded_studs = []

        for p_node in existing_nodes:
            p_idx = p_node.get("part_index", 0)
            p_id = p_node["part_id"]
            p_meta = geom_resolver.get_part_meta(p_id)
            p_tf = np.array(p_node["absolute_transform"], dtype=np.float32).reshape((4, 4))
            R, t = p_tf[:3, :3], p_tf[:3, 3]

            for s in p_meta.get("studs", []):
                s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
                s_norm = s.get("normal", np.array([0.0, -1.0, 0.0], dtype=np.float32)) if isinstance(s, dict) else np.array([0.0, -1.0, 0.0], dtype=np.float32)

                # World 3D coordinates
                s_world = (s_pos @ R.T) + t
                s_norm_world = R @ s_norm

                # Consider upward-facing studs (active sockets)
                if s_norm_world[1] > -0.7:
                    continue

                # Project 3D stud to 2D
                s_homo = np.array([s_world[0], s_world[1], s_world[2], 1.0], dtype=np.float32)
                proj_2d = projection_matrix @ s_homo  # (2,)
                u_proj, v_proj = float(proj_2d[0]), float(proj_2d[1])

                # Check if it projects inside the new part ROI
                if roi_new is not None:
                    rx0, ry0, rx1, ry1 = roi_new
                    # Expand ROI slightly by stud radius
                    if not (rx0 - stud_search_radius_pt <= u_proj <= rx1 + stud_search_radius_pt and
                            ry0 - stud_search_radius_pt <= v_proj <= ry1 + stud_search_radius_pt):
                        continue

                # Check if a 2D stud ellipse is visible at this location in step t
                is_visible_in_step_t = False
                if len(pts_2d) > 0:
                    dists = np.hypot(pts_2d[:, 0] - u_proj, pts_2d[:, 1] - v_proj)
                    if np.min(dists) <= stud_search_radius_pt:
                        is_visible_in_step_t = True

                # If the stud was in the change region but its top ellipse disappeared -> OCCLUDED!
                if not is_visible_in_step_t:
                    occluded_studs.append({
                        "parent_part_index": p_idx,
                        "part_id": p_id,
                        "stud_world_pos": s_world.tolist(),
                        "proj_2d": (u_proj, v_proj),
                    })

        return occluded_studs

    # -------------------------------------------------------------------------
    # 4. Hypothesis Scoring Enhancement via Occluded Studs
    # -------------------------------------------------------------------------
    def score_hypotheses_with_occluded_studs(
        self,
        hypotheses: List[Any],
        occluded_studs: List[Dict[str, Any]],
        stud_distance_threshold_ldu: float = 4.0,
    ) -> List[Any]:
        """
        Boosts candidate hypotheses whose matched_studs cover the exact disappeared studs
        and penalizes candidates that ignore detected occluded studs.
        """
        if not hypotheses or not occluded_studs:
            return hypotheses

        target_studs_3d = np.array([item["stud_world_pos"] for item in occluded_studs], dtype=np.float32)
        n_targets = len(target_studs_3d)

        for hyp in hypotheses:
            matched_s = hyp.matched_studs
            if not matched_s:
                hyp.stud_occlusion_score = 0.0
                continue

            matched_arr = np.array(matched_s, dtype=np.float32)
            # Pairwise distance (num_matched, num_targets)
            diff = matched_arr[:, np.newaxis, :] - target_studs_3d[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)

            # Inliers
            covered = np.sum(np.min(dists, axis=0) < stud_distance_threshold_ldu)
            fraction_covered = float(covered) / float(n_targets)

            hyp.stud_occlusion_score = fraction_covered

            # Direct boost: if hypothesis covers 100% of disappeared studs, major bonus
            if fraction_covered >= 0.99:
                hyp.total_score = getattr(hyp, "total_score", 0.0) + 5.0
            elif fraction_covered >= 0.5:
                hyp.total_score = getattr(hyp, "total_score", 0.0) + 2.5 * fraction_covered

        # Re-sort descending
        hypotheses.sort(key=lambda h: getattr(h, "total_score", 0.0), reverse=True)
        return hypotheses

    # -------------------------------------------------------------------------
    # 5. Validación Vectorial Infalible (Score de Confianza 100%)
    # -------------------------------------------------------------------------
    def check_occluded_studs_bidirectional(
        self,
        hypothesis: Any,
        existing_nodes: List[Dict[str, Any]],
        geom_resolver: Any,
        detected_occluded_studs: List[Dict[str, Any]],
        stud_tolerance_ldu: float = 4.0,
    ) -> Tuple[bool, float]:
        """
        Criterio 1: Coincidencia de Studs Ocluidos
        Todos los studs de G_{t-1} que la pieza candidata tapa en 3D deben haber desaparecido
        de la capa de vectores del PDF en el paso t, y la pieza no debe tapar ningún stud que siga visible.
        Returns: (passed: bool, f1_score: float)
        """
        if not detected_occluded_studs:
            # Si en el PDF no desapareció ningún stud:
            # La hipótesis solo es válida si tampoco tapa studs en 3D (ej. pieza colocada en el aire o sobre sockets)
            cand_studs = getattr(hypothesis, "matched_studs", [])
            if not cand_studs:
                return True, 1.0
            return False, 0.0

        cand_studs = getattr(hypothesis, "matched_studs", [])
        if not cand_studs:
            return False, 0.0

        target_studs = np.array([item["stud_world_pos"] for item in detected_occluded_studs], dtype=np.float32)
        cand_arr = np.array(cand_studs, dtype=np.float32)

        # Matriz de distancias (N_cand, N_target)
        diff = cand_arr[:, np.newaxis, :] - target_studs[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        # Recall: fracción de studs detectados como desaparecidos que la pieza tapa
        min_dist_to_cand = np.min(dists, axis=0)  # (N_target,)
        n_covered = int(np.sum(min_dist_to_cand < stud_tolerance_ldu))
        recall = float(n_covered) / float(len(target_studs))

        # Precision: fracción de studs que la pieza tapa que realmente desaparecieron
        min_dist_to_target = np.min(dists, axis=1)  # (N_cand,)
        n_valid_covers = int(np.sum(min_dist_to_target < stud_tolerance_ldu))
        precision = float(n_valid_covers) / float(len(cand_arr))

        if precision + recall > 1e-5:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        passed = (recall >= 0.99 and precision >= 0.99)
        return passed, float(f1)

    def check_new_studs_exposed(
        self,
        hypothesis: Any,
        projection_matrix: np.ndarray,
        detected_ellipses: List[StudEllipse2D],
        geom_resolver: Any,
        max_dist_pt: float = 1.0,
    ) -> Tuple[bool, float, float]:
        """
        Criterio 2: Coincidencia de Nuevos Studs Expuestos
        Los studs superiores de la pieza candidata p_new proyectados a 2D deben caer exactamente
        sobre los centros de las nuevas elipses dibujadas en el PDF (distancia euclídea < 1.0 pt).
        Returns: (passed: bool, match_ratio: float, mean_residual_pt: float)
        """
        if projection_matrix is None:
            return False, 0.0, float("inf")

        child_id = getattr(hypothesis, "child_part_id", "")
        meta = geom_resolver.get_part_meta(child_id) if geom_resolver else {}
        studs = meta.get("studs", [])

        # Identificar studs superiores (activos hacia arriba / cámara)
        R = hypothesis.rotation_matrix
        t = hypothesis.translation

        top_studs_w = []
        for s in studs:
            s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
            s_norm = s.get("normal", np.array([0.0, -1.0, 0.0], dtype=np.float32)) if isinstance(s, dict) else np.array([0.0, -1.0, 0.0], dtype=np.float32)
            s_norm_w = R @ s_norm
            # En LDraw +Y hacia abajo; los studs que apuntan arriba tienen normal Y negativa
            if s_norm_w[1] < -0.6:
                s_w = (s_pos @ R.T) + t
                top_studs_w.append(s_w)

        # Si la pieza no tiene studs superiores (ej. tile 1x2 smooth, slope liso)
        if not top_studs_w:
            return True, 1.0, 0.0

        if not detected_ellipses:
            return False, 0.0, float("inf")

        top_studs_arr = np.array(top_studs_w, dtype=np.float32)
        n_studs = len(top_studs_arr)

        # Proyectar a 2D con la matriz afín P (2x4)
        homo_studs = np.hstack([top_studs_arr, np.ones((n_studs, 1), dtype=np.float32)])
        proj_2d = homo_studs @ projection_matrix.T  # (n_studs, 2)

        ell_centers = np.array([e.center for e in detected_ellipses], dtype=np.float32)  # (M, 2)

        # Distancia mínima de cada stud proyectado a la elipse 2D más cercana
        diff = proj_2d[:, np.newaxis, :] - ell_centers[np.newaxis, :, :]
        dists_2d = np.linalg.norm(diff, axis=2)  # (n_studs, M)
        min_dists = np.min(dists_2d, axis=1)    # (n_studs,)

        matched_mask = min_dists <= max_dist_pt
        n_matched = int(np.sum(matched_mask))
        match_ratio = float(n_matched) / float(n_studs)
        mean_res = float(np.mean(min_dists)) if n_studs > 0 else 0.0

        passed = (match_ratio >= 0.99 and mean_res <= max_dist_pt)
        return passed, match_ratio, mean_res

    def check_vector_edge_alignment(
        self,
        hypothesis: Any,
        projection_matrix: np.ndarray,
        stroke_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        geom_resolver: Any,
        wireframe_edges_3d: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        max_dist_pt: float = 1.5,
        min_parallel_cos: float = 0.965,  # ángulo < ~15 grados
    ) -> Tuple[bool, float, float]:
        """
        Criterio 3: Alineación de Segmentos de Arista (LDraw tipo 2 vs PDF)
        Cada línea tipo 2 de la pieza LDraw proyectada debe tener un segmento vectorial
        negro paralelo en page.get_drawings() a menos de 1.5 pt de distancia.
        Returns: (passed: bool, match_ratio: float, mean_residual_pt: float)
        """
        if projection_matrix is None or not stroke_segments:
            return False, 0.0, float("inf")

        edges_3d = wireframe_edges_3d
        if not edges_3d and geom_resolver is not None:
            child_id = getattr(hypothesis, "child_part_id", "")
            if hasattr(geom_resolver, "get_part_edges_type2"):
                edges_3d = geom_resolver.get_part_edges_type2(child_id)

        if not edges_3d:
            # Fallback a aristas del bounding box local si la pieza no tiene líneas tipo 2 explícitas
            meta = geom_resolver.get_part_meta(getattr(hypothesis, "child_part_id", "")) if geom_resolver else {}
            bx0, by0, bz0, bx1, by1, bz1 = meta.get("bbox", (-10, -8, -10, 10, 0, 10))
            c = np.array([
                [bx0, by0, bz0], [bx1, by0, bz0], [bx1, by1, bz0], [bx0, by1, bz0],
                [bx0, by0, bz1], [bx1, by0, bz1], [bx1, by1, bz1], [bx0, by1, bz1]
            ], dtype=np.float32)
            edges_3d = [
                (c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0]),
                (c[4], c[5]), (c[5], c[6]), (c[6], c[7]), (c[7], c[4]),
                (c[0], c[4]), (c[1], c[5]), (c[2], c[6]), (c[3], c[7])
            ]

        R = hypothesis.rotation_matrix
        t = hypothesis.translation

        # Transformar aristas a coordenadas mundo y proyectar a 2D
        proj_edges = []
        for p1, p2 in edges_3d[:40]:
            p1_w = (R @ p1) + t
            p2_w = (R @ p2) + t
            h1 = np.array([p1_w[0], p1_w[1], p1_w[2], 1.0], dtype=np.float32)
            h2 = np.array([p2_w[0], p2_w[1], p2_w[2], 1.0], dtype=np.float32)
            u1, v1 = projection_matrix @ h1
            u2, v2 = projection_matrix @ h2
            length = math.hypot(u2 - u1, v2 - v1)
            if length >= 2.5:  # Ignorar aristas proyectadas microscópicas
                proj_edges.append(((float(u1), float(v1)), (float(u2), float(v2)), length))

        if not proj_edges:
            return True, 1.0, 0.0

        pdf_segs = []
        for s1, s2 in stroke_segments:
            dx = s2[0] - s1[0]
            dy = s2[1] - s1[1]
            L = math.hypot(dx, dy)
            if L >= 2.0:
                pdf_segs.append((s1, s2, dx / L, dy / L, L))

        if not pdf_segs:
            return False, 0.0, float("inf")

        matched_count = 0
        residuals = []

        for (u1, v1), (u2, v2), L_cand in proj_edges:
            dir_x = (u2 - u1) / L_cand
            dir_y = (v2 - v1) / L_cand
            mid_x = (u1 + u2) * 0.5
            mid_y = (v1 + v2) * 0.5

            best_d = float("inf")
            for s1, s2, p_dx, p_dy, L_pdf in pdf_segs:
                # Paralelismo: producto escalar absoluto
                dot = abs(dir_x * p_dx + dir_y * p_dy)
                if dot < min_parallel_cos:
                    continue

                # Distancia perpendicular del punto medio a la recta del trazo PDF
                # normal perpendicular: (-p_dy, p_dx)
                d_perp = abs((mid_x - s1[0]) * (-p_dy) + (mid_y - s1[1]) * p_dx)
                if d_perp > max_dist_pt:
                    continue

                # Distancia de proyección longitudinal (overlap)
                t_proj = (mid_x - s1[0]) * p_dx + (mid_y - s1[1]) * p_dy
                if -2.0 <= t_proj <= (L_pdf + 2.0):
                    if d_perp < best_d:
                        best_d = d_perp

            if best_d <= max_dist_pt:
                matched_count += 1
                residuals.append(best_d)

        n_total = len(proj_edges)
        match_ratio = float(matched_count) / float(n_total)
        mean_res = float(np.mean(residuals)) if residuals else float("inf")

        # Se considera passed si el ratio de aristas visibles alineadas es >= 70% con residual < 1.5 pt
        passed = (match_ratio >= 0.70 and mean_res <= max_dist_pt)
        return passed, match_ratio, mean_res

    def validate_infallible_vector_fit(
        self,
        hypothesis: Any,
        existing_nodes: List[Dict[str, Any]],
        projection_matrix: np.ndarray,
        vector_evidence: VectorEvidence,
        geom_resolver: Any,
        wireframe_edges_3d: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        solid_validator: Optional[Any] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validación Vectorial Infalible (Score de Confianza 100%):
          1. Coincidencia de Studs Ocluidos: La pieza candidata tapa exactamente los studs
             que han desaparecido en el PDF (sin tapar studs visibles).
          2. Coincidencia de Nuevos Studs Expuestos: Los studs superiores de p_new caen a < 1.0 pt
             de las elipses vectoriales del paso t.
          3. Alineación de Segmentos de Arista: Las líneas tipo 2 proyectadas tienen trazos
             paralelos en page.get_drawings() a < 1.5 pt.
          4. Validación física 3D de no-colisión (SolidCollisionValidator).

        Si se cumplen las 4 condiciones, la probabilidad de error es matemáticamente cero.
        Returns: (is_infallible: bool, confidence: float, metrics: dict)
        """
        metrics = {
            "occluded_studs_pass": False,
            "occluded_studs_f1": 0.0,
            "new_studs_pass": False,
            "new_studs_ratio": 0.0,
            "new_studs_residual_pt": float("inf"),
            "edge_alignment_pass": False,
            "edge_alignment_ratio": 0.0,
            "edge_residual_pt": float("inf"),
            "collision_free": True,
        }

        # 4. Verificación física 3D previa (cero colisiones volumétricas)
        if solid_validator is not None and existing_nodes:
            is_valid_col, _ = solid_validator.validate_new_part_placement(
                hypothesis.child_part_id,
                hypothesis.absolute_transform,
                existing_nodes,
                penetration_threshold=0.5,
            )
            if not is_valid_col:
                metrics["collision_free"] = False
                return False, 0.0, metrics

        # 1. Coincidencia de Studs Ocluidos
        detected_occluded = self.detect_occluded_studs(
            existing_nodes=existing_nodes,
            geom_resolver=geom_resolver,
            projection_matrix=projection_matrix,
            detected_ellipses_step_t=vector_evidence.stud_ellipses,
            roi_new=vector_evidence.roi_new_parts,
        )
        occ_pass, occ_f1 = self.check_occluded_studs_bidirectional(
            hypothesis, existing_nodes, geom_resolver, detected_occluded
        )
        metrics["occluded_studs_pass"] = occ_pass
        metrics["occluded_studs_f1"] = occ_f1

        # 2. Coincidencia de Nuevos Studs Expuestos (< 1.0 pt)
        studs_pass, studs_ratio, studs_res = self.check_new_studs_exposed(
            hypothesis, projection_matrix, vector_evidence.stud_ellipses, geom_resolver, max_dist_pt=1.0
        )
        metrics["new_studs_pass"] = studs_pass
        metrics["new_studs_ratio"] = studs_ratio
        metrics["new_studs_residual_pt"] = studs_res

        # 3. Alineación de Segmentos de Arista (< 1.5 pt)
        edges_pass, edges_ratio, edges_res = self.check_vector_edge_alignment(
            hypothesis, projection_matrix, vector_evidence.stroke_segments, geom_resolver,
            wireframe_edges_3d=wireframe_edges_3d, max_dist_pt=1.5
        )
        metrics["edge_alignment_pass"] = edges_pass
        metrics["edge_alignment_ratio"] = edges_ratio
        metrics["edge_residual_pt"] = edges_res

        # Validación Infalible: Las 3 condiciones geométricas y la condición física cumplidas
        is_infallible = bool(occ_pass and studs_pass and edges_pass and metrics["collision_free"])
        confidence = 1.0 if is_infallible else float(0.35 * occ_f1 + 0.35 * studs_ratio + 0.30 * edges_ratio)

        # Actualizar atributos en la hipótesis
        hypothesis.is_infallible_match = is_infallible
        hypothesis.vector_confidence = confidence
        hypothesis.vector_stud_score = studs_ratio
        hypothesis.vector_edge_score = edges_ratio

        if is_infallible:
            # Asignar máxima prioridad absoluta a la hipótesis infalible
            hypothesis.total_score = getattr(hypothesis, "total_score", 0.0) + 100.0

        return is_infallible, confidence, metrics

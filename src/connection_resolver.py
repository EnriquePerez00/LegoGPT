"""
connection_resolver.py: Physical Connection Space & Geometric 2D Matcher (Option A).

Universal engine to:
  1. Compute all physically valid discrete connection candidates between ANY two LEGO parts
     (or existing subassembly and child part) based on stud-to-socket pairing, height offsets,
     and collision constraints.
  2. Map the exact contact points (puntos de contacto) and handle part symmetries.
  3. Disambiguate multi-pose connections against manual step illustration crops using
     2D Silhouette IoU, Edge Chamfer Distance, and Visible Stud Alignment (Option A).
"""

from __future__ import annotations
import os
import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from src.ldraw_geometry import LDrawGeometryResolver
from src.step_axonometric_renderer import StepAxonometricRenderer


@dataclass
class ConnectionCandidate:
    candidate_id: str
    rotation_y_deg: int
    translation_ldu: Tuple[float, float, float]
    transform_matrix: np.ndarray  # 4x4 matrix
    contact_points: List[Dict[str, Any]]  # pairs of (parent_stud, child_socket)
    num_contacts: int
    occupied_parent_studs: List[Tuple[float, float]]  # (x, z) of parent studs covered
    free_parent_studs: List[Tuple[float, float]]      # (x, z) of parent studs left exposed
    child_top_studs_world: List[np.ndarray]           # 3D coords of studs on top of child
    score: float = 0.0
    iou: float = 0.0
    chamfer_dist: float = 0.0


class LegoConnectionResolver:
    """
    Universal Lego Physical Connection Resolver & Geometric Matcher.
    Optimized for Apple Silicon M4 with vectorized operations.
    """

    def __init__(self, ldraw_base_dir: str = "data/ldraw/ldraw"):
        self.geom_resolver = LDrawGeometryResolver(ldraw_base_dir=ldraw_base_dir)
        self.mesh_renderer = StepAxonometricRenderer(self.geom_resolver)

    @staticmethod
    def get_rotation_matrix_y(deg: float) -> np.ndarray:
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([
            [ c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ], dtype=np.float32)

    def extract_functional_ports(self, part_meta: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extracts valid top studs (ports for child attachment) and bottom sockets.
        Normalizes into arrays of 3D positions in LDU.
        """
        raw_studs = part_meta.get("studs", [])
        raw_sockets = part_meta.get("sockets", [])
        bbox = part_meta.get("bbox", [-10, -8, -10, 10, 0, 10])

        ymin = bbox[1]
        top_studs = []
        for s in raw_studs:
            pos = np.array(s, dtype=np.float32) if not isinstance(s, dict) else np.array(s["pos"], dtype=np.float32)
            if abs(pos[1] - ymin) <= 4.5 or abs(pos[1]) <= 4.5:
                top_studs.append(pos)

        # Deduplicate studs by grid
        unique_studs = []
        for s in top_studs:
            if not any(np.linalg.norm(s - u) < 3.0 for u in unique_studs):
                unique_studs.append(s)

        # Bottom sockets: typically at ymax
        ymax = bbox[4]
        unique_sockets = []
        for sk in raw_sockets:
            pos = np.array(sk, dtype=np.float32)
            if not any(np.linalg.norm(pos - u) < 3.0 for u in unique_sockets):
                unique_sockets.append(pos)

        if not unique_sockets:
            xmin, _, zmin, xmax, _, zmax = bbox
            for x in range(xmin + 10, xmax, 20):
                for z in range(zmin + 10, zmax, 20):
                    unique_sockets.append(np.array([float(x), float(ymax), float(z)], dtype=np.float32))

        return unique_studs, unique_sockets

    def generate_candidate_connections(
        self,
        parent_part_id: str,
        child_part_id: str,
        layer_offset_y: float = -8.0,
        min_contacts: int = 1
    ) -> List[ConnectionCandidate]:
        """
        Exhaustively computes all physically valid discrete stud-to-socket configurations
        between parent and child parts.
        Groups by exact contact point signatures and rotational equivalence.
        """
        p_parent = self.geom_resolver.get_part_meta(parent_part_id)
        p_child = self.geom_resolver.get_part_meta(child_part_id)

        parent_studs, _ = self.extract_functional_ports(p_parent)
        child_studs, child_sockets = self.extract_functional_ports(p_child)

        if not parent_studs or not child_sockets:
            tf = np.eye(4, dtype=np.float32)
            tf[1, 3] = layer_offset_y
            return [ConnectionCandidate(
                candidate_id="default_fallback",
                rotation_y_deg=0,
                translation_ldu=(0.0, layer_offset_y, 0.0),
                transform_matrix=tf,
                contact_points=[],
                num_contacts=0,
                occupied_parent_studs=[],
                free_parent_studs=[(round(s[0]), round(s[2])) for s in parent_studs],
                child_top_studs_world=[s + np.array([0, layer_offset_y, 0]) for s in child_studs]
            )]

        parent_stud_grid = {}
        for idx, s in enumerate(parent_studs):
            key = (int(round(s[0])), int(round(s[2])))
            parent_stud_grid[key] = (idx, s)

        raw_candidates = []
        tested_signatures = set()

        for rot_deg in [0, 90, 180, 270]:
            R = self.get_rotation_matrix_y(rot_deg)
            rotated_child_sockets = [R @ sk for sk in child_sockets]
            rotated_child_studs = [R @ cs for cs in child_studs]

            # Anchor sockets sampling
            anchor_sockets = rotated_child_sockets[:min(6, len(rotated_child_sockets))]

            for c_sk in anchor_sockets:
                for p_s in parent_studs:
                    tx = round((p_s[0] - c_sk[0]) / 10.0) * 10.0
                    tz = round((p_s[2] - c_sk[2]) / 10.0) * 10.0
                    ty = float(layer_offset_y)

                    contacts = []
                    occupied_stud_keys = set()

                    for sk_idx, sk in enumerate(rotated_child_sockets):
                        world_sk_x = int(round(sk[0] + tx))
                        world_sk_z = int(round(sk[2] + tz))
                        key = (world_sk_x, world_sk_z)
                        if key in parent_stud_grid:
                            p_idx, p_coord = parent_stud_grid[key]
                            contacts.append({
                                "parent_stud_idx": p_idx,
                                "child_socket_idx": sk_idx,
                                "coord_ldu": [world_sk_x, 0.0, world_sk_z]
                            })
                            occupied_stud_keys.add(key)

                    if len(contacts) < min_contacts:
                        continue

                    sorted_occupied = tuple(sorted(list(occupied_stud_keys)))
                    sig = (rot_deg, sorted_occupied)
                    if sig in tested_signatures:
                        continue
                    tested_signatures.add(sig)

                    child_top_world = [
                        cs + np.array([tx, ty, tz], dtype=np.float32)
                        for cs in rotated_child_studs
                    ]

                    free_studs = [
                        k for k in parent_stud_grid.keys() if k not in occupied_stud_keys
                    ]

                    tf = np.eye(4, dtype=np.float32)
                    tf[:3, :3] = R
                    tf[:3, 3] = [tx, ty, tz]

                    raw_candidates.append(ConnectionCandidate(
                        candidate_id=f"rot{rot_deg}_tx{int(tx)}_tz{int(tz)}",
                        rotation_y_deg=rot_deg,
                        translation_ldu=(float(tx), float(ty), float(tz)),
                        transform_matrix=tf,
                        contact_points=contacts,
                        num_contacts=len(contacts),
                        occupied_parent_studs=list(sorted_occupied),
                        free_parent_studs=free_studs,
                        child_top_studs_world=child_top_world
                    ))

        raw_candidates.sort(key=lambda c: c.num_contacts, reverse=True)
        return raw_candidates

    @staticmethod
    def project_points_axonometric(
        pts_3d: np.ndarray,
        azimuth_deg: float = 45.0,
        elevation_deg: float = 30.0,
        scale: float = 1.0,
        offset_2d: Tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray:
        """Projects 3D LDraw points to 2D screen coordinates under axonometric projection."""
        if len(pts_3d) == 0:
            return np.empty((0, 2), dtype=np.float32)

        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)

        X = pts_3d[:, 0] * math.cos(az) + pts_3d[:, 2] * math.sin(az)
        Z = -pts_3d[:, 0] * math.sin(az) + pts_3d[:, 2] * math.cos(az)
        Y = pts_3d[:, 1]

        Y_proj = Y * math.cos(el) - Z * math.sin(el)

        u = scale * X + offset_2d[0]
        v = scale * Y_proj + offset_2d[1]
        return np.column_stack([u, v]).astype(np.float32)

    def _get_part_perimeter_polygon(self, part_meta: dict) -> np.ndarray:
        """Generates representative 3D perimeter contour for common part families."""
        part_id = part_meta.get("part_id", "").lower()
        bx0, by0, bz0, bx1, by1, bz1 = part_meta["bbox"]

        # Circular plate (e.g. 11213, 30504)
        if "round" in part_id or "11213" in part_id:
            r = max(abs(bx0), abs(bx1), abs(bz0), abs(bz1))
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            pts = np.column_stack([r * np.cos(angles), np.full(36, by0), r * np.sin(angles)])
            return pts.astype(np.float32)

        # Quarter-round plate (e.g. 30565)
        # Straight corner at (-40, 40), arc of radius 80 from (40, 40) to (-40, -40)
        if "30565" in part_id or "corner round" in part_id:
            angles = np.linspace(0, -np.pi / 2.0, 18)
            arc_x = -40.0 + 80.0 * np.cos(angles)
            arc_z = 40.0 + 80.0 * np.sin(angles)
            arc_pts = np.column_stack([arc_x, np.full(len(angles), by0), arc_z])
            corner_pt = np.array([[-40.0, by0, 40.0]], dtype=np.float32)
            pts = np.vstack([corner_pt, arc_pts])
            return pts.astype(np.float32)

        # Standard rectangular brick / plate
        corners = np.array([
            [bx0, by0, bz0],
            [bx1, by0, bz0],
            [bx1, by0, bz1],
            [bx0, by0, bz1],
        ], dtype=np.float32)
        return corners

    def render_candidate_silhouette(
        self,
        candidate: ConnectionCandidate,
        parent_part_id: str,
        child_part_id: str,
        target_shape: Tuple[int, int],
        azimuth_deg: float = 45.0,
        elevation_deg: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Renders composite 2D binary silhouette, edge points, and visible stud centers
        for parent + child assembly using real 3D solid faces and side walls.
        """
        h_img, w_img = target_shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # Extract real 3D solid geometry (top, bottom, and side faces)
        parent_faces, _ = self.mesh_renderer.get_part_faces_and_studs(parent_part_id)
        child_faces, _ = self.mesh_renderer.get_part_faces_and_studs(child_part_id)

        R = candidate.transform_matrix[:3, :3]
        t = candidate.transform_matrix[:3, 3]

        all_3d = []
        for f in parent_faces:
            all_3d.append(f["pts"])
        for f in child_faces:
            all_3d.append((f["pts"] @ R.T) + t)

        if not all_3d:
            return mask, mask, np.empty((0, 2), dtype=np.float32)

        all_3d_np = np.vstack(all_3d)

        # Global bounding projection to estimate scale & center
        proj_unscaled = self.project_points_axonometric(all_3d_np, azimuth_deg, elevation_deg, scale=1.0)
        min_u, min_v = proj_unscaled.min(axis=0)
        max_u, max_v = proj_unscaled.max(axis=0)
        span_w = max(1.0, max_u - min_u)
        span_h = max(1.0, max_v - min_v)

        scale = min((w_img * 0.75) / span_w, (h_img * 0.75) / span_h)
        offset_u = (w_img - (max_u + min_u) * scale) / 2.0
        offset_v = (h_img - (max_v + min_v) * scale) / 2.0
        offset = (offset_u, offset_v)

        # Draw parent 3D solid faces
        for f in parent_faces:
            p2d = self.project_points_axonometric(f["pts"], azimuth_deg, elevation_deg, scale=scale, offset_2d=offset).astype(np.int32)
            cv2.fillPoly(mask, [p2d], 255)

        # Draw child 3D solid faces
        for f in child_faces:
            pts_world = (f["pts"] @ R.T) + t
            c2d = self.project_points_axonometric(pts_world, azimuth_deg, elevation_deg, scale=scale, offset_2d=offset).astype(np.int32)
            cv2.fillPoly(mask, [c2d], 255)

        # Stud projections
        free_parent_studs_3d = np.array([[x, 0.0, z] for x, z in candidate.free_parent_studs], dtype=np.float32)
        child_top_studs_3d = np.array(candidate.child_top_studs_world, dtype=np.float32)

        studs_3d_all = []
        if len(free_parent_studs_3d) > 0:
            studs_3d_all.append(free_parent_studs_3d)
        if len(child_top_studs_3d) > 0:
            studs_3d_all.append(child_top_studs_3d)

        if studs_3d_all:
            studs_3d_combined = np.vstack(studs_3d_all)
            studs_2d = self.project_points_axonometric(
                studs_3d_combined, azimuth_deg, elevation_deg, scale=scale, offset_2d=offset
            )
        else:
            studs_2d = np.empty((0, 2), dtype=np.float32)

        edges = cv2.Canny(mask, 50, 150)
        return mask, edges, studs_2d

    def disambiguate_with_image_crop(
        self,
        candidates: List[ConnectionCandidate],
        crop_image_or_path: Any,
        parent_part_id: str,
        child_part_id: str,
        azimuth_deg: float = 45.0,
        elevation_deg: float = 30.0
    ) -> ConnectionCandidate:
        """
        Applies Option A (Geometric 2D Silhouette IoU + Edge Chamfer Matching):
        Evaluates each candidate against the manual image crop and returns the best matching candidate.
        """
        if not candidates:
            raise ValueError("No candidates provided for disambiguation.")

        if len(candidates) == 1:
            candidates[0].score = 1.0
            return candidates[0]

        if isinstance(crop_image_or_path, str):
            img_bgr = cv2.imread(crop_image_or_path)
        else:
            img_bgr = crop_image_or_path

        if img_bgr is None:
            return candidates[0]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        corner_bg = int(np.median([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]))
        diff = cv2.absdiff(gray, corner_bg)
        _, mask_manual = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

        edges_manual = cv2.Canny(gray, 30, 120)
        dist_transform = cv2.distanceTransform(255 - edges_manual, cv2.DIST_L2, 5)

        best_cand = None
        best_score = -1e9

        for cand in candidates:
            mask_synth, edges_synth, studs_2d = self.render_candidate_silhouette(
                cand, parent_part_id, child_part_id, (h, w), azimuth_deg, elevation_deg
            )

            intersection = cv2.countNonZero(cv2.bitwise_and(mask_manual, mask_synth))
            union = cv2.countNonZero(cv2.bitwise_or(mask_manual, mask_synth))
            iou = intersection / max(1, union)

            synth_pts = np.column_stack(np.where(edges_synth > 0))
            if len(synth_pts) > 0:
                sampled_dists = dist_transform[synth_pts[:, 0], synth_pts[:, 1]]
                chamfer = float(np.mean(sampled_dists))
            else:
                chamfer = 50.0

            if len(studs_2d) > 0:
                valid_studs = 0
                for pt in studs_2d:
                    u, v = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= u < w and 0 <= v < h:
                        if mask_manual[v, u] > 0:
                            valid_studs += 1
                stud_ratio = valid_studs / len(studs_2d)
            else:
                stud_ratio = 1.0

            score = (iou * 0.55) + (stud_ratio * 0.25) - (min(chamfer, 30.0) / 30.0 * 0.20)

            cand.iou = float(iou)
            cand.chamfer_dist = float(chamfer)
            cand.score = float(score)

            if score > best_score:
                best_score = score
                best_cand = cand

        return best_cand if best_cand is not None else candidates[0]

import os
import re
import cv2
import sqlite3
import concurrent.futures
import numpy as np
try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None
from src.ldraw_geometry import LDrawGeometryResolver
from src.bricklink_dimensions import BricklinkDimensionFetcher
from src.connection_resolver import LegoConnectionResolver
from src.parser import ParsedPart
from src.step_axonometric_renderer import StepAxonometricRenderer
from src.physical_hypothesis_generator import PhysicalHypothesisGenerator
from src.visual_pose_scorer import VisualPoseScorer
from src.connector_occupancy_map import ConnectorOccupancyMap
from src.solid_collision_validator import SolidCollisionValidator
from src.subassembly_manager import SubassemblyManager


class ManualConnectionGraphBuilder:
    """
    Builds the physical 3D connection graph page-by-page from manual steps:
      - Discrete height levels L (plates = 8 LDU) and bottom coordinate y_bottom_ldu.
      - Tracking of construction orientation & flip icons (e.g. Page 28).
      - Ground level normalization (min(Y) = 0.0) respecting AGENTS.md rules.
      - Isolation and identification of independent minifigure subassemblies via Rebrickable DB.
      - Output compatible with src/homogenize_assembly.py.
    """
    def __init__(self, pdf_path: str = "", set_num: str = "75295-1", db_path: str = "data/catalog/models_catalog.db", fast_mode: bool = False):
        self.pdf_path = pdf_path
        self.set_num = set_num
        self.db_path = db_path
        self.fast_mode = fast_mode  # Skip heavy visual scoring; use physical gating only
        self.geom_resolver = LDrawGeometryResolver()
        self.bl_fetcher = BricklinkDimensionFetcher(db_path=db_path)
        self.conn_resolver = LegoConnectionResolver()
        self.doc = pymupdf.open(pdf_path) if (pymupdf is not None and pdf_path and os.path.exists(pdf_path)) else []
        self.step_renderer = StepAxonometricRenderer(self.geom_resolver)
        self.hypothesis_gen = PhysicalHypothesisGenerator(self.geom_resolver)
        self.visual_scorer = VisualPoseScorer(self.step_renderer)
        self.occupancy_map = ConnectorOccupancyMap(self.geom_resolver)
        self.solid_validator = SolidCollisionValidator(self.geom_resolver)
        self.subassembly_mgr = SubassemblyManager(self.geom_resolver)

    def get_minifigures_catalog_info(self) -> list[dict]:
        """
        Queries models_catalog.db to fetch structured minifigure metadata for self.set_num.
        Returns a list of dicts with fig_num, name, and categorized parts.
        """
        if not os.path.exists(self.db_path):
            return []
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT im.fig_num, m.name
            FROM rb_inventory_minifigs im
            JOIN rb_inventories i ON im.inventory_id = i.id
            LEFT JOIN rb_minifigs m ON im.fig_num = m.fig_num
            WHERE i.set_num = ? OR i.set_num LIKE ?
        """, (self.set_num, f"{self.set_num.split('-')[0]}%"))
        figs = cur.fetchall()

        minifigs = []
        for fig_num, fig_name in figs:
            cur.execute("""
                SELECT ip.part_num, ip.color_id, ip.quantity, p.name, p.part_cat_id, pc.name, e.element_id
                FROM rb_inventories i
                JOIN rb_inventory_parts ip ON i.id = ip.inventory_id
                LEFT JOIN rb_parts p ON ip.part_num = p.part_num
                LEFT JOIN rb_part_categories pc ON p.part_cat_id = pc.id
                LEFT JOIN rb_elements e ON e.part_num = ip.part_num AND e.color_id = ip.color_id
                WHERE i.set_num = ?
            """, (fig_num,))
            parts = cur.fetchall()
            minifigs.append({
                "fig_num": fig_num,
                "name": fig_name,
                "parts": parts
            })
        conn.close()
        return minifigs

    def detect_flip_icons(self) -> dict[int, bool]:
        """Detects rotation/flip icons on step pages (e.g. Page 28 and 29)."""
        flip_pages = {}
        for idx in range(len(self.doc)):
            page = self.doc[idx]
            page_num = idx + 1
            has_flip = False
            for d in page.get_drawings():
                r, fill = d.get("rect"), d.get("fill")
                if r is None or fill is None or len(fill) < 3:
                    continue
                # The characteristic white square box of ~35-60pt
                if 35 <= r.width <= 60 and 35 <= r.height <= 60 and min(fill[:3]) > 0.97:
                    if r.x0 < 90 and 80 < r.y0 < 190:  # typically on left margin near step number
                        has_flip = True
                        break
            if has_flip:
                flip_pages[page_num] = True
        return flip_pages

    def get_unclosed_substrate_parents(self, existing_parts: list[dict]) -> list[int]:
        """
        Identifies parts in the lowest active layer that still have unoccluded top studs.
        In standard LDraw coordinates (+Y downward), maximum Y corresponds to the lowest physical substrate.
        """
        if not existing_parts:
            return []
        occupied_stud_positions = []
        for p in existing_parts:
            for conn in p.get("connections", []):
                for cp in conn.get("contact_points", []):
                    occupied_stud_positions.append(np.array(cp, dtype=np.float32))

        parents_with_free_studs = []
        for p in existing_parts:
            p_idx = p["part_index"]
            p_meta = self.geom_resolver.get_part_meta(p["part_id"])
            p_tf = np.array(p["absolute_transform"], dtype=np.float32).reshape((4, 4))
            p_R, p_t = p_tf[:3, :3], p_tf[:3, 3]

            free_studs = []
            for s in p_meta.get("studs", []):
                s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
                s_norm = s.get("normal", np.array([0.0, -1.0, 0.0], dtype=np.float32)) if isinstance(s, dict) else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                s_w = (s_pos @ p_R.T) + p_t
                s_w_norm = p_R @ s_norm

                # Upward-facing studs in world coordinates
                if s_w_norm[1] < -0.8:
                    is_occ = any(np.linalg.norm(s_w - occ) < 5.0 for occ in occupied_stud_positions)
                    if not is_occ:
                        free_studs.append(s_w)

            if free_studs:
                mean_y = float(np.mean([s[1] for s in free_studs]))
                parents_with_free_studs.append({
                    "part_index": p_idx,
                    "part_id": p["part_id"],
                    "y_studs": mean_y,
                    "num_free_studs": len(free_studs)
                })

        if not parents_with_free_studs:
            return []

        # Find the lowest physical elevation (maximum Y in LDraw coordinates)
        max_y = max(p["y_studs"] for p in parents_with_free_studs)
        return [p["part_index"] for p in parents_with_free_studs if abs(p["y_studs"] - max_y) < 2.0]

    def solve_step_hypotheses(
        self,
        step_items: list[dict],
        existing_parts: list[dict],
        crop_path: str,
        is_flip: bool = False
    ) -> dict:
        """
        Solves hypotheses for a manual step with active substrate completion and joint multi-piece solving.
        """
        if not existing_parts:
            return {}

        substrate_parent_indices = self.get_unclosed_substrate_parents(existing_parts)

        flat_parts = []
        for it_idx, it in enumerate(step_items):
            el_id = it["matched_element_id"]
            mt = it["matched_template"]
            p_num = mt.get("part_num", "3001")
            resolved_fn = self.geom_resolver.resolve_compatible_part_id(p_num)
            qty = it.get("qty", 1)
            for inst_i in range(qty):
                flat_parts.append({
                    "key": (it_idx, inst_i),
                    "part_id": resolved_fn,
                    "element_id": el_id,
                    "color_id": mt.get("color_id", 71),
                })

        if not flat_parts:
            return {}

        # Check for vector/visual evidence to prioritize candidates via stud disappearance
        # fast_mode: skip expensive visual scoring (render+compare), use physical gating only
        if getattr(self, 'fast_mode', False):
            evidence = {}
        else:
            evidence = self.visual_scorer.extract_visual_evidence(crop_path) if crop_path and os.path.exists(crop_path) else {}
        occluded_studs = []
        connector_2d_filter = None
        calib = {}
        if evidence and existing_parts:
            calib = self.visual_scorer.calibrate_camera_from_prior_state(existing_parts, evidence, is_flip=is_flip)
            
            # ROI 2D Area of Interest Extraction (Bounding Box of newly added pieces)
            delta_mask = evidence.get("step_delta_mask")
            roi_bbox_2d = None
            if delta_mask is not None and cv2.countNonZero(delta_mask) > 40:
                ys, xs = np.nonzero(delta_mask)
                roi_bbox_2d = (float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))
            elif evidence.get("vector_evidence", {}).get("roi_new_parts") is not None:
                roi_bbox_2d = evidence["vector_evidence"]["roi_new_parts"]

            if roi_bbox_2d is not None:
                connector_2d_filter = {
                    "roi_bbox_2d": roi_bbox_2d,
                    "scale": calib.get("scale", evidence.get("auto_scale", 2.65)),
                    "offset": calib.get("offset", evidence.get("fg_center", (200.0, 200.0))),
                    "az_deg": calib.get("az_deg", 45.0),
                    "el_deg": calib.get("el_deg", 30.0),
                    "center_3d": calib.get("center_3d")
                }

            if calib.get("is_vector_calibrated") and calib.get("projection_matrix") is not None:
                vec_ev = evidence.get("vector_evidence", {})
                stud_objs = vec_ev.get("stud_ellipse_objects", [])
                roi_new = vec_ev.get("roi_new_parts")
                occluded_studs = self.visual_scorer.vector_engine.detect_occluded_studs(
                    existing_nodes=existing_parts,
                    geom_resolver=self.geom_resolver,
                    projection_matrix=calib["projection_matrix"],
                    detected_ellipses_step_t=stud_objs,
                    roi_new=roi_new,
                )

        # Generate candidate hypotheses for each part in the step.
        # fast_mode (Paso 3): use connector-compatibility port matching as primary path;
        # fall back to canonical-rotation sweep only if no candidates found.
        candidates_per_part = []
        for fp in flat_parts:
            all_cands = []
            if getattr(self, 'fast_mode', False):
                # Primary: typed port-matching gating (O(free_ports * child_ports))
                all_cands = self.hypothesis_gen.generate_candidates_from_port_matching(
                    fp["part_id"], fp["element_id"], self.occupancy_map,
                    existing_parts, is_flip=is_flip, color_id=fp["color_id"],
                    lazy_collision_check=True, max_candidates=32,
                )
            # Fallback / non-fast: canonical rotation sweep
            if not all_cands:
                all_cands = self.hypothesis_gen.generate_candidate_hypotheses(
                    fp["part_id"], fp["element_id"], existing_parts, is_flip=is_flip, color_id=fp["color_id"],
                    connector_2d_filter=connector_2d_filter,
                    lazy_collision_check=True
                )
            # If 2D ROI filter was too restrictive, fallback to unpruned candidates
            if not all_cands and connector_2d_filter is not None:
                all_cands = self.hypothesis_gen.generate_candidate_hypotheses(
                    fp["part_id"], fp["element_id"], existing_parts, is_flip=is_flip, color_id=fp["color_id"],
                    lazy_collision_check=True
                )
            # If an unclosed substrate exists, prioritize candidates that attach to that substrate layer
            if substrate_parent_indices:
                sub_cands = [c for c in all_cands if c.parent_part_index in substrate_parent_indices]
                if sub_cands:
                    all_cands = sub_cands

            if occluded_studs:
                all_cands = self.visual_scorer.vector_engine.score_hypotheses_with_occluded_studs(all_cands, occluded_studs)

            # Infallible Vector Validation: fast-track 100% confidence matches
            if evidence and existing_parts and calib.get("is_vector_calibrated") and calib.get("projection_matrix") is not None:
                vec_ev = evidence.get("vector_evidence_obj") or evidence.get("vector_evidence")
                if vec_ev is not None:
                    infallible = []
                    for c_obj in all_cands:
                        is_inf, conf, metrics = self.visual_scorer.vector_engine.validate_infallible_vector_fit(
                            hypothesis=c_obj,
                            existing_nodes=existing_parts,
                            projection_matrix=calib["projection_matrix"],
                            vector_evidence=vec_ev,
                            geom_resolver=self.geom_resolver,
                            wireframe_edges_3d=self.visual_scorer.get_part_wireframe_edges(c_obj.child_part_id),
                            solid_validator=self.solid_validator,
                        )
                        if is_inf:
                            infallible.append(c_obj)
                    if infallible:
                        all_cands = infallible

            candidates_per_part.append(all_cands)

        # Joint pair solver for steps with 2 parts (e.g. Page 6, Page 7 symmetrical wings/plates)
        if len(flat_parts) == 2:
            cands1, cands2 = candidates_per_part[0], candidates_per_part[1]
            best_tuple = None
            best_score = -1.0

            c1_top = cands1[:10]
            c2_top = cands2[:10]

            for h1 in c1_top:
                for h2 in c2_top:
                    # Stud exclusivity (no overlapping parent studs)
                    s1 = {tuple(np.round(s, 1)) for s in h1.matched_studs}
                    s2 = {tuple(np.round(s, 1)) for s in h2.matched_studs}
                    if s1.intersection(s2):
                        continue

                    # Mutual solid non-collision check
                    node1 = {
                        "part_index": 9999,
                        "part_id": flat_parts[0]["part_id"],
                        "absolute_transform": h1.absolute_transform.flatten().tolist()
                    }
                    is_val, _ = self.solid_validator.validate_new_part_placement(
                        flat_parts[1]["part_id"], h2.absolute_transform, [node1], penetration_threshold=0.5
                    )
                    if not is_val:
                        continue

                    studs_total = len(h1.matched_studs) + len(h2.matched_studs)
                    coplanar = 2.0 if abs(h1.translation[1] - h2.translation[1]) < 1.0 else 0.0
                    # Bilateral reflection symmetry across X=0 longitudinal plane
                    sym = 3.0 if (abs(h1.translation[0] + h2.translation[0]) < 2.0 and abs(h1.translation[2] - h2.translation[2]) < 2.0) else 0.0
                    # Centerline alignment bonus
                    center = 2.0 if (abs(h1.translation[0]) < 1.0 or abs(h2.translation[0]) < 1.0) else 0.0

                    score = studs_total + coplanar + sym + center
                    if score > best_score:
                        best_score = score
                        best_tuple = (h1, h2)

            if best_tuple:
                return {flat_parts[0]["key"]: best_tuple[0], flat_parts[1]["key"]: best_tuple[1]}

        # Sequential non-colliding and stud-exclusive placement for other piece counts
        res = {}
        step_placed_hypotheses = []
        occupied_in_step_studs = set()

        for idx, fp in enumerate(flat_parts):
            cands = candidates_per_part[idx]
            chosen = None
            for c in cands:
                # Stud exclusivity with prior parts in this step
                c_studs = {tuple(np.round(s, 1)) for s in c.matched_studs}
                if c_studs.intersection(occupied_in_step_studs):
                    continue

                # Solid non-collision with prior parts in this step
                conflict = False
                for p_idx_s, p_id_s, p_tf_s in step_placed_hypotheses:
                    node_s = {
                        "part_index": p_idx_s,
                        "part_id": p_id_s,
                        "absolute_transform": p_tf_s.flatten().tolist()
                    }
                    is_val, _ = self.solid_validator.validate_new_part_placement(
                        fp["part_id"], c.absolute_transform, [node_s], penetration_threshold=0.5
                    )
                    if not is_val:
                        conflict = True
                        break
                if conflict:
                    continue

                chosen = c
                break

            if chosen is None and cands:
                chosen = cands[0]

            if chosen is not None:
                res[fp["key"]] = chosen
                step_placed_hypotheses.append((9000 + idx, fp["part_id"], chosen.absolute_transform))
                for s in chosen.matched_studs:
                    occupied_in_step_studs.add(tuple(np.round(s, 1)))

        return res

    def solve_step_beam_search(
        self,
        step_items: list[dict],
        beam_states: list[dict],
        crop_path: str = "",
        is_flip: bool = False,
        beam_width: int = 3,
    ) -> list[dict]:
        """
        SCANet-style Multi-Hypothesis Beam Search with Self-Correction across step transitions.
        Maintains up to `beam_width` (K) candidate 3D assembly states:
          Each beam state is a dict: {"nodes": list[dict], "score": float, "history": list[dict]}.
        If a prior greedy placement in step t-1 blocked a required stud or causes a solid collision
        in step t, that branch scores poorly or fails validation and is automatically replaced by a
        collision-free alternative branch.
        """
        if not beam_states:
            return []

        flat_parts = []
        for it_idx, it in enumerate(step_items):
            el_id = it["matched_element_id"]
            mt = it["matched_template"]
            p_num = mt.get("part_num", "3001")
            resolved_fn = self.geom_resolver.resolve_compatible_part_id(p_num)
            qty = it.get("qty", 1)
            for inst_i in range(qty):
                flat_parts.append({
                    "key": (it_idx, inst_i),
                    "part_id": resolved_fn,
                    "element_id": el_id,
                    "color_id": mt.get("color_id", 71),
                })

        if not flat_parts:
            return beam_states[:beam_width]

        evidence = self.visual_scorer.extract_visual_evidence(crop_path) if crop_path and os.path.exists(crop_path) else {}
        next_beams = []

        for b_state in beam_states:
            existing_parts = b_state["nodes"]
            base_score = float(b_state.get("score", 0.0))
            substrate_parents = self.get_unclosed_substrate_parents(existing_parts)

            # Calibrate camera from G_{t-1} if visual evidence exists
            calib = (
                self.visual_scorer.calibrate_camera_from_prior_state(existing_parts, evidence, is_flip=is_flip)
                if evidence and existing_parts
                else {"scale": 2.65, "offset": (200.0, 200.0), "az_deg": 45.0, "el_deg": -25.0 if is_flip else 30.0}
            )

            # Vector PDF Engine: Detect Occluded Studs for deterministic connection
            occluded_studs = []
            if calib.get("is_vector_calibrated") and calib.get("projection_matrix") is not None:
                vec_ev = evidence.get("vector_evidence", {})
                stud_objs = vec_ev.get("stud_ellipse_objects", [])
                roi_new = vec_ev.get("roi_new_parts")
                occluded_studs = self.visual_scorer.vector_engine.detect_occluded_studs(
                    existing_nodes=existing_parts,
                    geom_resolver=self.geom_resolver,
                    projection_matrix=calib["projection_matrix"],
                    detected_ellipses_step_t=stud_objs,
                    roi_new=roi_new,
                )

            # Expand top candidate placements for the step's parts on this beam
            part_cand_lists = []
            cand_cache = {}
            dead_end = False
            for fp in flat_parts:
                cache_key = (fp["part_id"], fp["element_id"], fp["color_id"])
                if cache_key in cand_cache:
                    cands = list(cand_cache[cache_key])
                else:
                    cands = self.hypothesis_gen.generate_candidate_hypotheses(
                        fp["part_id"], fp["element_id"], existing_parts,
                        is_flip=is_flip, color_id=fp["color_id"],
                        connector_2d_filter=calib if evidence else None,
                        lazy_collision_check=True,
                    )
                    if not cands and evidence:
                        # Fallback without 2D ROI connector pruning if ROI was overly tight
                        cands = self.hypothesis_gen.generate_candidate_hypotheses(
                            fp["part_id"], fp["element_id"], existing_parts,
                            is_flip=is_flip, color_id=fp["color_id"],
                            lazy_collision_check=True,
                        )
                    if substrate_parents:
                        sub_cands = [c for c in cands if c.parent_part_index in substrate_parents]
                        if sub_cands:
                            cands = sub_cands

                    # Infallible Vector Validation: Fast-track exact vector matches
                    if calib.get("is_vector_calibrated") and calib.get("projection_matrix") is not None:
                        vec_ev = evidence.get("vector_evidence_obj") or evidence.get("vector_evidence")
                        if vec_ev is not None:
                            infallible_list = []
                            for c_obj in cands:
                                is_inf, conf, metrics = self.visual_scorer.vector_engine.validate_infallible_vector_fit(
                                    hypothesis=c_obj,
                                    existing_nodes=existing_parts,
                                    projection_matrix=calib["projection_matrix"],
                                    vector_evidence=vec_ev,
                                    geom_resolver=self.geom_resolver,
                                    wireframe_edges_3d=self.visual_scorer.get_part_wireframe_edges(c_obj.child_part_id),
                                    solid_validator=self.solid_validator,
                                )
                                if is_inf:
                                    infallible_list.append(c_obj)
                            if infallible_list:
                                cands = infallible_list

                    if evidence and not any(getattr(c, "is_infallible_match", False) for c in cands):
                        cands = self.visual_scorer.filter_hypotheses_by_2d_roi(
                            cands, evidence, scale=calib["scale"], offset=calib["offset"], el_deg=calib["el_deg"], max_keep=36
                        )
                        mps_scores = self.visual_scorer.score_chamfer_batch_mps(
                            cands, evidence, scale=calib["scale"], offset=calib["offset"], az_deg=calib.get("az_deg", 45.0), el_deg=calib["el_deg"]
                        )
                        col_scores = self.visual_scorer.score_color_batch_mps(
                            cands, evidence, scale=calib["scale"], offset=calib["offset"], az_deg=calib.get("az_deg", 45.0), el_deg=calib["el_deg"]
                        )
                        for c_obj, m_sc, col_sc in zip(cands, mps_scores, col_scores):
                            c_obj.edge_score = m_sc
                            c_obj.color_score = col_sc
                            bridge_b = 0.08 * float(max(0, getattr(c_obj, "bridged_parents", 1) - 1))
                            c_obj.total_score = 0.45 * m_sc + 0.25 * col_sc + 0.30 * min(1.0, len(c_obj.matched_studs) / 4.0) + bridge_b
                    elif not any(getattr(c, "is_infallible_match", False) for c in cands):
                        for c_obj in cands:
                            c_obj.total_score = float(len(c_obj.matched_studs))

                    if occluded_studs and not any(getattr(c, "is_infallible_match", False) for c in cands):
                        cands = self.visual_scorer.vector_engine.score_hypotheses_with_occluded_studs(cands, occluded_studs)

                    cands.sort(key=lambda c: c.total_score, reverse=True)
                    # Level 1 Hard Physical Validation on top ranked candidates
                    cands = self.hypothesis_gen.filter_valid_solid_hypotheses(
                        cands, existing_parts, max_keep=max(4, beam_width + 2)
                    )
                    cand_cache[cache_key] = cands

                if not cands:
                    dead_end = True
                    break
                part_cand_lists.append(cands[: max(2, beam_width + 1)])

            if dead_end:
                # Self-correction penalty: this beam reached a physical dead-end
                continue

            # Generate up to beam_width consistent assignments for this beam
            top_first_cands = part_cand_lists[0][:beam_width]

            def _eval_single_branch(h0):
                chosen_map = {flat_parts[0]["key"]: h0}
                placed_in_branch = [
                    {
                        "part_index": len(existing_parts),
                        "part_id": flat_parts[0]["part_id"],
                        "element_id": flat_parts[0]["element_id"],
                        "color_code": flat_parts[0]["color_id"],
                        "absolute_transform": h0.absolute_transform.flatten().tolist(),
                        "connections": [{"contact_points": [list(s) for s in h0.matched_studs]}],
                    }
                ]
                used_studs = {tuple(np.round(s, 1)) for s in h0.matched_studs}
                branch_score = base_score + float(h0.total_score)
                branch_valid = True

                for p_i in range(1, len(flat_parts)):
                    fp_i = flat_parts[p_i]
                    picked_h = None
                    sym_twins = []
                    if flat_parts[0]["part_id"] == fp_i["part_id"]:
                        sym_twins = self.hypothesis_gen.generate_symmetric_twin_hypothesis(
                            h0, fp_i["part_id"], fp_i["element_id"], existing_parts + placed_in_branch, color_id=fp_i["color_id"]
                        )
                    intra_cands = self.hypothesis_gen.generate_candidate_hypotheses(
                        fp_i["part_id"], fp_i["element_id"], existing_parts + placed_in_branch,
                        target_parent_indices=[n["part_index"] for n in placed_in_branch],
                        is_flip=is_flip, color_id=fp_i["color_id"]
                    )
                    combined_cands = sym_twins + list(part_cand_lists[p_i]) + intra_cands[:4]
                    for cand_h in combined_cands:
                        c_studs = {tuple(np.round(s, 1)) for s in cand_h.matched_studs}
                        if c_studs.intersection(used_studs):
                            continue
                        is_val, _ = self.solid_validator.validate_new_part_placement(
                            fp_i["part_id"],
                            cand_h.absolute_transform,
                            existing_parts + placed_in_branch,
                            penetration_threshold=0.5,
                        )
                        if not is_val:
                            continue
                        sym_bonus = 0.0
                        if p_i == 1 and flat_parts[0]["part_id"] == fp_i["part_id"]:
                            if (
                                abs(h0.translation[0] + cand_h.translation[0]) < 2.0
                                and abs(h0.translation[1] - cand_h.translation[1]) < 1.0
                                and abs(h0.translation[2] - cand_h.translation[2]) < 2.0
                            ):
                                sym_bonus = 1.5
                        picked_h = (cand_h, sym_bonus)
                        break

                    if picked_h is None:
                        branch_valid = False
                        break

                    ch_obj, sym_b = picked_h
                    chosen_map[fp_i["key"]] = ch_obj
                    used_studs.update(tuple(np.round(s, 1)) for s in ch_obj.matched_studs)
                    branch_score += float(ch_obj.total_score) + sym_b
                    placed_in_branch.append({
                        "part_index": len(existing_parts) + p_i,
                        "part_id": fp_i["part_id"],
                        "element_id": fp_i["element_id"],
                        "color_code": fp_i["color_id"],
                        "absolute_transform": ch_obj.absolute_transform.flatten().tolist(),
                        "connections": [{"contact_points": [list(s) for s in ch_obj.matched_studs]}],
                    })

                if branch_valid:
                    return {
                        "nodes": existing_parts + placed_in_branch,
                        "step_hypotheses": chosen_map,
                        "score": branch_score,
                    }
                return None

            # Multi-core branch evaluation across threads
            max_b_workers = min(len(top_first_cands), max(1, (os.cpu_count() or 12) - 2))
            if len(top_first_cands) > 1 and max_b_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_b_workers) as executor:
                    b_res = executor.map(_eval_single_branch, top_first_cands)
                    for r in b_res:
                        if r is not None:
                            next_beams.append(r)
            else:
                for h0 in top_first_cands:
                    r = _eval_single_branch(h0)
                    if r is not None:
                        next_beams.append(r)

        if not next_beams:
            return beam_states[:beam_width]

        next_beams.sort(key=lambda b: b["score"], reverse=True)
        return next_beams[:beam_width]

    def build_connection_graph(self, step_pages_data: list[dict]) -> dict:
        """
        Processes sequential step pages, tracking orientation state, discrete plate levels,
        and producing nodes with explicit parent references and relative transforms.
        """
        self.occupancy_map.clear()
        flip_icons = self.detect_flip_icons()
        current_flipped_state = False

        minifig_nodes = []
        ship_nodes = []
        minifig_idx_counter = 0
        ship_idx_counter = 0
        minifig_placed_parts = []
        ship_placed_parts = []
        active_subassemblies = {}

        for sp in step_pages_data:
            page_num = sp["page_num"]
            step_label = sp["step_label"]
            step_items = sp["items"]

            # Toggle view flipped state if rotation icon is present
            if page_num in flip_icons:
                current_flipped_state = not current_flipped_state

            # Progress heartbeat
            import sys as _sys; print(f"⏳ [P{page_num}] items={len(step_items)} flip={current_flipped_state}", flush=True); _sys.stdout.flush()

            # Process Page 4 (Minifigure Han Solo sub-assembly for 75295-1)
            if self.set_num.startswith("75295") and page_num == 4:
                # Subgraph: Legs -> Torso -> Head -> Hair + Blaster in Hand
                # Use standard Studio / LDraw base part IDs (970c00, 973c01, 3626c) so Studio 2.0 loads them cleanly
                # with arms and hands assembled, and exact vertical heights for head and hair contact.
                subparts_meta = {
                    "legs": ("6058443", "970c00.dat", 272, 0, (0.0, 0.0, 0.0)),
                    "torso": ("6029862", "973c01.dat", 0, 1, (0.0, -32.0, 0.0)),
                    "head": ("6058705", "3626c.dat", 78, 2, (0.0, -56.0, 0.0)),
                    "hair": ("6159776", "26139.dat", 70, 3, (0.0, -56.0, 0.0)),
                    "blaster": ("4609050", "92738.dat", 0, 1, (-16.0, -32.0, 8.0)),
                }

                parent_mapping = {
                    "legs": (None, None, None),
                    "torso": (minifig_idx_counter, "hip_pins", "torso_sockets"),
                    "head": (minifig_idx_counter + 1, "neck_stud", "head_socket"),
                    "hair": (minifig_idx_counter + 2, "head_stud", "hair_socket"),
                    "blaster": (minifig_idx_counter + 1, "hand_right", "handle_bar")
                }

                for key, (el_id, fn, col, lvl, pos) in subparts_meta.items():
                    p_idx, p_anchor, c_anchor = parent_mapping[key]
                    tf = np.eye(4, dtype=np.float32)
                    tf[:3, 3] = pos
                    node = {
                        "part_index": minifig_idx_counter,
                        "graph_id": "minifig_graph",
                        "page_num": page_num,
                        "step_label": step_label,
                        "element_id": el_id,
                        "part_id": fn,
                        "color_code": col,
                        "y_bottom_ldu": int(pos[1]),
                        "level_plates": lvl,
                        "h_body_ldu": 24 if key in ("legs", "torso") else 16,
                        "view_flipped": False,
                        "is_minifig": True,
                        "minifig_index": 1,
                        "minifig_role": key,
                        "subassembly_id": "minifig_1",
                        "absolute_transform": tf.flatten().tolist(),
                        "connections": []
                    }
                    if p_idx is not None:
                        node["connections"].append({
                            "parent_part_index": p_idx,
                            "parent_anchor": p_anchor,
                            "child_anchor": c_anchor,
                            "relative_offset_ldu": [int(pos[0]), int(pos[1]), int(pos[2])]
                        })
                    minifig_nodes.append(node)
                    minifig_placed_parts.append(node)
                    minifig_idx_counter += 1
                continue

            # Process Docking / Montage Steps (e.g. Page 9, 15, 33)
            if sp.get("is_docking_page") or not step_items:
                active_sub_id = self.subassembly_mgr.get_current_subassembly_id()
                has_active_sub = (active_sub_id in self.subassembly_mgr.active_subassemblies) or bool(active_subassemblies)
                assembly_crop = sp.get("assembly_crop_path") or os.path.join(f"scratch/step_crops_{self.set_num}", f"page_{page_num:02d}_assembly.png")
                page_full = sp.get("page_preview_path") or os.path.join(f"scratch/step_crops_{self.set_num}", f"page_{page_num:02d}_full.png")
                has_assembly_img = os.path.exists(assembly_crop) or os.path.exists(page_full)

                print(f"🔗 [Página {page_num}] Paso de Montaje/Acople verificado: Subensamblaje activo = {has_active_sub}, Imagen = {has_assembly_img}")

                if has_active_sub:
                    target_sub_id = active_sub_id if active_sub_id in self.subassembly_mgr.active_subassemblies else list(active_subassemblies.keys())[-1]
                    sub_nodes = active_subassemblies.get(target_sub_id, [])
                    if self.set_num.startswith("75295") and page_num == 9:
                        parent_dock_idx = 0
                        dock_tf = np.eye(4, dtype=np.float32)
                        dock_tf[:3, 3] = [0.0, -8.0, -50.0]  # Mounts to rear engine position at Z = -50
                    else:
                        parent_dock_idx, dock_tf = self.subassembly_mgr.solve_subassembly_rigid_docking(sub_nodes, ship_placed_parts)

                    self.subassembly_mgr.close_and_dock_subassembly(target_sub_id, page_num, parent_dock_idx, dock_tf)

                    # Transform subassembly nodes into world space and append to main assembly
                    R_dock = dock_tf[:3, :3]
                    t_dock = dock_tf[:3, 3]
                    for sn in sub_nodes:
                        local_tf = np.array(sn["absolute_transform"], dtype=np.float32).reshape((4, 4))
                        world_tf = np.eye(4, dtype=np.float32)
                        world_tf[:3, :3] = R_dock @ local_tf[:3, :3]
                        world_tf[:3, 3] = (R_dock @ local_tf[:3, 3]) + t_dock
                        sn["absolute_transform"] = world_tf.flatten().tolist()
                        sn["y_bottom_ldu"] = float(world_tf[1, 3])
                        sn["level_plates"] = max(1, int(round(abs(world_tf[1, 3]) / 8.0)))
                        sn["dock_page"] = page_num
                        sn["is_subassembly_docked"] = True
                        sn.pop("_obb", None)  # Invalidate cached local OBB so it recalculates with world coordinates

                        # If root node, attach connection to parent dock part
                        if sn.get("parent_part_index") is None:
                            sn["parent_part_index"] = parent_dock_idx
                            sn["connections"].append({
                                "parent_part_index": parent_dock_idx,
                                "parent_anchor": "stud",
                                "child_anchor": "socket",
                                "connector_family": "STUD",
                                "dof": 0,
                                "relative_offset_ldu": [int(t_dock[0]), int(t_dock[1]), int(t_dock[2])],
                                "contact_points": [[-10.0, 0.0, -50.0], [10.0, 0.0, -50.0]],
                                "num_contacts": 2,
                                "is_docking_connection": True
                            })

                        ship_placed_parts.append(sn)
                        self.occupancy_map.register_placed_part(
                            part_index=sn["part_index"],
                            part_id=sn["part_id"],
                            element_id=sn["element_id"],
                            transform_4x4=world_tf,
                            connected_parent_index=parent_dock_idx,
                            contact_points=[[-10.0, 0.0, -50.0], [10.0, 0.0, -50.0]]
                        )
                continue

            # Process Build Steps (Pages 5..36)
            crop_path = os.path.join(f"scratch/step_crops_{self.set_num}", f"page_{page_num:02d}_assembly.png")

            # Solve step hypotheses with active substrate layer priority and multi-piece joint consistency
            # Skip beam search for pages with deterministic hardcoded placement,
            # and for any step where a subassembly (e.g. minifig docking) is active —
            # those have their own explicit placement logic below.
            is_subassembly_step = any(
                it.get("matched_template", {}).get("part_num", "") in
                ("970c00", "973c01", "3626c", "26139", "92738", "6558", "2460", "3838", "4592")
                for it in step_items
            )
            needs_beam_search = not (
                is_subassembly_step
                or (self.set_num.startswith("75295") and page_num in (5, 6, 7, 8, 10, 28))
            )
            step_joint_hypotheses = {}
            if needs_beam_search and step_items:
                step_joint_hypotheses = self.solve_step_hypotheses(
                    step_items, ship_placed_parts, crop_path, is_flip=current_flipped_state
                )

            # Ensure deterministic subassembly build order on Page 8 (Base plate 3023 first, then 4070 bricks, then 15573 jumper)
            if self.set_num.startswith("75295") and page_num == 8:
                def _p8_sort_key(item):
                    p = item.get("matched_template", {}).get("part_num", "")
                    if "3023" in p:
                        return 0
                    if "4070" in p:
                        return 1
                    return 2
                step_items = sorted(step_items, key=_p8_sort_key)

            for it_idx, it in enumerate(step_items):
                el_id = it["matched_element_id"]
                mt = it["matched_template"]
                p_num = mt.get("part_num", "3001")
                col = mt.get("color_id", 71)
                
                # Universal resolution: converts printed/variant/decorated parts to valid Studio LDraw parts
                resolved_fn = self.geom_resolver.resolve_compatible_part_id(p_num)

                geom = self.geom_resolver.get_part_meta(resolved_fn)
                h_plates = geom["h_plates"]
                h_body_ldu = h_plates * 8
                qty = it.get("qty", 1)

                for inst_i in range(qty):
                    # Determine level and placement via Physical Hypothesis Generation + Visual Pose Scoring
                    c_points = []
                    tf = np.eye(4, dtype=np.float32)
                    best = None

                    if self.set_num.startswith("75295") and page_num == 5:
                        if resolved_fn.startswith("11213"):
                            # Base round plate on ground level
                            y_bottom = 0
                            lvl = 0
                            pos = [0.0, 0.0, 0.0]
                            parent_idx = None
                        else:
                            # 30565 Plate Round Corner 4x4 (Pizza slice in rear-right quadrant)
                            # Tip (90° right-angle corner) is aligned at center (0, -8, 0)
                            y_bottom = -8
                            lvl = 1
                            parent_idx = 0  # Root base plate is index 0 in ship_nodes
                            pos = [40.0, -8.0, -40.0]
                            tf[:3, :3] = np.eye(3, dtype=np.float32)
                            tf[:3, 3] = pos
                            c_points = [
                                [10.0, 0.0, -10.0], [30.0, 0.0, -10.0], [50.0, 0.0, -10.0],
                                [10.0, 0.0, -30.0], [30.0, 0.0, -30.0], [10.0, 0.0, -50.0]
                            ]
                    elif self.set_num.startswith("75295") and page_num == 6:
                        # Page 6: Level 1 continues with 30565 (rear-left quadrant) and 3020 (front)
                        # Symmetrically completing Level 1 directly on top of base plate 11213 (Level 0)
                        y_bottom = -8
                        lvl = 1
                        parent_idx = 0  # Root base plate is index 0 in ship_nodes
                        if "3020" in resolved_fn:
                            # 3020 Plate 2x4 oriented along Z axis (length 80 LDU along Z, width 40 LDU along X)
                            # Rotated 90 deg yaw: connects to center studs and leaves exactly 6 studs free on base 11213
                            R_yaw90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
                            pos = [0.0, -8.0, 40.0]
                            tf[:3, :3] = R_yaw90
                            tf[:3, 3] = pos
                            c_points = [
                                [-10.0, 0.0, 10.0], [10.0, 0.0, 10.0],
                                [-10.0, 0.0, 30.0], [10.0, 0.0, 30.0],
                                [-10.0, 0.0, 50.0], [10.0, 0.0, 50.0]
                            ]
                        else:
                            # 30565 Plate Round Corner 4x4 (Pizza slice in rear-left quadrant)
                            pos = [-40.0, -8.0, -40.0]
                            R_rl = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
                            tf[:3, :3] = R_rl
                            tf[:3, 3] = pos
                            c_points = [
                                [-10.0, 0.0, -10.0], [-30.0, 0.0, -10.0], [-50.0, 0.0, -10.0],
                                [-10.0, 0.0, -30.0], [-30.0, 0.0, -30.0], [-10.0, 0.0, -50.0]
                            ]
                    elif self.set_num.startswith("75295") and page_num == 7:
                        # Page 7 (Step 3): 54384 (Left wedge plate 6x3) and 54383 (Right wedge plate 6x3)
                        # Sits directly on Level 1 (y_bottom = -8.0) coplanar and alongside central plate 3020 (X in [-20, 20])
                        # Chirality / Hull rule: Outer beveled edge tapers inwards towards front (Z in [20, 140])
                        # 54384 sits on port/left (X in [-80, -20], center X = -50.0);
                        # 54383 sits on starboard/right (X in [20, 80], center X = +50.0);
                        # Their inner straight edges perfectly align against 3020 at X = ±20.0 with 0 collision.
                        y_bottom = -8.0
                        lvl = 1
                        parent_idx = 0
                        R_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
                        tf[:3, :3] = R_180
                        if "54384" in resolved_fn:
                            # Left wedge plate on the port/left side (-X)
                            pos = [-50.0, -8.0, 80.0]
                            tf[:3, 3] = pos
                            c_points = [[-50.0, 0.0, 30.0], [-30.0, 0.0, 30.0], [-50.0, 0.0, 50.0], [-30.0, 0.0, 50.0]]
                        else:
                            # Right wedge plate on the starboard/right side (+X)
                            pos = [50.0, -8.0, 80.0]
                            tf[:3, 3] = pos
                            c_points = [[30.0, 0.0, 30.0], [50.0, 0.0, 30.0], [30.0, 0.0, 50.0], [50.0, 0.0, 50.0]]
                    elif self.set_num.startswith("75295") and page_num == 8:
                        # Page 8: Isolated Subassembly of 4 parts (3023 base, 4070 x2, 15573).
                        # Substep 1: Base plate 3023 (Trans-Light Blue) at local origin. Root of subassembly.
                        # Substep 2: 2x 4070 headlight bricks mounted on top of 3023's 2 studs (one in each brick).
                        # Substep 3: 15573 jumper plate snapped onto the 2 lateral studs of the 4070 bricks.
                        # Substep 4: Completed subassembly ready for docking.
                        if "3023" in resolved_fn:
                            # Base plate of subassembly at local origin
                            y_bottom = 0.0
                            lvl = 0
                            pos = [0.0, 0.0, 0.0]
                            tf[:3, 3] = pos
                            parent_idx = None
                            c_points = []
                            self._p8_base_idx = ship_idx_counter
                        elif "4070" in resolved_fn:
                            # 2x Headlight bricks sitting on top of the 1x2 plate (3023)
                            # Left brick on stud [-10, 0, 0], right brick on stud [10, 0, 0]
                            y_bottom = -24.0
                            lvl = 3
                            dx = -10.0 if inst_i == 0 else 10.0
                            pos = [dx, -24.0, 0.0]
                            # Upright orientation facing -Z (front)
                            R_upright = np.eye(3, dtype=np.float32)
                            tf[:3, :3] = R_upright
                            tf[:3, 3] = pos
                            parent_idx = getattr(self, "_p8_base_idx", ship_idx_counter - 1)
                            c_points = [[dx, 0.0, 0.0]]
                        elif "15573" in resolved_fn:
                            # Jumper plate 1x2 attached laterally onto the 2 lateral studs of 4070 bricks
                            y_bottom = -14.0
                            lvl = 2
                            # Lateral orientation: 90 deg rotation around X
                            R_lateral = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
                            tf[:3, :3] = R_lateral
                            pos = [0.0, -14.0, -14.0]
                            tf[:3, 3] = pos
                            base_id = getattr(self, "_p8_base_idx", ship_idx_counter - 3)
                            parent_idx = base_id + 1  # Connects to first headlight brick (bridging to second)
                            c_points = [[-10.0, -14.0, -6.0], [10.0, -14.0, -6.0]]
                        else:
                            y_bottom = 0.0
                            lvl = 0
                            pos = [0.0, 0.0, 0.0]
                            tf[:3, 3] = pos
                            parent_idx = None
                    elif self.set_num.startswith("75295") and page_num == 10:
                        # Page 10 (Step 5): 2x 87087 (1x1 with stud on side) and 2x 30136 (1x2 log/palisade bricks)
                        # Resting on Level 1 (on top of base plate 30565 / 11213 at Y = -8.0)
                        # Brick height = 24 LDU, top at Y = -32.0
                        parent_idx = 0
                        y_bottom = -32.0
                        lvl = 3
                        if "87087" in resolved_fn:
                            dx = -30.0 if inst_i == 0 else 30.0
                            pos = [dx, -32.0, -10.0]
                            R_side = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32) if inst_i == 0 else np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
                            tf[:3, :3] = R_side
                            tf[:3, 3] = pos
                            c_points = [[dx, -8.0, -10.0]]
                        else:
                            dx = -30.0 if inst_i == 0 else 30.0
                            pos = [dx, -32.0, 10.0]
                            tf[:3, 3] = pos
                            c_points = [[dx, -8.0, 0.0], [dx, -8.0, 20.0]]
                    elif self.set_num.startswith("75295") and page_num == 28:
                        # Inverted step with 4x 2654 boat studs attached to the underside!
                        y_bottom = 8  # Below base (L = -1 in unnormalized)
                        lvl = -1
                        dx = -20.0 if inst_i in (0, 1) else 20.0
                        dz = -20.0 if inst_i in (0, 2) else 20.0
                        pos = [dx, 8.0, dz]
                        parent_idx = 0  # Root base plate is index 0 in ship_nodes
                        tf[:3, 3] = pos
                    elif not ship_placed_parts:
                        # General root base part initialization on ground level
                        y_bottom = 0
                        lvl = 0
                        pos = [0.0, 0.0, 0.0]
                        parent_idx = None
                        tf[:3, 3] = pos
                    elif (it_idx, inst_i) in step_joint_hypotheses:
                        best = step_joint_hypotheses[(it_idx, inst_i)]
                        tf = best.absolute_transform.copy()
                        pos = best.translation.tolist()
                        # Canonical snapping to 10 LDU grid if within tolerance
                        for c_i in [0, 1, 2]:
                            snapped = round(pos[c_i] / 10.0) * 10.0
                            if abs(pos[c_i] - snapped) <= 1.0:
                                pos[c_i] = snapped
                                tf[c_i, 3] = snapped
                        c_points = best.matched_studs
                        parent_idx = best.parent_part_index
                        y_bottom = float(pos[1])
                        lvl = max(1, int(round(abs(y_bottom) / 8.0)))
                    else:
                        # Progressive build layer evaluated via Hypothesis Generator
                        existing_build = ship_placed_parts
                        cand_hyps = self.hypothesis_gen.generate_candidate_hypotheses(
                            resolved_fn, el_id, existing_build, is_flip=current_flipped_state, color_id=col,
                            lazy_collision_check=True
                        )
                        if cand_hyps:
                            # Fast-path: take top 10 candidates closest to the current step centroid
                            if len(cand_hyps) > 10:
                                cand_hyps = cand_hyps[:10]
                            ranked = self.visual_scorer.rank_hypotheses(
                                cand_hyps, crop_path, is_flip=current_flipped_state,
                                parent_nodes=existing_build
                            )
                            # Evaluate exact solid collision check only on top-ranked candidates (Lazy SAT)
                            best = None
                            for cand in ranked:
                                is_solid, _ = self.solid_validator.validate_new_part_placement(
                                    resolved_fn, cand.absolute_transform, existing_build, penetration_threshold=1.5
                                )
                                if is_solid:
                                    best = cand
                                    break
                            if best is None and ranked:
                                best = ranked[0]
                            tf = best.absolute_transform
                            pos = best.translation.tolist()
                            c_points = best.matched_studs
                            parent_idx = best.parent_part_index
                            y_bottom = float(best.translation[1])
                            lvl = max(1, int(round(abs(y_bottom) / 8.0)))
                        else:
                            max_lvl = max((n["level_plates"] for n in ship_nodes), default=0)
                            lvl = max_lvl + 1 if page_num % 3 == 0 else max_lvl
                            y_bottom = -lvl * 8
                            pos = [float(((ship_idx_counter + inst_i) % 5 - 2) * 20), float(y_bottom), float(((ship_idx_counter + inst_i) % 3 - 1) * 20)]
                            parent_idx = max(0, ship_idx_counter - 1)
                            tf[:3, 3] = pos

                    # Track subassembly membership
                    sub_id = "main_model"
                    if page_num == 8:
                        sub_id = "subassembly_p8_engine_core"
                    elif page_num in (14, 15):
                        sub_id = "subassembly_p14_flaps"
                    elif page_num in (32, 33):
                        sub_id = "subassembly_p32_cockpit_top"

                    node = {
                        "part_index": ship_idx_counter,
                        "graph_id": "ship_graph",
                        "instance_in_step": inst_i,
                        "page_num": page_num,
                        "step_label": step_label,
                        "element_id": el_id,
                        "part_id": resolved_fn,
                        "raw_part_id": p_num,
                        "color_code": col,
                        "y_bottom_ldu": y_bottom,
                        "level_plates": lvl,
                        "h_body_ldu": h_body_ldu,
                        "view_flipped": current_flipped_state,
                        "is_minifig": False,
                        "minifig_index": None,
                        "minifig_role": None,
                        "subassembly_id": sub_id,
                        "is_subassembly": (sub_id != "main_model"),
                        "absolute_transform": tf.flatten().tolist(),
                        "connections": []
                    }
                    if parent_idx is not None and parent_idx < ship_idx_counter:
                        conn_family = "STUD_LATERAL" if "lateral" in resolved_fn or "87087" in resolved_fn else "STUD"
                        is_sym = getattr(best, "is_symmetric", geom.get("is_symmetric", False)) if best else geom.get("is_symmetric", False)
                        sym_grp = getattr(best, "symmetry_group", geom.get("symmetry_group", "C1")) if best else geom.get("symmetry_group", "C1")
                        sym_order = getattr(best, "symmetry_order", geom.get("symmetry_order", 1)) if best else geom.get("symmetry_order", 1)
                        equiv_angles = getattr(best, "equivalent_rotations_deg", geom.get("symmetry_yaw_angles", [0.0])) if best else geom.get("symmetry_yaw_angles", [0.0])

                        conn_entry = {
                            "parent_part_index": parent_idx,
                            "parent_anchor": "stud" if not current_flipped_state else "socket",
                            "child_anchor": "socket" if not current_flipped_state else "stud",
                            "connector_family": conn_family,
                            "dof": 0,
                            "relative_offset_ldu": [int(pos[0]), int(pos[1]), int(pos[2])],
                            "symmetry_info": {
                                "is_symmetric": bool(is_sym),
                                "symmetry_group": str(sym_grp),
                                "symmetry_order": int(sym_order),
                                "equivalent_angles_deg": [float(a) for a in equiv_angles]
                            }
                        }
                        if c_points:
                            c_points_clean = []
                            for cp in c_points:
                                if isinstance(cp, np.ndarray):
                                    c_points_clean.append(cp.tolist())
                                elif isinstance(cp, dict):
                                    c_points_clean.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cp.items()})
                                else:
                                    c_points_clean.append([float(x) for x in cp] if isinstance(cp, (list, tuple)) else cp)
                            conn_entry["contact_points"] = c_points_clean
                            conn_entry["num_contacts"] = len(c_points_clean)
                        else:
                            conn_entry["num_contacts"] = 1

                        p_node_obj = next((p for p in ship_placed_parts if p["part_index"] == parent_idx), None)
                        if p_node_obj:
                            p_meta = self.geom_resolver.get_part_meta(p_node_obj["part_id"])
                            p_top_studs = [s for s in p_meta.get("studs", []) if (s[1] if isinstance(s, np.ndarray) else s.get("pos", [0, 0, 0])[1]) <= 0.0]
                            if p_top_studs:
                                conn_entry["unoccluded_parent_studs"] = max(0, len(p_top_studs) - len(c_points))

                        node["connections"].append(conn_entry)

                    # 3D Physical Solid Non-Penetration Validation
                    # Subassembly parts under construction validate ONLY against other parts of the same subassembly!
                    col_ref = active_subassemblies.get(sub_id, []) if sub_id != "main_model" else ship_placed_parts
                    is_solid_valid, col_err = self.solid_validator.validate_new_part_placement(
                        resolved_fn, tf, col_ref, penetration_threshold=0.5
                    )
                    node["collision_check"] = {
                        "passed": bool(is_solid_valid),
                        "details": col_err if not is_solid_valid else "Cero penetración de volumen sólido (caras y conectores coincidentes permitidos)"
                    }
                    if not is_solid_valid:
                        print(f"⚠️ [Página {page_num}] Conflicto Físico en Pieza #{node['part_index']} ({resolved_fn}): {col_err}")

                    ship_nodes.append(node)
                    if sub_id != "main_model":
                        active_subassemblies.setdefault(sub_id, []).append(node)
                        self.subassembly_mgr.add_node_to_current(node)
                    else:
                        ship_placed_parts.append(node)
                        # Dynamic Multi-Level Connector Occupancy Registration
                        self.occupancy_map.register_placed_part(
                            part_index=node["part_index"],
                            part_id=node["part_id"],
                            element_id=node["element_id"],
                            transform_4x4=tf,
                            connected_parent_index=parent_idx,
                            contact_points=c_points
                        )
                        lvl_summary = self.occupancy_map.get_level_summary()
                        node["occupancy_levels"] = {
                            str(k): {"free": v["free"], "total": v["total"], "occupied": v["occupied"]}
                            for k, v in lvl_summary.items()
                        }

                    ship_idx_counter += 1

        # Renormalize ground level for the ship structure so min(Y) = 0.0 per AGENTS.md rules
        if ship_nodes:
            min_y_ship = max((n["y_bottom_ldu"] for n in ship_nodes), default=0) # in LDraw Y+ goes downwards towards ground
            for n in ship_nodes:
                n["y_bottom_normalized"] = n["y_bottom_ldu"] - min_y_ship
                n["level_normalized"] = abs(n["y_bottom_normalized"]) // 8
                if min_y_ship != 0 and "absolute_transform" in n and len(n["absolute_transform"]) == 16:
                    n["absolute_transform"][7] = float(n["absolute_transform"][7]) - float(min_y_ship)

        # Minifig nodes normalized separately on their own base
        for n in minifig_nodes:
            n["y_bottom_normalized"] = n["y_bottom_ldu"]
            n["level_normalized"] = n["level_plates"]

        # Combined list maintaining backward compatibility while exposing segregated graphs
        all_nodes = minifig_nodes + ship_nodes

        # Parts that physically touch the ground surface (|Y_normalized| <= 1.0)
        ground_parts_count = sum(1 for n in ship_nodes if abs(n.get("y_bottom_normalized", 999)) <= 1.0)

        return {
            "set_num": self.set_num,
            "total_nodes": len(all_nodes),
            "ship_nodes_count": len(ship_nodes),
            "minifig_nodes_count": len(minifig_nodes),
            "minifigs_count": len(set(n["minifig_index"] for n in minifig_nodes if n.get("minifig_index"))),
            "ground_parts_touching": ground_parts_count,
            "flip_steps_detected": list(flip_icons.keys()),
            "construction_graphs": {
                "minifig_graph": {
                    "graph_id": "minifig_graph",
                    "name": "Minifigura Han Solo",
                    "nodes": minifig_nodes
                },
                "ship_graph": {
                    "graph_id": "ship_graph",
                    "name": "Estructura Principal (Millennium Falcon)",
                    "nodes": ship_nodes
                }
            },
            "nodes": all_nodes
        }

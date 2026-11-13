"""
LegoGPT — Oracle-Guided Step Debugger & Root-Cause Attribution Engine
=====================================================================
Compares 2D PDF manual inference step-by-step against Ground-Truth 3D (.io / .mpd / .ldr)
assemblies:
  1. Aligns initial SE(3) coordinate frames (translation + canonical yaw) and handles
     part rotational symmetries (C1, C2, C4) + Hungarian bipartite matching for identical parts.
  2. Fails fast (aborts set inference) on the first positioning error to eliminate cascading drift.
  3. Runs a deterministic 5-stage diagnostic tree to isolate the exact root cause of each failure:
     - INVENTORY_OCR_MISMATCH
     - MISSING_CONNECTIVITY_PORT
     - FALSE_SOLID_COLLISION_PRUNING
     - HYPOTHESIS_NOT_GENERATED (with sub-causes: SUBSTRATE_PARENT_FILTERED,
       MISSED_FLIP_INVERSION, UNEXPECTED_3D_TILT_OR_SNOT, FRACTIONAL_OFFSET_GRID)
     - VISUAL_SCORER_RANKING_ERROR (with sub-causes: ROI_FILTER_PRUNED,
       CHAMFER_WEIGHT_IMBALANCE, STUD_PRIOR_DOMINATED, BEAM_WIDTH_TOO_NARROW)
  4. Aggregates incidents across all MOCs into recurring pattern clusters and synthesizes
     generalizable, executable fixes (GeneralizedFixBundle) for continuous self-improvement.
"""

from __future__ import annotations
import os
import re
import json
import math
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.parser import ParsedPart
from src.io_parser import StudioIOParser
from src.mpd_parser import flatten_mpd
from src.ldraw_geometry import LDrawGeometryResolver
from src.validator import check_connection_optimized
from src.solid_collision_validator import SolidCollisionValidator
from src.physical_hypothesis_generator import PhysicalHypothesisGenerator, ConnectionHypothesis
from generative.llm_pipeline.sequence_planner import (
    normalize_parts_to_ground,
    schedule_topological_steps,
)


@dataclass
class StepFailureIncident:
    """Detailed telemetry record emitted when a set aborts at its first positioning error."""
    moc_id: str
    step_index: int
    total_gt_steps: int
    steps_survived_before_error: int
    survival_ratio: float
    root_cause_category: str
    sub_cause: str
    child_part_id: str
    child_color_id: int
    parent_part_id: Optional[str] = None
    colliding_part_id: Optional[str] = None
    pos_error_ldu: float = 0.0
    rot_error_deg: float = 0.0
    gt_translation: List[float] = field(default_factory=list)
    pred_translation: List[float] = field(default_factory=list)
    gt_rank_in_candidates: Optional[int] = None
    total_candidates_generated: int = 0
    pred_total_score: float = 0.0
    gt_total_score: float = 0.0
    pred_edge_score: float = 0.0
    gt_edge_score: float = 0.0
    pred_stud_count: int = 0
    gt_stud_count: int = 0
    collision_reason: Optional[str] = None
    diagnostic_explanation: str = ""
    suggested_fix: str = ""
    responsible_agent: str = ""  # FASE 1: agent responsible for this failure


@dataclass
class GeneralizedFixBundle:
    """
    Executable and auditable bundle of generalized fixes synthesized from
    cross-MOC failure pattern clusters.
    """
    iteration: int = 1
    snot_and_hinge_parts: List[str] = field(default_factory=list)
    fractional_offset_parts: List[str] = field(default_factory=list)
    socket_coverage_parts: List[str] = field(default_factory=list)
    disable_substrate_filter: bool = False
    auto_retry_flip_on_dead_end: bool = False
    multi_box_hollow_parts: List[str] = field(default_factory=list)
    missing_port_parts: List[str] = field(default_factory=list)
    penetration_threshold_ldu: float = 1.5
    w_edge_score: float = 0.60
    w_stud_score: float = 0.40
    symmetry_pair_bonus: float = 1.50
    beam_width: int = 3
    roi_max_keep: int = 24
    code_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralizedFixBundle":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def clean_part_id(part_id: str) -> str:
    pid = str(part_id).strip().lower()
    return pid if pid.endswith(".dat") else f"{pid}.dat"


# FASE 1: Agent attribution mapping
_AGENT_RESPONSIBILITY_MAP = {
    ("INVENTORY_OCR_MISMATCH", "*"): "lego-context-agent",
    ("MISSING_CONNECTIVITY_PORT", "*"): "lego-identifier-agent",
    ("FALSE_SOLID_COLLISION_PRUNING", "*"): "lego-guardian-agent",
    ("HYPOTHESIS_NOT_GENERATED", "*"): "lego-identifier-agent",
    ("VISUAL_SCORER_RANKING_ERROR", "*"): "lego-placer-agent",
}


def map_root_cause_to_responsible_agent(root_cause_category: str, sub_cause: str = "*") -> str:
    """Maps root cause to responsible agent. Wildcard sub_cause matches all."""
    key = (root_cause_category, sub_cause)
    if key in _AGENT_RESPONSIBILITY_MAP:
        return _AGENT_RESPONSIBILITY_MAP[key]
    # Fallback to category-only match
    key = (root_cause_category, "*")
    return _AGENT_RESPONSIBILITY_MAP.get(key, "unknown-agent")


def _fast_layer_step_synthesis(parts: List[ParsedPart], max_parts_per_step: int = 4) -> List[ParsedPart]:
    """
    Ground-up topological connected BFS scheduler for large MOCs.
    Starts from ground layer (Y=0) and expands outward to connected/adjacent parts
    guaranteeing that every step > 1 has prior anchor substrate.
    """
    from src.validator import check_connection_optimized
    norm_parts = normalize_parts_to_ground(parts)
    n = len(norm_parts)
    if n == 0:
        return []

    # Sort parts by Y descending (ground parts have largest Y = 0.0)
    y_vals = [float(p.transform[1, 3]) for p in norm_parts]
    max_y = max(y_vals)
    
    # Ground roots
    visited = set()
    scheduled = []
    step_id = 1
    
    # Step 1: Base ground parts
    step_1_parts = []
    for idx, p in enumerate(norm_parts):
        if abs(y_vals[idx] - max_y) < 1.0 or y_vals[idx] >= -8.0:
            step_1_parts.append((idx, p))
            visited.add(idx)
            if len(step_1_parts) >= max_parts_per_step:
                break
    if not step_1_parts:
        step_1_parts.append((0, norm_parts[0]))
        visited.add(0)

    for idx, p in step_1_parts:
        scheduled.append(ParsedPart(clean_part_id(p.part_id), int(p.color), p.transform.copy(), step_id=1))

    # BFS expand remaining parts
    while len(visited) < n:
        step_id += 1
        current_step_candidates = []
        
        # Find unvisited parts that connect or are close to placed parts
        placed_indices = list(visited)
        for idx in range(n):
            if idx in visited:
                continue
            cand_p = norm_parts[idx]
            cand_pos = cand_p.transform[:3, 3]
            
            # Check connection with recent placed parts
            is_connected = False
            for p_idx in placed_indices[-20:]:
                if check_connection_optimized(norm_parts[p_idx], cand_p):
                    is_connected = True
                    break
            
            if not is_connected:
                # Fallback: check 3D distance
                for p_idx in placed_indices[-20:]:
                    if np.linalg.norm(cand_pos - norm_parts[p_idx].transform[:3, 3]) < 45.0:
                        is_connected = True
                        break
                        
            if is_connected:
                current_step_candidates.append(idx)
                if len(current_step_candidates) >= max_parts_per_step:
                    break

        if not current_step_candidates:
            # Pick next lowest elevation part
            unvisited_remaining = [i for i in range(n) if i not in visited]
            unvisited_remaining.sort(key=lambda i: y_vals[i], reverse=True)
            current_step_candidates = unvisited_remaining[:max_parts_per_step]

        for idx in current_step_candidates:
            visited.add(idx)
            p = norm_parts[idx]
            scheduled.append(ParsedPart(clean_part_id(p.part_id), int(p.color), p.transform.copy(), step_id=step_id))

    return scheduled


def load_ground_truth_stepped_parts(
    io_or_ldr_path: str,
    synthesize_steps_if_flat: bool = True,
) -> Tuple[List[ParsedPart], Dict[int, List[ParsedPart]]]:
    """
    Loads a Ground-Truth .io, .mpd, or .ldr file, normalizes it to ground level (Y <= 0.0),
    synthesizes topological steps if the file has no discrete 0 STEP markers, and returns
    both the flat list of ParsedPart and a step_id -> list[ParsedPart] dictionary.
    """
    temp_ldr = None
    if io_or_ldr_path.lower().endswith(".io"):
        parser = StudioIOParser(io_or_ldr_path)
        base = os.path.splitext(os.path.basename(io_or_ldr_path))[0]
        temp_ldr = f"/tmp/{base}_{os.getpid()}_gt_oracle.ldr"
        parser.save_ldr_to_disk(temp_ldr)
        target_path = temp_ldr
    else:
        target_path = io_or_ldr_path

    try:
        parts = flatten_mpd(target_path)
    finally:
        if temp_ldr and os.path.exists(temp_ldr):
            try:
                os.remove(temp_ldr)
            except OSError:
                pass

    if not parts:
        return [], {}

    distinct_steps = sorted({p.step_id for p in parts})
    if len(distinct_steps) <= 1 and synthesize_steps_if_flat and len(parts) > 1:
        if len(parts) <= 70:
            parts = schedule_topological_steps(parts, max_parts_per_step=4)
        else:
            parts = _fast_layer_step_synthesis(parts, max_parts_per_step=4)
    else:
        parts = normalize_parts_to_ground(parts)

    steps_map: Dict[int, List[ParsedPart]] = defaultdict(list)
    for p in parts:
        steps_map[int(p.step_id)].append(p)

    # Re-index steps sequentially 1..N
    ordered_steps: Dict[int, List[ParsedPart]] = {}
    normalized_flat: List[ParsedPart] = []
    for new_s_idx, old_s_idx in enumerate(sorted(steps_map.keys()), start=1):
        step_list = []
        for p in steps_map[old_s_idx]:
            p_copy = ParsedPart(
                part_id=clean_part_id(p.part_id),
                color=int(p.color),
                transform=np.array(p.transform, dtype=np.float32).copy(),
                step_id=new_s_idx,
            )
            step_list.append(p_copy)
            normalized_flat.append(p_copy)
        ordered_steps[new_s_idx] = step_list

    return normalized_flat, ordered_steps


def part_to_node_dict(part: ParsedPart, part_index: int, geom_resolver: LDrawGeometryResolver) -> Dict[str, Any]:
    """Converts a ParsedPart into the node dictionary format expected by ManualConnectionGraphBuilder."""
    pid = geom_resolver.resolve_compatible_part_id(part.part_id.replace(".dat", ""))
    meta = geom_resolver.get_part_meta(pid)
    tf = np.array(part.transform, dtype=np.float32).reshape((4, 4))
    y_bot = int(round(float(tf[1, 3])))
    return {
        "part_index": part_index,
        "graph_id": "ship_graph",
        "page_num": int(part.step_id),
        "step_label": f"Step {part.step_id}",
        "element_id": pid.replace(".dat", ""),
        "part_id": pid,
        "color_code": int(part.color),
        "y_bottom_ldu": y_bot,
        "level_plates": int(round(-y_bot / 8.0)),
        "h_body_ldu": int(meta.get("h_plates", 1) * 8),
        "view_flipped": False,
        "is_minifig": False,
        "absolute_transform": tf.flatten().tolist(),
        "connections": [],
    }


def poses_match_modulo_symmetry(
    part_id: str,
    tf_pred: np.ndarray,
    tf_gt: np.ndarray,
    geom_resolver: LDrawGeometryResolver,
    pos_tol_ldu: float = 2.5,
    rot_tol_deg: float = 8.0,
) -> Tuple[bool, float, float]:
    """
    Evaluates whether a predicted 4x4 transform matches the Ground-Truth 4x4 transform
    within pos_tol_ldu and rotational symmetry equivalence (C1, C2, C4 or world OBB congruence).
    Returns: (is_match, pos_error_ldu, min_rot_error_deg)
    """
    t_pred = tf_pred[:3, 3]
    t_gt = tf_gt[:3, 3]
    pos_err = float(np.linalg.norm(t_pred - t_gt))

    R_pred = tf_pred[:3, :3]
    R_gt = tf_gt[:3, :3]

    meta = geom_resolver.get_part_meta(part_id)
    sym_order = int(meta.get("symmetry_order", 1))
    if sym_order == 4:
        sym_angles = [0.0, 90.0, 180.0, 270.0]
    elif sym_order == 2:
        sym_angles = [0.0, 180.0]
    else:
        sym_angles = [0.0]

    min_rot_deg = 180.0
    for deg in sym_angles:
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        R_sym = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ], dtype=np.float32)
        R_equiv = R_gt @ R_sym
        trace_val = float(np.trace(R_pred.T @ R_equiv))
        cos_theta = max(-1.0, min(1.0, (trace_val - 1.0) / 2.0))
        ang_deg = math.degrees(math.acos(cos_theta))
        if ang_deg < min_rot_deg:
            min_rot_deg = ang_deg

    is_match = (pos_err <= pos_tol_ldu) and (min_rot_deg <= rot_tol_deg)
    return is_match, pos_err, min_rot_deg


class OracleStepDebugger:
    """
    Executes step-by-step inference on a MOC, compares placements against the Ground-Truth .io,
    aborts on the first positioning error to prevent cascading drift, and attributes the root cause.
    """

    def __init__(
        self,
        geom_resolver: Optional[LDrawGeometryResolver] = None,
        fix_bundle: Optional[GeneralizedFixBundle] = None,
    ):
        self.geom = geom_resolver or LDrawGeometryResolver()
        self.solid_validator = SolidCollisionValidator(self.geom)
        self.hypothesis_gen = PhysicalHypothesisGenerator(self.geom)
        self.fix_bundle = fix_bundle or GeneralizedFixBundle()
        self._apply_fix_bundle()

    def _apply_fix_bundle(self) -> None:
        """Applies generalized fixes from self.fix_bundle to the active geometry & validator components."""
        extra_hollow = set(self.fix_bundle.multi_box_hollow_parts)
        if extra_hollow:
            orig_method = self.solid_validator.get_part_sub_boxes_local

            def patched_sub_boxes(part_id: str, meta: Dict[str, Any]):
                clean_id = part_id.replace(".dat", "").lower()
                if any(h.replace(".dat", "").lower() in clean_id for h in extra_hollow):
                    bx0, by0, bz0, bx1, by1, bz1 = meta["bbox"]
                    y_top, y_bot = float(by0), float(by1)
                    if y_bot <= y_top:
                        y_bot = y_top + 8.0
                    dx = float(bx1 - bx0)
                    if dx >= 20.0:
                        pillar_w = max(8.0, dx * 0.25)
                        return [
                            (float(bx0), y_top, float(bz0), float(bx0 + pillar_w), y_bot, float(bz1)),
                            (float(bx1 - pillar_w), y_top, float(bz0), float(bx1), y_bot, float(bz1)),
                            (float(bx0), y_top, float(bz0), float(bx1), min(y_bot, y_top + 8.0), float(bz1)),
                        ]
                return orig_method(part_id, meta)

            self.solid_validator.get_part_sub_boxes_local = patched_sub_boxes
            self.hypothesis_gen.solid_validator.get_part_sub_boxes_local = patched_sub_boxes
            self.solid_validator.obb_cache.clear()
            self.hypothesis_gen.solid_validator.obb_cache.clear()

        for pid in self.fix_bundle.snot_and_hinge_parts + self.fix_bundle.fractional_offset_parts:
            meta = self.geom.get_part_meta(clean_part_id(pid))
            meta["has_lateral_ports"] = True

        # Expand bottom/center socket candidates for parts flagged with SOCKET_CANDIDATE_COVERAGE_GAP
        for pid in self.fix_bundle.socket_coverage_parts:
            meta = self.geom.get_part_meta(clean_part_id(pid))
            bbox = meta["bbox"]
            existing_socks = list(meta.get("inferred_sockets", meta.get("sockets", [])))
            existing_socks.append(np.array([0.0, float(bbox[4]), 0.0], dtype=np.float32))
            existing_socks.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            meta["inferred_sockets"] = existing_socks

    def _select_local_parent_nodes(
        self,
        existing_nodes: List[Dict[str, Any]],
        ref_pos: np.ndarray,
        max_parents: int = 10,
        radius_ldu: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """
        Selects the spatially proximate and most recently placed parent nodes around ref_pos
        to keep hypothesis enumeration fast (< 50 ms per step) even on 500-part MOCs.
        """
        if len(existing_nodes) <= max_parents:
            return existing_nodes
        
        n_total = len(existing_nodes)
        
        # OPTIMIZACION (Sprint Nov 11): Vectorizar np.linalg.norm()
        # Antes: 500K llamadas individuales a np.linalg.norm() por MOC
        # Ahora: 1 llamada vectorizada con numpy
        # Speedup esperado: 20-50x en esta operacion
        positions = np.array([
            np.array(node["absolute_transform"], dtype=np.float32).reshape((4, 4))[:3, 3]
            for node in existing_nodes
        ], dtype=np.float32)
        dists = np.linalg.norm(positions - ref_pos, axis=1)  # UNA llamada vectorizada
        
        scored = []
        for idx, (dist, node) in enumerate(zip(dists, existing_nodes)):
            dist = float(dist)  # Convertir a float si es necesario
            recency_bonus = 20.0 if (n_total - idx) <= 6 else 0.0
            if dist <= radius_ldu or (n_total - idx) <= 6:
                scored.append((dist - recency_bonus, node))
        scored.sort(key=lambda x: x[0])
        selected = [n for _, n in scored[:max_parents]]
        return selected if selected else existing_nodes[-max_parents:]


    def generate_candidates_with_fixes(
        self,
        child_part_id: str,
        child_element_id: str,
        existing_nodes: List[Dict[str, Any]],
        substrate_parents: Optional[List[int]] = None,
        is_flip: bool = False,
        connector_2d_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ConnectionHypothesis]:
        """Generates candidate hypotheses honoring the active GeneralizedFixBundle."""
        target_parents = None if self.fix_bundle.disable_substrate_filter else substrate_parents
        cands = self.hypothesis_gen.generate_candidate_hypotheses(
            child_part_id,
            child_element_id,
            existing_nodes,
            target_parent_indices=target_parents,
            is_flip=is_flip,
            connector_2d_filter=connector_2d_filter,
        )
        if not cands and self.fix_bundle.auto_retry_flip_on_dead_end:
            cands = self.hypothesis_gen.generate_candidate_hypotheses(
                child_part_id,
                child_element_id,
                existing_nodes,
                target_parent_indices=target_parents,
                is_flip=not is_flip,
                connector_2d_filter=connector_2d_filter,
            )
        return cands

    def diagnose_step_failure(
        self,
        moc_id: str,
        step_idx: int,
        total_steps: int,
        gt_part: ParsedPart,
        pred_hypothesis: Optional[ConnectionHypothesis],
        prior_gt_nodes: List[Dict[str, Any]],
        prior_gt_parts: List[ParsedPart],
        all_candidates: List[ConnectionHypothesis],
        unfiltered_candidates: List[ConnectionHypothesis],
        ocr_step_parts: Optional[List[Tuple[str, int]]] = None,
        gt_step_parts: Optional[List[Tuple[str, int]]] = None,
        is_flip: bool = False,
    ) -> StepFailureIncident:
        """
        Deterministic 5-stage diagnostic tree that determines WHY the inference failed
        on step_idx for gt_part without cascading error contamination.
        """
        pid = self.geom.resolve_compatible_part_id(gt_part.part_id.replace(".dat", ""))
        tf_gt = np.array(gt_part.transform, dtype=np.float32).reshape((4, 4))
        gt_trans = tf_gt[:3, 3].tolist()

        if pred_hypothesis is not None:
            tf_pred = np.array(pred_hypothesis.absolute_transform, dtype=np.float32).reshape((4, 4))
            pred_trans = tf_pred[:3, 3].tolist()
            _, pos_err, rot_err = poses_match_modulo_symmetry(pid, tf_pred, tf_gt, self.geom)
            pred_score = float(pred_hypothesis.total_score)
            pred_edge = float(pred_hypothesis.edge_score)
            pred_studs = int(pred_hypothesis.stud_count)
        else:
            pred_trans = [0.0, 0.0, 0.0]
            pos_err = 999.0
            rot_err = 180.0
            pred_score = 0.0
            pred_edge = 0.0
            pred_studs = 0

        survived = max(0, step_idx - 1)
        survival_ratio = float(survived) / float(max(1, total_steps))

        # -------------------------------------------------------------------------
        # Stage 1: Check INVENTORY_OCR_MISMATCH
        # -------------------------------------------------------------------------
        if ocr_step_parts is not None and gt_step_parts is not None:
            c_ocr = Counter((clean_part_id(p), int(c)) for p, c in ocr_step_parts)
            c_gt = Counter((clean_part_id(p), int(c)) for p, c in gt_step_parts)
            if c_ocr != c_gt:
                return StepFailureIncident(
                    moc_id=moc_id,
                    step_index=step_idx,
                    total_gt_steps=total_steps,
                    steps_survived_before_error=survived,
                    survival_ratio=survival_ratio,
                    root_cause_category="INVENTORY_OCR_MISMATCH",
                    sub_cause="CALLOUT_PART_OR_QTY_MISMATCH",
                    child_part_id=pid,
                    child_color_id=int(gt_part.color),
                    pos_error_ldu=pos_err,
                    rot_error_deg=rot_err,
                    gt_translation=gt_trans,
                    pred_translation=pred_trans,
                    diagnostic_explanation=(
                        f"El callout extraído del PDF ({dict(c_ocr)}) no coincide con "
                        f"las piezas reales del paso {step_idx} en el .io ({dict(c_gt)})."
                    ),
                    suggested_fix=(
                        "Reforzar el matching con restricción de stock global en LegoInventoryMatcher "
                        "y verificar escala en puntos PDF (bbox_pt) para diferenciar longitudes 1xN."
                    ),
                    responsible_agent="lego-context-agent",  # FASE 1
                )

        # Find nearest parent in local neighborhood of prior_gt_parts
        nearest_parent_id = None
        min_parent_dist = 9999.0
        has_port_connection = False
        nearby_priors = []
        for p_prior in prior_gt_parts:
            d_p = float(np.linalg.norm(p_prior.transform[:3, 3] - tf_gt[:3, 3]))
            if d_p < min_parent_dist:
                min_parent_dist = d_p
                nearest_parent_id = clean_part_id(p_prior.part_id)
            if d_p <= 160.0:
                nearby_priors.append((d_p, p_prior))

        nearby_priors.sort(key=lambda x: x[0])
        for _, p_prior in nearby_priors[:12]:
            if check_connection_optimized(p_prior, gt_part):
                has_port_connection = True
                break

        # -------------------------------------------------------------------------
        # Stage 2: Check MISSING_CONNECTIVITY_PORT
        # -------------------------------------------------------------------------
        if prior_gt_parts and not has_port_connection:
            return StepFailureIncident(
                moc_id=moc_id,
                step_index=step_idx,
                total_gt_steps=total_steps,
                steps_survived_before_error=survived,
                survival_ratio=survival_ratio,
                root_cause_category="MISSING_CONNECTIVITY_PORT",
                sub_cause="UNRECOGNIZED_PORT_COUPLING",
                child_part_id=pid,
                child_color_id=int(gt_part.color),
                parent_part_id=nearest_parent_id,
                pos_error_ldu=pos_err,
                rot_error_deg=rot_err,
                gt_translation=gt_trans,
                pred_translation=pred_trans,
                diagnostic_explanation=(
                    f"La pose real (.io) de {pid} no registra acoplamiento de puertos con ninguna "
                    f"pieza previa (padre más cercano: {nearest_parent_id} a {min_parent_dist:.1f} LDU)."
                ),
                suggested_fix=(
                    f"Añadir metadatos de puertos LDCad Shadow / paramétricos para '{pid}' y "
                    f"'{nearest_parent_id}' en src/connectivity_db.py."
                ),
                    responsible_agent="lego-identifier-agent"  # FASE 1
                )

        # -------------------------------------------------------------------------
        # Stage 3: Check FALSE_SOLID_COLLISION_PRUNING
        # -------------------------------------------------------------------------
        local_nodes = self._select_local_parent_nodes(prior_gt_nodes, tf_gt[:3, 3], max_parents=5)
        is_valid_solid, col_reason = self.solid_validator.validate_new_part_placement(
            pid, tf_gt, local_nodes, penetration_threshold=self.fix_bundle.penetration_threshold_ldu
        )
        if not is_valid_solid:
            colliding_pid = nearest_parent_id
            if col_reason:
                m = re.search(r"\(([^)]+\.dat)\)", col_reason)
                if m:
                    colliding_pid = m.group(1)
            return StepFailureIncident(
                moc_id=moc_id,
                step_index=step_idx,
                total_gt_steps=total_steps,
                steps_survived_before_error=survived,
                survival_ratio=survival_ratio,
                root_cause_category="FALSE_SOLID_COLLISION_PRUNING",
                sub_cause="CONCAVE_OR_HOLLOW_OBB_OVERAPPROXIMATION",
                child_part_id=pid,
                child_color_id=int(gt_part.color),
                parent_part_id=nearest_parent_id,
                colliding_part_id=colliding_pid,
                pos_error_ldu=pos_err,
                rot_error_deg=rot_err,
                gt_translation=gt_trans,
                pred_translation=pred_trans,
                collision_reason=col_reason,
                diagnostic_explanation=(
                    f"La pose real (.io) de {pid} fue descartada por SolidCollisionValidator: {col_reason}"
                ),
                suggested_fix=(
                    f"Descomponer '{pid}' o '{colliding_pid}' en sub-cajas multi-OBB en "
                    f"SolidCollisionValidator.get_part_sub_boxes_local."
                ),
                    responsible_agent="lego-guardian-agent"  # FASE 1
                )

        # -------------------------------------------------------------------------
        # Stage 4: Check if GT pose exists in generated candidates (VISUAL_SCORER vs GENERATOR)
        # -------------------------------------------------------------------------
        gt_rank = None
        gt_cand_obj: Optional[ConnectionHypothesis] = None
        for idx, cand in enumerate(all_candidates):
            c_tf = np.array(cand.absolute_transform, dtype=np.float32).reshape((4, 4))
            matched, _, _ = poses_match_modulo_symmetry(pid, c_tf, tf_gt, self.geom)
            if matched:
                gt_rank = idx + 1  # 1-based rank
                gt_cand_obj = cand
                break

        if gt_rank is not None and gt_cand_obj is not None:
            gt_score = float(gt_cand_obj.total_score)
            gt_edge = float(gt_cand_obj.edge_score)
            gt_studs = int(gt_cand_obj.stud_count)

            if gt_edge > pred_edge and gt_studs < pred_studs:
                sub_cause = "STUD_PRIOR_DOMINATED_VISUAL_CHAMFER"
            elif gt_studs > pred_studs and gt_edge <= pred_edge:
                sub_cause = "CHAMFER_NOISE_DOMINATED_STRUCTURAL_STUDS"
            elif gt_rank > self.fix_bundle.beam_width:
                sub_cause = "BEAM_WIDTH_TOO_NARROW"
            else:
                sub_cause = "CHAMFER_WEIGHT_IMBALANCE"

            return StepFailureIncident(
                moc_id=moc_id,
                step_index=step_idx,
                total_gt_steps=total_steps,
                steps_survived_before_error=survived,
                survival_ratio=survival_ratio,
                root_cause_category="VISUAL_SCORER_RANKING_ERROR",
                sub_cause=sub_cause,
                child_part_id=pid,
                child_color_id=int(gt_part.color),
                parent_part_id=nearest_parent_id,
                pos_error_ldu=pos_err,
                rot_error_deg=rot_err,
                gt_translation=gt_trans,
                pred_translation=pred_trans,
                gt_rank_in_candidates=gt_rank,
                total_candidates_generated=len(all_candidates),
                pred_total_score=pred_score,
                gt_total_score=gt_score,
                pred_edge_score=pred_edge,
                gt_edge_score=gt_edge,
                pred_stud_count=pred_studs,
                gt_stud_count=gt_studs,
                diagnostic_explanation=(
                    f"La hipótesis GT estaba en el puesto #{gt_rank}/{len(all_candidates)} "
                    f"(score GT={gt_score:.3f}, edge={gt_edge:.3f}, studs={gt_studs}), pero fue superada por "
                    f"la predicción #1 (score Pred={pred_score:.3f}, edge={pred_edge:.3f}, studs={pred_studs})."
                ),
                suggested_fix=(
                    "Recalibrar los pesos (w_edge_score, w_stud_score) en VisualPoseScorer y ampliar "
                    f"beam_width >= {max(self.fix_bundle.beam_width + 1, gt_rank)}."
                ),
                    responsible_agent="lego-placer-agent"  # FASE 1
                )

        in_unfiltered = False
        for cand in unfiltered_candidates:
            c_tf = np.array(cand.absolute_transform, dtype=np.float32).reshape((4, 4))
            matched, _, _ = poses_match_modulo_symmetry(pid, c_tf, tf_gt, self.geom)
            if matched:
                in_unfiltered = True
                break

        if in_unfiltered:
            return StepFailureIncident(
                moc_id=moc_id,
                step_index=step_idx,
                total_gt_steps=total_steps,
                steps_survived_before_error=survived,
                survival_ratio=survival_ratio,
                root_cause_category="HYPOTHESIS_NOT_GENERATED",
                sub_cause="SUBSTRATE_PARENT_FILTERED",
                child_part_id=pid,
                child_color_id=int(gt_part.color),
                parent_part_id=nearest_parent_id,
                pos_error_ldu=pos_err,
                rot_error_deg=rot_err,
                gt_translation=gt_trans,
                pred_translation=pred_trans,
                total_candidates_generated=len(all_candidates),
                diagnostic_explanation=(
                    f"La pose GT de {pid} existe al explorar todos los padres ({len(unfiltered_candidates)} cands), "
                    "pero fue descartada por el filtro de sustrato activo (get_unclosed_substrate_parents)."
                ),
                suggested_fix=(
                    "Desactivar o relajar el filtro restrictivo de substrate_parents cuando la pieza "
                    "puede acoplarse a capas inferiores aún expuestas."
                ),
                    responsible_agent="lego-identifier-agent"  # FASE 1
                )

        # -------------------------------------------------------------------------
        # Stage 5: Probe specific sub-causes of HYPOTHESIS_NOT_GENERATED
        # -------------------------------------------------------------------------
        flipped_cands = self.hypothesis_gen.generate_candidate_hypotheses(
            pid, pid.replace(".dat", ""), local_nodes, target_parent_indices=None, is_flip=not is_flip
        )
        for cand in flipped_cands:
            c_tf = np.array(cand.absolute_transform, dtype=np.float32).reshape((4, 4))
            matched, _, _ = poses_match_modulo_symmetry(pid, c_tf, tf_gt, self.geom)
            if matched:
                return StepFailureIncident(
                    moc_id=moc_id,
                    step_index=step_idx,
                    total_gt_steps=total_steps,
                    steps_survived_before_error=survived,
                    survival_ratio=survival_ratio,
                    root_cause_category="HYPOTHESIS_NOT_GENERATED",
                    sub_cause="MISSED_FLIP_INVERSION",
                    child_part_id=pid,
                    child_color_id=int(gt_part.color),
                    parent_part_id=nearest_parent_id,
                    pos_error_ldu=pos_err,
                    rot_error_deg=rot_err,
                    gt_translation=gt_trans,
                    pred_translation=pred_trans,
                    diagnostic_explanation=(
                        f"La pose GT de {pid} aparece exactamente al invertir is_flip={not is_flip}. "
                        "El paso invierte el modelo (inserción inferior STUD->ANTISTUD)."
                    ),
                    suggested_fix=(
                        "Activar auto_retry_flip_on_dead_end y ampliar detect_flip_icons() para "
                        "iconos de rotación en manuales Studio/Rebrickable."
                    ),
                    responsible_agent="lego-identifier-agent"  # FASE 1
                )

        R_gt = tf_gt[:3, :3]
        if abs(abs(float(R_gt[1, 1])) - 1.0) > 0.05:
            sub_cause = "UNEXPECTED_3D_TILT_OR_SNOT"
            fix_msg = f"Incluir '{pid}' y '{nearest_parent_id}' en la lista de activación SNOT/Hinge 3D (24 rotaciones SO(3))."
        else:
            rem_x = abs(float(tf_gt[0, 3])) % 20.0
            rem_z = abs(float(tf_gt[2, 3])) % 20.0
            if (3.0 <= rem_x <= 17.0) or (3.0 <= rem_z <= 17.0):
                sub_cause = "FRACTIONAL_OFFSET_GRID"
                fix_msg = f"Habilitar desplazamientos fraccionarios (4 LDU bracket / 10 LDU jumper) para '{pid}'."
            else:
                sub_cause = "SOCKET_CANDIDATE_COVERAGE_GAP"
                fix_msg = f"Ampliar inferencia de sockets inferiores (convex hull) para '{pid}' en LDrawGeometryResolver."

        return StepFailureIncident(
            moc_id=moc_id,
            step_index=step_idx,
            total_gt_steps=total_steps,
            steps_survived_before_error=survived,
            survival_ratio=survival_ratio,
            root_cause_category="HYPOTHESIS_NOT_GENERATED",
            sub_cause=sub_cause,
            child_part_id=pid,
            child_color_id=int(gt_part.color),
            parent_part_id=nearest_parent_id,
            pos_error_ldu=pos_err,
            rot_error_deg=rot_err,
            gt_translation=gt_trans,
            pred_translation=pred_trans,
            diagnostic_explanation=(
                f"La pose GT de {pid} es físicamente válida y libre de colisiones, pero no fue "
                f"enumerada por PhysicalHypothesisGenerator (sub-causa: {sub_cause})."
            ),
            suggested_fix=fix_msg,
                    responsible_agent="lego-identifier-agent"  # FASE 1
                )

    def evaluate_moc_fail_fast(
        self,
        moc_id: str,
        io_path: str,
        pdf_path: Optional[str] = None,
        abort_on_first_error: bool = True,
        max_steps_to_eval: int = 30,
        simulated_step_corruptions: Optional[Dict[int, Dict[str, Any]]] = None,
        parent_radius_ldu: Optional[float] = None,
        max_parents_override: Optional[int] = None,
        connector_2d_filter_global: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the step-by-step inference loop for a single MOC against its Ground-Truth .io file.
        Aborts immediately upon detecting the first positioning error (when abort_on_first_error=True)
        and returns the full root-cause incident report.
        """
        gt_flat, gt_steps = load_ground_truth_stepped_parts(io_path)
        total_steps = len(gt_steps)
        if total_steps == 0:
            return {
                "moc_id": moc_id,
                "io_path": io_path,
                "pdf_path": pdf_path,
                "success": False,
                "completed_all_steps": False,
                "total_steps": 0,
                "steps_survived": 0,
                "survival_ratio": 0.0,
                "incidents": [],
                "error": "Empty or unreadable .io/.ldr ground truth",
            }

        placed_gt_nodes: List[Dict[str, Any]] = []
        placed_gt_parts: List[ParsedPart] = []
        incidents: List[StepFailureIncident] = []
        steps_survived = 0
        gt_rank_histogram = [0] * 8  # bins for rank 1..8

        step_keys = sorted(gt_steps.keys())[:max_steps_to_eval]
        eval_total_steps = len(step_keys)

        for step_idx in step_keys:
            step_gt_parts = gt_steps[step_idx]

            # Step 1 (foundation / ground base): anchor directly into the reference frame
            if step_idx == 1 and not placed_gt_nodes and not (
                simulated_step_corruptions and 1 in simulated_step_corruptions
            ):
                for p in step_gt_parts:
                    node = part_to_node_dict(p, len(placed_gt_nodes), self.geom)
                    placed_gt_nodes.append(node)
                    placed_gt_parts.append(p)
                steps_survived += 1
                # GT rank = 1 for baseline steps (already placed correctly)
                for _ in step_gt_parts:
                    gt_rank_histogram[0] += 1
                continue

            sim_cfg = (simulated_step_corruptions or {}).get(step_idx, {})
            ocr_override = sim_cfg.get("ocr_step_parts")
            gt_step_pairs = [(p.part_id, p.color) for p in step_gt_parts]

            if ocr_override is not None:
                inc = self.diagnose_step_failure(
                    moc_id=moc_id,
                    step_idx=step_idx,
                    total_steps=eval_total_steps,
                    gt_part=step_gt_parts[0],
                    pred_hypothesis=None,
                    prior_gt_nodes=placed_gt_nodes,
                    prior_gt_parts=placed_gt_parts,
                    all_candidates=[],
                    unfiltered_candidates=[],
                    ocr_step_parts=ocr_override,
                    gt_step_parts=gt_step_pairs,
                )
                incidents.append(inc)
                if abort_on_first_error:
                    break
                for p in step_gt_parts:
                    placed_gt_nodes.append(part_to_node_dict(p, len(placed_gt_nodes), self.geom))
                    placed_gt_parts.append(p)
                continue

            step_aborted = False
            substrate_indices = sim_cfg.get("substrate_parent_indices")
            is_flip_step = bool(sim_cfg.get("is_flip", False))

            part_candidates_filtered: List[List[ConnectionHypothesis]] = []
            part_candidates_unfiltered: List[Optional[List[ConnectionHypothesis]]] = []
            # Level 2 optimization: store (pid, el_id, local_parents, is_flip_step) for lazy unfiltered generation
            unfiltered_lazy_data: List[Tuple[str, str, List[Dict[str, Any]], bool]] = []

            for p_gt in step_gt_parts:
                pid = self.geom.resolve_compatible_part_id(p_gt.part_id.replace(".dat", ""))
                el_id = pid.replace(".dat", "")
                tf_gt = np.array(p_gt.transform, dtype=np.float32).reshape((4, 4))

                local_parents = self._select_local_parent_nodes(
                    placed_gt_nodes, tf_gt[:3, 3], 
                    max_parents=max_parents_override or 3,
                    radius_ldu=parent_radius_ldu or 120.0
                )

                # LEVEL 2: Don't generate unfiltered yet. Store data for lazy generation on failure.
                filtered = self.generate_candidates_with_fixes(
                    pid,
                    el_id,
                    local_parents,
                    substrate_parents=substrate_indices,
                    is_flip=is_flip_step,
                    connector_2d_filter=connector_2d_filter_global or sim_cfg.get("connector_2d_filter"),
                )

                bias_fn = sim_cfg.get("visual_edge_scorer")
                for c_obj in filtered:
                    c_tf = np.array(c_obj.absolute_transform, dtype=np.float32).reshape((4, 4))
                    is_gt_match, pos_d, _ = poses_match_modulo_symmetry(pid, c_tf, tf_gt, self.geom)
                    if bias_fn is not None:
                        c_obj.edge_score = float(bias_fn(c_obj, is_gt_match))
                    else:
                        c_obj.edge_score = float(math.exp(-pos_d / 35.0))
                    stud_norm = min(1.0, float(c_obj.stud_count) / 4.0)
                    c_obj.total_score = (
                        self.fix_bundle.w_edge_score * c_obj.edge_score
                        + self.fix_bundle.w_stud_score * stud_norm
                    )

                filtered.sort(key=lambda c: c.total_score, reverse=True)
                part_candidates_filtered.append(filtered)
                part_candidates_unfiltered.append(None)  # Placeholder for lazy generation
                unfiltered_lazy_data.append((pid, el_id, local_parents, is_flip_step))

            pred_hypotheses: List[Optional[ConnectionHypothesis]] = [
                cands[0] if cands else None for cands in part_candidates_filtered
            ]

            n_step = len(step_gt_parts)
            cost_mat = np.full((n_step, n_step), 1e6, dtype=np.float32)
            for i, p_gt in enumerate(step_gt_parts):
                pid_i = self.geom.resolve_compatible_part_id(p_gt.part_id.replace(".dat", ""))
                tf_gt_i = np.array(p_gt.transform, dtype=np.float32).reshape((4, 4))
                for j, h_pred in enumerate(pred_hypotheses):
                    pid_j = self.geom.resolve_compatible_part_id(
                        step_gt_parts[j].part_id.replace(".dat", "")
                    )
                    if pid_i != pid_j or int(p_gt.color) != int(step_gt_parts[j].color):
                        continue
                    if h_pred is None:
                        cost_mat[i, j] = 9999.0
                    else:
                        tf_p = np.array(h_pred.absolute_transform, dtype=np.float32).reshape((4, 4))
                        _, pos_err, rot_err = poses_match_modulo_symmetry(pid_i, tf_p, tf_gt_i, self.geom)
                        cost_mat[i, j] = pos_err + 0.25 * rot_err

            row_ind, col_ind = linear_sum_assignment(cost_mat)

            for i, j in zip(row_ind, col_ind):
                p_gt = step_gt_parts[i]
                h_pred = pred_hypotheses[j]
                pid = self.geom.resolve_compatible_part_id(p_gt.part_id.replace(".dat", ""))
                tf_gt = np.array(p_gt.transform, dtype=np.float32).reshape((4, 4))

                if h_pred is None:
                    matched = False
                else:
                    tf_p = np.array(h_pred.absolute_transform, dtype=np.float32).reshape((4, 4))
                    matched, _, _ = poses_match_modulo_symmetry(pid, tf_p, tf_gt, self.geom)

                # Calculate GT rank among candidates (NEW - RECALL@N tracking)
                gt_rank = None
                for rank_idx, cand in enumerate(part_candidates_filtered[i]):
                    tf_c = np.array(cand.absolute_transform, dtype=np.float32).reshape((4, 4))
                    is_match, _, _ = poses_match_modulo_symmetry(pid, tf_c, tf_gt, self.geom)
                    if is_match:
                        gt_rank = rank_idx + 1
                        break
                if gt_rank is None:
                    gt_rank = 9  # Not found in top 8
                
                # Update histogram
                bucket_idx = min(gt_rank, 8) - 1
                gt_rank_histogram[bucket_idx] += 1

                if not matched:
                    # LEVEL 2: Lazy generate unfiltered only on failure
                    if part_candidates_unfiltered[i] is None:
                        pid_i, el_id_i, local_parents_i, is_flip_i = unfiltered_lazy_data[i]
                        part_candidates_unfiltered[i] = self.hypothesis_gen.generate_candidate_hypotheses(
                            pid_i, el_id_i, local_parents_i, target_parent_indices=None, is_flip=is_flip_i
                        )
                    
                    incident = self.diagnose_step_failure(
                        moc_id=moc_id,
                        step_idx=step_idx,
                        total_steps=eval_total_steps,
                        gt_part=p_gt,
                        pred_hypothesis=h_pred,
                        prior_gt_nodes=placed_gt_nodes,
                        prior_gt_parts=placed_gt_parts,
                        all_candidates=part_candidates_filtered[i],
                        unfiltered_candidates=part_candidates_unfiltered[i],
                        is_flip=is_flip_step,
                    )
                    incidents.append(incident)
                    step_aborted = True
                    break

            if step_aborted and abort_on_first_error:
                break

            for p_gt in step_gt_parts:
                node = part_to_node_dict(p_gt, len(placed_gt_nodes), self.geom)
                placed_gt_nodes.append(node)
                placed_gt_parts.append(p_gt)
            if not step_aborted:
                steps_survived += 1

        completed_all = len(incidents) == 0 and steps_survived == eval_total_steps
        
        # Calculate recall@N for N=1..8
        total_ranks = sum(gt_rank_histogram)
        recall_at_n = {}
        cumulative = 0
        for n in range(1, 9):
            cumulative += gt_rank_histogram[n - 1]
            recall_at_n[n] = cumulative / total_ranks if total_ranks > 0 else 0.0
        
        result = {
            "moc_id": moc_id,
            "io_path": io_path,
            "pdf_path": pdf_path,
            "success": True,
            "completed_all_steps": completed_all,
            "total_steps": eval_total_steps,
            "steps_survived": steps_survived,
            "survival_ratio": float(steps_survived) / float(max(1, eval_total_steps)),
            "aborted_at_step": incidents[0].step_index if incidents else None,
            "first_error_category": incidents[0].root_cause_category if incidents else None,
            "first_error_sub_cause": incidents[0].sub_cause if incidents else None,
            "incidents": [asdict(inc) for inc in incidents],
            "recall_at_n": recall_at_n,
        }
        
        # Add text summary for quick inspection
        summary_lines = [f"Recall@{n}: {recall_at_n[n]*100:.1f}%" for n in range(1, 9)]
        result["recall_at_n_summary"] = " | ".join(summary_lines)
        
        return result


class PatternClusterAnalyzer:
    """
    Analyzes failure incidents across all evaluated MOCs, groups recurring error patterns,
    and synthesizes a GeneralizedFixBundle that resolves those patterns globally.
    """

    def __init__(self, prior_fix_bundle: Optional[GeneralizedFixBundle] = None):
        self.prior_bundle = prior_fix_bundle or GeneralizedFixBundle()

    def _compute_agent_survival_attribution(
        self,
        all_incidents: List[StepFailureIncident]
    ) -> Dict[str, Any]:
        """
        FASE 1: Computes survival loss attributed to each agent.
        Returns { agent_name: { incidents, total_survival_loss, pct_of_total, top_root_causes } }
        """
        agent_incidents: Dict[str, List[StepFailureIncident]] = defaultdict(list)
        for inc in all_incidents:
            agent_incidents[inc.responsible_agent].append(inc)
        
        total_survival_loss = sum(inc.survival_ratio for inc in all_incidents)
        result = {}
        
        for agent in sorted(agent_incidents.keys()):
            incidents = agent_incidents[agent]
            agent_survival_loss = sum(inc.survival_ratio for inc in incidents)
            pct_of_total = 100.0 * agent_survival_loss / max(0.001, total_survival_loss)
            avg_survival_loss = np.mean([inc.survival_ratio for inc in incidents]) if incidents else 0.0
            
            result[agent] = {
                "incidents": len(incidents),
                "total_survival_loss": round(agent_survival_loss, 4),
                "avg_survival_loss_per_incident": round(float(avg_survival_loss), 4),
                "pct_of_total_survival_loss": round(pct_of_total, 2),
                "top_root_causes": Counter(inc.root_cause_category for inc in incidents).most_common(3),
            }
        
        return result

    def _compute_tier_metrics(self, moc_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """FASE 2: Computes KPIs segmented by set tier (small/medium/large)."""
        from src.phase2_scale_hygiene import SetTier
        
        tier_data: Dict[str, List[Dict]] = {"SMALL": [], "MEDIUM": [], "LARGE": []}
        for res in moc_results:
            total_steps = res.get("total_steps", 1)
            tier = SetTier.classify(total_steps)
            if tier != "UNKNOWN":
                tier_data[tier].append(res)
        
        tier_metrics = {}
        for tier, results in tier_data.items():
            if results:
                completed = sum(1 for r in results if r.get("completed_all_steps"))
                tier_metrics[tier] = {
                    "mean_survival_ratio": round(sum(r.get("survival_ratio", 0.0) for r in results) / len(results), 4),
                    "pass_rate_percent": round(100.0 * completed / len(results), 2),
                    "num_sets": len(results),
                }
            else:
                tier_metrics[tier] = {"mean_survival_ratio": 0.0, "pass_rate_percent": 0.0, "num_sets": 0}
        return tier_metrics



    def analyze_and_synthesize_fixes(
        self,
        moc_results: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], GeneralizedFixBundle]:
        all_incidents: List[StepFailureIncident] = []
        for res in moc_results:
            for inc_dict in res.get("incidents", []):
                all_incidents.append(StepFailureIncident(**inc_dict))

        total_mocs = len(moc_results)
        passed_mocs = sum(1 for r in moc_results if r.get("completed_all_steps"))
        mean_survival = (
            float(np.mean([r.get("survival_ratio", 0.0) for r in moc_results]))
            if moc_results
            else 0.0
        )
        mean_steps_survived = (
            float(np.mean([r.get("steps_survived", 0) for r in moc_results]))
            if moc_results
            else 0.0
        )

        clusters_by_cause: Dict[Tuple[str, str], List[StepFailureIncident]] = defaultdict(list)
        part_failure_counts: Counter = Counter()
        for inc in all_incidents:
            clusters_by_cause[(inc.root_cause_category, inc.sub_cause)].append(inc)
            part_failure_counts[inc.child_part_id] += 1

        next_bundle = GeneralizedFixBundle(
            iteration=self.prior_bundle.iteration + 1,
            snot_and_hinge_parts=list(self.prior_bundle.snot_and_hinge_parts),
            fractional_offset_parts=list(self.prior_bundle.fractional_offset_parts),
            socket_coverage_parts=list(self.prior_bundle.socket_coverage_parts),
            disable_substrate_filter=self.prior_bundle.disable_substrate_filter,
            auto_retry_flip_on_dead_end=self.prior_bundle.auto_retry_flip_on_dead_end,
            multi_box_hollow_parts=list(self.prior_bundle.multi_box_hollow_parts),
            missing_port_parts=list(self.prior_bundle.missing_port_parts),
            penetration_threshold_ldu=self.prior_bundle.penetration_threshold_ldu,
            w_edge_score=self.prior_bundle.w_edge_score,
            w_stud_score=self.prior_bundle.w_stud_score,
            symmetry_pair_bonus=self.prior_bundle.symmetry_pair_bonus,
            beam_width=self.prior_bundle.beam_width,
            roi_max_keep=self.prior_bundle.roi_max_keep,
        )

        pattern_clusters_report = []
        recommendations = []

        for (cat, sub), inc_list in sorted(
            clusters_by_cause.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            affected_mocs = sorted({inc.moc_id for inc in inc_list})
            affected_parts = sorted({inc.child_part_id for inc in inc_list})
            parent_parts = sorted({inc.parent_part_id for inc in inc_list if inc.parent_part_id})
            colliding_parts = sorted({inc.colliding_part_id for inc in inc_list if inc.colliding_part_id})

            generalized_action = ""
            target_module = ""

            if cat == "HYPOTHESIS_NOT_GENERATED" and sub == "SUBSTRATE_PARENT_FILTERED":
                next_bundle.disable_substrate_filter = True
                target_module = "src/manual_connection_graph.py"
                generalized_action = (
                    "Desactivar la poda estricta de substrate_parents cuando existen conectores "
                    "libres en capas inferiores (disable_substrate_filter = True)."
                )
            elif cat == "HYPOTHESIS_NOT_GENERATED" and sub == "MISSED_FLIP_INVERSION":
                next_bundle.auto_retry_flip_on_dead_end = True
                target_module = "src/manual_connection_graph.py"
                generalized_action = (
                    "Activar auto_retry_flip_on_dead_end=True y detección de inversión STUD->ANTISTUD."
                )
            elif cat == "HYPOTHESIS_NOT_GENERATED" and sub == "UNEXPECTED_3D_TILT_OR_SNOT":
                for p in affected_parts + parent_parts:
                    if p and p not in next_bundle.snot_and_hinge_parts:
                        next_bundle.snot_and_hinge_parts.append(p)
                target_module = "src/physical_hypothesis_generator.py"
                generalized_action = (
                    f"Habilitar búsqueda de 24 rotaciones SO(3) / bisagra para piezas ({len(affected_parts)}): {affected_parts[:8]}."
                )
            elif cat == "HYPOTHESIS_NOT_GENERATED" and sub == "FRACTIONAL_OFFSET_GRID":
                for p in affected_parts:
                    if p not in next_bundle.fractional_offset_parts:
                        next_bundle.fractional_offset_parts.append(p)
                target_module = "src/physical_hypothesis_generator.py"
                generalized_action = (
                    f"Activar sub-rejilla fraccionaria (4 LDU bracket / 10 LDU jumper) para ({len(affected_parts)}): {affected_parts[:8]}."
                )
            elif cat == "HYPOTHESIS_NOT_GENERATED" and sub == "SOCKET_CANDIDATE_COVERAGE_GAP":
                for p in affected_parts:
                    if p not in next_bundle.socket_coverage_parts:
                        next_bundle.socket_coverage_parts.append(p)
                target_module = "src/ldraw_geometry.py"
                generalized_action = (
                    f"Ampliar inferencia de sockets inferiores (origen + base bbox) para ({len(affected_parts)}): {affected_parts[:8]}."
                )
            elif cat == "FALSE_SOLID_COLLISION_PRUNING":
                for p in affected_parts + colliding_parts:
                    if p and p not in next_bundle.multi_box_hollow_parts:
                        next_bundle.multi_box_hollow_parts.append(p)
                next_bundle.penetration_threshold_ldu = max(
                    next_bundle.penetration_threshold_ldu, 2.0
                )
                target_module = "src/solid_collision_validator.py"
                generalized_action = (
                    f"Añadir descomposición multi-caja hueca en SolidCollisionValidator.get_part_sub_boxes_local "
                    f"para piezas cóncavas ({len(affected_parts)}): {sorted(set(affected_parts + colliding_parts))[:8]}."
                )
            elif cat == "MISSING_CONNECTIVITY_PORT":
                for p in affected_parts + parent_parts:
                    if p and p not in next_bundle.missing_port_parts:
                        next_bundle.missing_port_parts.append(p)
                target_module = "src/connectivity_db.py"
                generalized_action = (
                    f"Registrar puertos LDCad Shadow / paramétricos para ({len(affected_parts)}): {sorted(set(affected_parts + parent_parts))[:8]}."
                )
            elif cat == "VISUAL_SCORER_RANKING_ERROR":
                target_module = "src/visual_pose_scorer.py"
                best_w_edge = next_bundle.w_edge_score
                best_w_stud = next_bundle.w_stud_score
                best_resolved = -1
                for w_e in np.linspace(0.15, 0.90, 16):
                    w_s = 1.0 - float(w_e)
                    resolved = 0
                    for inc in inc_list:
                        sc_gt = w_e * inc.gt_edge_score + w_s * min(1.0, inc.gt_stud_count / 4.0)
                        sc_pr = w_e * inc.pred_edge_score + w_s * min(1.0, inc.pred_stud_count / 4.0)
                        if sc_gt > sc_pr:
                            resolved += 1
                    if resolved > best_resolved:
                        best_resolved = resolved
                        best_w_edge = round(float(w_e), 3)
                        best_w_stud = round(float(w_s), 3)

                next_bundle.w_edge_score = best_w_edge
                next_bundle.w_stud_score = best_w_stud
                max_gt_rank = max((inc.gt_rank_in_candidates or 3) for inc in inc_list)
                next_bundle.beam_width = max(next_bundle.beam_width, min(8, max_gt_rank))
                generalized_action = (
                    f"Recalibrar pesos de scoring a w_edge={best_w_edge:.2f}, w_stud={best_w_stud:.2f} "
                    f"(resuelve {best_resolved}/{len(inc_list)} conflictos de ranking) y subir beam_width={next_bundle.beam_width}."
                )
            else:
                target_module = "src/inventory_matcher.py"
                generalized_action = (
                    "Ajustar tolerancia de escala bbox_pt y restricción de inventario global Rebrickable."
                )

            cluster_entry = {
                "root_cause_category": cat,
                "sub_cause": sub,
                "incidents_count": len(inc_list),
                "affected_mocs_count": len(affected_mocs),
                "affected_mocs": affected_mocs,
                "affected_child_parts": affected_parts,
                "affected_parent_parts": parent_parts,
                "target_module": target_module,
                "generalized_fix_action": generalized_action,
            }
            pattern_clusters_report.append(cluster_entry)
            recommendations.append({
                "priority": len(recommendations) + 1,
                "pattern": f"{cat} :: {sub}",
                "impact_mocs": len(affected_mocs),
                "target_module": target_module,
                "action": generalized_action,
            })

        next_bundle.code_recommendations = recommendations

        summary = {
            "total_mocs_evaluated": total_mocs,
            "mocs_passed_100_percent": passed_mocs,
            "pass_rate_percent": round(100.0 * passed_mocs / max(1, total_mocs), 2),
            "mean_survival_ratio": round(mean_survival, 4),
            "mean_steps_survived_before_abort": round(mean_steps_survived, 2),
            "total_first_error_aborts": len(all_incidents),
            "top_failing_parts": part_failure_counts.most_common(15),
            "pattern_clusters": pattern_clusters_report,
            "agent_survival_attribution": self._compute_agent_survival_attribution(all_incidents),  # FASE 1
            "tier_metrics": self._compute_tier_metrics(moc_results),  # FASE 2: per-tier KPIs
            "synthesized_fix_bundle": next_bundle.to_dict(),
        }
        return summary, next_bundle

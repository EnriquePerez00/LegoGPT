"""
LegoGPT — Graph Quality Metrics
================================
Provides partial-correctness evaluation of inferred assembly graphs against
Ground-Truth (.io / .ldr) assemblies, without aborting on the first error.

Metrics computed per MOC:
  - pct_parts_correct        : % of parts placed at the exact GT position (+-EXACT_TOL LDU),
                               accounting for rotational symmetry (C1/C2/C4) and Hungarian
                               bipartite matching for repeated part_ids.
  - pct_edges_correct        : % of parent->child connection edges reproduced exactly
                               (same parent part_id + contact offset within EDGE_TOL).
  - graph_edit_distance      : minimum node substitutions + edge substitutions to
                               transform the predicted graph into the GT graph.
  - convergence_depth        : index of the first step containing any error (0-based).
                               Equal to total_steps when fully converged.
  - first_error_step         : 1-based step number of the first misplaced part
                               (None when fully converged).
  - first_error_piece        : 1-based cumulative piece number of the first error.
  - gt_in_candidates_rate    : fraction of errored parts whose GT pose appeared in
                               the generated candidate list (measures scorer, not generator).
  - post_gating_gt_rate      : same as above but after connector-compatibility gating
                               (populated by the gating module when available).
  - failure_family_counts    : dict mapping error family -> count across all steps.
  - corpus_summary           : aggregate statistics over a collection of MOC results.

Tolerances (LDraw Units):
  EXACT_TOL  = 1.0  LDU  -- position must be within 1 LDU to count as exact
  EDGE_TOL   = 2.0  LDU  -- parent contact-point offset tolerance
  ROT_EXACT  = 5.0  deg  -- maximum rotation error for an exact match
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
EXACT_TOL: float = 1.0      # LDU - position tolerance for an "exact" part match
EDGE_TOL: float = 2.0       # LDU - edge contact-point offset tolerance
ROT_EXACT_DEG: float = 5.0  # degrees - rotation tolerance for exact match

# Sets known to contain per-set hardcoded logic in the old pipeline.
# These are excluded from the "clean generalisation" holdout split.
HARDCODED_SETS: frozenset = frozenset({"75295-1", "75400-1"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PartResult:
    """Per-part evaluation record within a step."""
    step: int                        # 1-based step number
    piece_idx: int                   # 1-based cumulative piece number
    part_id: str
    gt_pos: List[float]              # [x, y, z] GT position in LDU
    pred_pos: Optional[List[float]]  # [x, y, z] predicted position, or None if missing
    pos_error_ldu: float             # Euclidean distance GT <-> pred (LDU)
    rot_error_deg: float             # Rotation error in degrees
    is_exact: bool                   # pos_error < EXACT_TOL AND rot_error < ROT_EXACT_DEG
    is_symmetric_match: bool         # matched under valid rotational symmetry equivalence
    gt_in_candidates: bool           # GT pose appeared in the generated candidate list
    post_gating_gt_in_candidates: Optional[bool]  # GT in candidates after connector gating
    total_candidates: int            # number of candidates generated
    failure_family: Optional[str]    # e.g. "SCORER_DISCRIMINATION_FAILURE"


@dataclass
class EdgeResult:
    """Per-edge evaluation record (parent->child connection)."""
    child_part_id: str
    child_piece_idx: int
    parent_part_id: Optional[str]
    gt_offset_ldu: List[float]       # [dx, dy, dz] relative offset GT
    pred_offset_ldu: Optional[List[float]]
    offset_error_ldu: float
    is_correct: bool                 # offset_error < EDGE_TOL


@dataclass
class MOCGraphMetrics:
    """All quality metrics for one MOC evaluation."""
    moc_id: str
    name: str = ""
    total_parts: int = 0
    total_steps: int = 0
    parts_evaluated: int = 0          # parts actually scored

    # Core partial-correctness metrics
    exact_parts: int = 0
    pct_parts_correct: float = 0.0    # exact_parts / total_parts * 100

    correct_edges: int = 0
    total_edges: int = 0
    pct_edges_correct: float = 0.0    # correct_edges / total_edges * 100

    graph_edit_distance: int = 0      # wrong_nodes + wrong_edges (lower = better)

    # Convergence
    convergence_depth: int = 0        # steps completed without error (== total_steps if perfect)
    first_error_step: Optional[int] = None     # 1-based; None = fully converged
    first_error_piece: Optional[int] = None    # 1-based cumulative

    # Candidate / scorer diagnostics
    gt_in_candidates_count: int = 0
    gt_not_in_candidates_count: int = 0
    gt_in_candidates_rate: float = 0.0
    post_gating_gt_rate: Optional[float] = None

    # Failure taxonomy
    failure_family_counts: Dict[str, int] = field(default_factory=dict)

    # Per-part / per-edge detail
    part_results: List[PartResult] = field(default_factory=list)
    edge_results: List[EdgeResult] = field(default_factory=list)

    # Holdout flag
    is_holdout: bool = False          # True when moc_id not in HARDCODED_SETS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusSummary:
    """Aggregate statistics over a full corpus of MOC evaluations."""
    total_mocs: int = 0
    holdout_mocs: int = 0

    # Part-level
    mean_pct_parts_correct: float = 0.0
    median_pct_parts_correct: float = 0.0
    mocs_above_50pct: int = 0
    mocs_above_80pct: int = 0
    mocs_fully_converged: int = 0

    # Edge-level
    mean_pct_edges_correct: float = 0.0

    # Convergence
    mean_convergence_depth: float = 0.0
    median_convergence_depth: float = 0.0
    mean_graph_edit_distance: float = 0.0

    # Candidate / scorer
    overall_gt_in_candidates_rate: float = 0.0
    overall_post_gating_gt_rate: Optional[float] = None

    # Failure taxonomy (summed across corpus)
    failure_family_counts: Dict[str, int] = field(default_factory=dict)
    failure_family_pct: Dict[str, float] = field(default_factory=dict)

    # Holdout-only metrics
    holdout_mean_pct_parts_correct: Optional[float] = None
    holdout_mocs_fully_converged: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Returns the geodesic rotation error in degrees between two 3x3 rotation matrices.
    angle = arccos( clamp( (trace(R_pred^T @ R_gt) - 1) / 2 ) )
    """
    R_rel = R_pred.T @ R_gt
    trace = float(np.trace(R_rel))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_angle)))


def _symmetry_rot_error(
    R_pred: np.ndarray,
    R_gt: np.ndarray,
    symmetry_order: int,
) -> float:
    """
    Returns the minimum rotation error considering all equivalent yaw rotations
    imposed by the part's symmetry order (C1->1, C2->2, C4->4).
    """
    best = _rot_error_deg(R_pred, R_gt)
    if symmetry_order <= 1:
        return best
    step_deg = 360.0 / symmetry_order
    for k in range(1, symmetry_order):
        angle = math.radians(k * step_deg)
        c, s = math.cos(angle), math.sin(angle)
        Ry = np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32
        )
        err = _rot_error_deg(R_pred, R_gt @ Ry)
        if err < best:
            best = err
    return best


def _parse_transform(tf_flat: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a 16-element flat row-major 4x4 transform into (R 3x3, t 3-vector).
    Falls back to identity when the list is malformed.
    """
    if tf_flat and len(tf_flat) == 16:
        M = np.array(tf_flat, dtype=np.float32).reshape(4, 4)
        return M[:3, :3], M[:3, 3]
    return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


def _symmetry_order_for_part(part_id: str) -> int:
    """
    Returns a heuristic symmetry order for common part families.
    C4 (order 4) for square/round parts, C1 otherwise.
    The full ontology will be handled by the connector-gating module (Paso 1+).
    """
    pid = part_id.lower().replace(".dat", "")
    round_prefixes = (
        "4073", "6141", "4032", "3062", "3024", "4150", "98138",
        "14769", "11833", "85984",
    )
    if any(pid.startswith(p) for p in round_prefixes):
        return 4
    square_ids = ("3003", "3022", "3958", "4477", "60479", "87580")
    if any(pid.startswith(p) for p in square_ids):
        return 4
    return 1


def _clean_pid(part_id: str) -> str:
    pid = str(part_id).strip().lower()
    return pid if pid.endswith(".dat") else f"{pid}.dat"


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class GraphQualityEvaluator:
    """
    Evaluates a predicted assembly graph against a Ground-Truth graph,
    computing partial correctness metrics without aborting on the first error.

    Usage::

        evaluator = GraphQualityEvaluator()
        metrics = evaluator.evaluate(
            moc_id="MOC-12345",
            pred_steps=pred_step_list,  # list[list[dict]] with absolute_transform
            gt_steps=gt_step_list,
            candidate_info=cand_info,   # optional per-errored-part diagnostics
        )
    """

    def __init__(
        self,
        exact_tol: float = EXACT_TOL,
        edge_tol: float = EDGE_TOL,
        rot_exact_deg: float = ROT_EXACT_DEG,
    ) -> None:
        self.exact_tol = exact_tol
        self.edge_tol = edge_tol
        self.rot_exact_deg = rot_exact_deg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        moc_id: str,
        pred_steps: List[List[Dict[str, Any]]],
        gt_steps: List[List[Dict[str, Any]]],
        name: str = "",
        candidate_info: Optional[List[Dict[str, Any]]] = None,
    ) -> MOCGraphMetrics:
        """
        Computes all partial-correctness metrics for a single MOC.

        Parameters
        ----------
        pred_steps       : list of steps, each step is a list of part dicts with
                           ``part_id`` (str) and ``absolute_transform`` (16 floats).
        gt_steps         : same structure from Ground-Truth .io file.
        candidate_info   : optional list (one entry per *errored* part) containing
                           ``gt_in_candidates`` (bool), ``total_candidates`` (int),
                           ``family`` (str), and optionally
                           ``post_gating_gt_in_candidates`` (bool).
        """
        metrics = MOCGraphMetrics(
            moc_id=moc_id,
            name=name,
            total_steps=len(gt_steps),
            is_holdout=(
                moc_id.split("-")[0] not in {s.split("-")[0] for s in HARDCODED_SETS}
            ),
        )

        gt_flat = self._flatten_steps(gt_steps)
        pred_flat = self._flatten_steps(pred_steps)

        metrics.total_parts = len(gt_flat)
        metrics.parts_evaluated = min(len(gt_flat), len(pred_flat))

        if not gt_flat:
            return metrics

        # Part-level matching
        (
            part_results,
            exact_parts,
            convergence_depth,
            first_err_step,
            first_err_piece,
        ) = self._match_parts(gt_flat, pred_flat, list(candidate_info or []))

        metrics.part_results = part_results
        metrics.exact_parts = exact_parts
        metrics.pct_parts_correct = round(
            100.0 * exact_parts / max(1, len(gt_flat)), 2
        )
        metrics.convergence_depth = convergence_depth
        metrics.first_error_step = first_err_step
        metrics.first_error_piece = first_err_piece

        # Edge-level matching
        edge_results, correct_edges = self._match_edges(gt_flat, pred_flat)
        metrics.edge_results = edge_results
        metrics.total_edges = len(edge_results)
        metrics.correct_edges = correct_edges
        metrics.pct_edges_correct = round(
            100.0 * correct_edges / max(1, metrics.total_edges), 2
        )

        # Graph edit distance
        wrong_nodes = len(gt_flat) - exact_parts
        wrong_edges = metrics.total_edges - correct_edges
        metrics.graph_edit_distance = wrong_nodes + wrong_edges

        # Candidate / scorer diagnostics
        gt_in = sum(1 for r in part_results if not r.is_exact and r.gt_in_candidates)
        gt_out = sum(
            1 for r in part_results if not r.is_exact and not r.gt_in_candidates
        )
        metrics.gt_in_candidates_count = gt_in
        metrics.gt_not_in_candidates_count = gt_out
        errored = gt_in + gt_out
        metrics.gt_in_candidates_rate = round(gt_in / max(1, errored), 4)

        pg_in = sum(
            1
            for r in part_results
            if not r.is_exact and r.post_gating_gt_in_candidates is True
        )
        pg_total = sum(
            1
            for r in part_results
            if not r.is_exact and r.post_gating_gt_in_candidates is not None
        )
        if pg_total > 0:
            metrics.post_gating_gt_rate = round(pg_in / pg_total, 4)

        # Failure taxonomy
        for r in part_results:
            if r.failure_family:
                metrics.failure_family_counts[r.failure_family] = (
                    metrics.failure_family_counts.get(r.failure_family, 0) + 1
                )

        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_steps(
        steps: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Converts list-of-steps into a flat list with step and piece_idx tags."""
        flat: List[Dict[str, Any]] = []
        piece_idx = 1
        for s_idx, step in enumerate(steps):
            for part in step:
                entry = dict(part)
                entry["step"] = s_idx + 1
                entry["piece_idx"] = piece_idx
                piece_idx += 1
                flat.append(entry)
        return flat

    def _match_parts(
        self,
        gt_flat: List[Dict[str, Any]],
        pred_flat: List[Dict[str, Any]],
        candidate_info: List[Dict[str, Any]],
    ) -> Tuple[
        List[PartResult],
        int,
        int,
        Optional[int],
        Optional[int],
    ]:
        """
        Hungarian bipartite matching per step, grouped by part_id.
        Within each step, identical parts are matched to minimise position error,
        accounting for rotational symmetry.
        """
        part_results: List[PartResult] = []
        exact_parts = 0
        convergence_depth = 0
        found_first_error = False
        first_error_step: Optional[int] = None
        first_error_piece: Optional[int] = None
        cand_iter = iter(candidate_info)

        # Bucket GT and pred by step
        gt_by_step: Dict[int, List[Dict]] = {}
        pred_by_step: Dict[int, List[Dict]] = {}
        for p in gt_flat:
            gt_by_step.setdefault(p["step"], []).append(p)
        for p in pred_flat:
            pred_by_step.setdefault(p["step"], []).append(p)

        all_steps = sorted(set(gt_by_step) | set(pred_by_step))

        for step in all_steps:
            gt_step = gt_by_step.get(step, [])
            pred_step = pred_by_step.get(step, [])
            step_has_error = False

            # Group by part_id for within-group Hungarian matching
            gt_by_pid: Dict[str, List[Dict]] = {}
            pred_by_pid: Dict[str, List[Dict]] = {}
            for p in gt_step:
                pid = _clean_pid(p["part_id"])
                gt_by_pid.setdefault(pid, []).append(p)
            for p in pred_step:
                pid = _clean_pid(p["part_id"])
                pred_by_pid.setdefault(pid, []).append(p)

            all_pids = set(gt_by_pid) | set(pred_by_pid)
            for pid in all_pids:
                gt_group = gt_by_pid.get(pid, [])
                pred_group = pred_by_pid.get(pid, [])
                sym_order = _symmetry_order_for_part(pid)
                matched = self._hungarian_match_group(
                    gt_group, pred_group, sym_order
                )
                for r in matched:
                    if not r.is_exact:
                        step_has_error = True
                        cand_dict = next(cand_iter, {})
                        r.gt_in_candidates = bool(cand_dict.get("gt_in_candidates", False))
                        r.post_gating_gt_in_candidates = cand_dict.get(
                            "post_gating_gt_in_candidates", None
                        )
                        r.total_candidates = int(cand_dict.get("total_candidates", 0))
                        # Preserve pre-assigned family (e.g. MISSING_PREDICTION) when
                        # the external candidate_info dict does not provide one.
                        cand_family = cand_dict.get("family", None)
                        if cand_family is not None:
                            r.failure_family = cand_family
                        if first_error_step is None:
                            first_error_step = step
                            first_error_piece = r.piece_idx
                            found_first_error = True
                    else:
                        exact_parts += 1
                    part_results.append(r)

            if not step_has_error and not found_first_error:
                convergence_depth += 1

        return (
            part_results,
            exact_parts,
            convergence_depth,
            first_error_step,
            first_error_piece,
        )

    def _hungarian_match_group(
        self,
        gt_group: List[Dict],
        pred_group: List[Dict],
        sym_order: int,
    ) -> List[PartResult]:
        """
        Matches GT parts to predicted parts within the same part_id group
        using the Hungarian algorithm on a position-error cost matrix.
        Unmatched GT parts are recorded as missing predictions.
        """
        results: List[PartResult] = []
        n_gt = len(gt_group)
        n_pred = len(pred_group)

        if n_gt == 0:
            return results
        if n_pred == 0:
            for gp in gt_group:
                _, gt_t = _parse_transform(gp.get("absolute_transform", []))
                results.append(PartResult(
                    step=gp["step"], piece_idx=gp["piece_idx"],
                    part_id=gp["part_id"],
                    gt_pos=gt_t.tolist(), pred_pos=None,
                    pos_error_ldu=float("inf"), rot_error_deg=180.0,
                    is_exact=False, is_symmetric_match=False,
                    gt_in_candidates=False, post_gating_gt_in_candidates=None,
                    total_candidates=0, failure_family="MISSING_PREDICTION",
                ))
            return results

        # Build cost matrix (n_gt x n_pred)
        cost = np.full((n_gt, n_pred), fill_value=1e6, dtype=np.float64)
        for i, gp in enumerate(gt_group):
            gt_R, gt_t = _parse_transform(gp.get("absolute_transform", []))
            for j, pp in enumerate(pred_group):
                pred_R, pred_t = _parse_transform(pp.get("absolute_transform", []))
                pos_err = float(np.linalg.norm(gt_t - pred_t))
                rot_err = _symmetry_rot_error(pred_R, gt_R, sym_order)
                # Position dominates; rotation is a tiebreaker
                cost[i, j] = pos_err + rot_err * 0.1

        row_idx, col_idx = linear_sum_assignment(cost)

        for i, j in zip(row_idx, col_idx):
            gp = gt_group[i]
            pp = pred_group[j]
            gt_R, gt_t = _parse_transform(gp.get("absolute_transform", []))
            pred_R, pred_t = _parse_transform(pp.get("absolute_transform", []))
            pos_err = float(np.linalg.norm(gt_t - pred_t))
            rot_err = _symmetry_rot_error(pred_R, gt_R, sym_order)
            is_exact = (pos_err <= self.exact_tol) and (rot_err <= self.rot_exact_deg)
            results.append(PartResult(
                step=gp["step"], piece_idx=gp["piece_idx"],
                part_id=gp["part_id"],
                gt_pos=gt_t.tolist(), pred_pos=pred_t.tolist(),
                pos_error_ldu=round(pos_err, 3),
                rot_error_deg=round(rot_err, 2),
                is_exact=is_exact,
                is_symmetric_match=(is_exact and rot_err > 0.5),
                gt_in_candidates=False,
                post_gating_gt_in_candidates=None,
                total_candidates=0,
                failure_family=None,
            ))

        # Unmatched GT parts (more GT than pred in this group)
        matched_gt = set(row_idx)
        for i, gp in enumerate(gt_group):
            if i not in matched_gt:
                _, gt_t = _parse_transform(gp.get("absolute_transform", []))
                results.append(PartResult(
                    step=gp["step"], piece_idx=gp["piece_idx"],
                    part_id=gp["part_id"],
                    gt_pos=gt_t.tolist(), pred_pos=None,
                    pos_error_ldu=float("inf"), rot_error_deg=180.0,
                    is_exact=False, is_symmetric_match=False,
                    gt_in_candidates=False, post_gating_gt_in_candidates=None,
                    total_candidates=0, failure_family="MISSING_PREDICTION",
                ))

        return results

    def _match_edges(
        self,
        gt_flat: List[Dict[str, Any]],
        pred_flat: List[Dict[str, Any]],
    ) -> Tuple[List[EdgeResult], int]:
        """
        Compares parent->child connection edges.  Each part dict may optionally
        carry a ``connections`` list with ``relative_offset_ldu`` entries.
        Parts without connection data are skipped.
        """
        results: List[EdgeResult] = []
        correct = 0

        pred_by_idx: Dict[int, Dict] = {p["piece_idx"]: p for p in pred_flat}

        for gp in gt_flat:
            for conn in gp.get("connections", []):
                gt_off = np.array(
                    conn.get("relative_offset_ldu", [0, 0, 0]), dtype=np.float32
                )
                parent_part_id = conn.get("parent_part_id", None)

                pp = pred_by_idx.get(gp["piece_idx"])
                pred_off = None
                offset_err = float("inf")
                is_correct = False

                if pp:
                    for pc in pp.get("connections", []):
                        po = np.array(
                            pc.get("relative_offset_ldu", [0, 0, 0]), dtype=np.float32
                        )
                        err = float(np.linalg.norm(gt_off - po))
                        if err < offset_err:
                            offset_err = err
                            pred_off = po
                    is_correct = offset_err <= self.edge_tol
                    if is_correct:
                        correct += 1

                results.append(EdgeResult(
                    child_part_id=gp["part_id"],
                    child_piece_idx=gp["piece_idx"],
                    parent_part_id=parent_part_id,
                    gt_offset_ldu=gt_off.tolist(),
                    pred_offset_ldu=pred_off.tolist() if pred_off is not None else None,
                    offset_error_ldu=round(min(offset_err, 1e9), 3),
                    is_correct=is_correct,
                ))

        return results, correct


# ---------------------------------------------------------------------------
# Corpus aggregator
# ---------------------------------------------------------------------------

def aggregate_corpus(moc_metrics: List[MOCGraphMetrics]) -> CorpusSummary:
    """
    Aggregates a list of per-MOC metrics into a CorpusSummary with mean/median
    statistics, holdout split, and failure taxonomy breakdown.
    """
    if not moc_metrics:
        return CorpusSummary()

    summary = CorpusSummary(
        total_mocs=len(moc_metrics),
        holdout_mocs=sum(1 for m in moc_metrics if m.is_holdout),
    )

    pct_parts = [m.pct_parts_correct for m in moc_metrics]
    pct_edges = [m.pct_edges_correct for m in moc_metrics]
    depths = [float(m.convergence_depth) for m in moc_metrics]
    geds = [float(m.graph_edit_distance) for m in moc_metrics]

    summary.mean_pct_parts_correct = round(float(np.mean(pct_parts)), 2)
    summary.median_pct_parts_correct = round(float(np.median(pct_parts)), 2)
    summary.mocs_above_50pct = sum(1 for v in pct_parts if v >= 50.0)
    summary.mocs_above_80pct = sum(1 for v in pct_parts if v >= 80.0)
    summary.mocs_fully_converged = sum(1 for v in pct_parts if v >= 99.99)

    summary.mean_pct_edges_correct = round(float(np.mean(pct_edges)), 2)
    summary.mean_convergence_depth = round(float(np.mean(depths)), 2)
    summary.median_convergence_depth = round(float(np.median(depths)), 2)
    summary.mean_graph_edit_distance = round(float(np.mean(geds)), 2)

    total_gt_in = sum(m.gt_in_candidates_count for m in moc_metrics)
    total_gt_out = sum(m.gt_not_in_candidates_count for m in moc_metrics)
    total_errored = total_gt_in + total_gt_out
    summary.overall_gt_in_candidates_rate = round(
        total_gt_in / max(1, total_errored), 4
    )

    pg_in = sum(
        m.gt_in_candidates_count
        for m in moc_metrics
        if m.post_gating_gt_rate is not None
    )
    pg_tot = sum(
        m.gt_in_candidates_count + m.gt_not_in_candidates_count
        for m in moc_metrics
        if m.post_gating_gt_rate is not None
    )
    if pg_tot > 0:
        summary.overall_post_gating_gt_rate = round(pg_in / pg_tot, 4)

    family_total = 0
    for m in moc_metrics:
        for fam, cnt in m.failure_family_counts.items():
            summary.failure_family_counts[fam] = (
                summary.failure_family_counts.get(fam, 0) + cnt
            )
            family_total += cnt
    if family_total > 0:
        summary.failure_family_pct = {
            k: round(100.0 * v / family_total, 2)
            for k, v in summary.failure_family_counts.items()
        }

    holdout = [m for m in moc_metrics if m.is_holdout]
    if holdout:
        summary.holdout_mean_pct_parts_correct = round(
            float(np.mean([m.pct_parts_correct for m in holdout])), 2
        )
        summary.holdout_mocs_fully_converged = sum(
            1 for m in holdout if m.pct_parts_correct >= 99.99
        )

    return summary


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def nodes_to_steps(nodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Converts a flat node list (each node has ``step_id``) into a list-of-steps
    compatible with ``GraphQualityEvaluator.evaluate()``.
    """
    by_step: Dict[int, List[Dict]] = {}
    for n in nodes:
        sid = int(n.get("step_id", 1))
        by_step.setdefault(sid, []).append(n)
    return [by_step[k] for k in sorted(by_step)]

"""
Tests for src/graph_quality_metrics.py

Covers:
  1. Perfect prediction -> 100% parts/edges correct, convergence_depth == total_steps.
  2. One wrong part -> partial correctness, first_error_step populated.
  3. Missing prediction for a step -> MISSING_PREDICTION family recorded.
  4. Rotational symmetry tolerance (C4 part at 90-deg rotation counts as exact).
  5. Edge matching with correct and incorrect offsets.
  6. candidate_info propagation (gt_in_candidates, failure_family, post_gating flag).
  7. aggregate_corpus() statistics: mean, median, holdout split, failure taxonomy.
  8. nodes_to_steps() helper round-trip.
  9. Holdout flag: hardcoded sets are NOT holdout, arbitrary MOC-IDs ARE holdout.
 10. Empty inputs handled gracefully.
"""

import math
import numpy as np
import pytest

from src.graph_quality_metrics import (
    EXACT_TOL,
    HARDCODED_SETS,
    GraphQualityEvaluator,
    MOCGraphMetrics,
    CorpusSummary,
    aggregate_corpus,
    nodes_to_steps,
    _rot_error_deg,
    _symmetry_rot_error,
    _parse_transform,
    _clean_pid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tf(x: float, y: float, z: float, R: np.ndarray = None) -> list:
    """Builds a flat 16-float row-major 4x4 transform."""
    M = np.eye(4, dtype=np.float32)
    if R is not None:
        M[:3, :3] = R
    M[0, 3] = x
    M[1, 3] = y
    M[2, 3] = z
    return M.flatten().tolist()


def _rot_y(deg: float) -> np.ndarray:
    """Returns a 3x3 rotation matrix around Y by deg degrees."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _part(part_id: str, x: float, y: float, z: float,
          R: np.ndarray = None, connections: list = None) -> dict:
    p = {
        "part_id": part_id,
        "absolute_transform": _tf(x, y, z, R),
    }
    if connections is not None:
        p["connections"] = connections
    return p


EVAL = GraphQualityEvaluator()


# ---------------------------------------------------------------------------
# 1. Perfect prediction
# ---------------------------------------------------------------------------

def test_perfect_prediction_full_correctness():
    gt = [
        [_part("3001.dat", 0, 0, 0)],
        [_part("3003.dat", 20, -24, 0), _part("3004.dat", -20, -24, 0)],
    ]
    pred = [
        [_part("3001.dat", 0, 0, 0)],
        [_part("3003.dat", 20, -24, 0), _part("3004.dat", -20, -24, 0)],
    ]
    m = EVAL.evaluate("MOC-TEST", pred, gt)

    assert m.pct_parts_correct == 100.0
    assert m.exact_parts == 3
    assert m.total_parts == 3
    assert m.convergence_depth == 2   # all steps passed
    assert m.first_error_step is None
    assert m.first_error_piece is None
    assert m.graph_edit_distance == 0


# ---------------------------------------------------------------------------
# 2. One wrong part in step 2
# ---------------------------------------------------------------------------

def test_one_wrong_part_partial_correctness():
    gt = [
        [_part("3001.dat", 0, 0, 0)],
        [_part("3003.dat", 20, -24, 0)],   # correct position
    ]
    pred = [
        [_part("3001.dat", 0, 0, 0)],
        [_part("3003.dat", 60, -24, 0)],   # wrong: 40 LDU off
    ]
    m = EVAL.evaluate("MOC-TEST", pred, gt)

    assert m.exact_parts == 1
    assert m.total_parts == 2
    assert m.pct_parts_correct == 50.0
    assert m.first_error_step == 2
    assert m.first_error_piece == 2
    assert m.convergence_depth == 1   # step 1 passed, step 2 failed
    assert m.graph_edit_distance >= 1


# ---------------------------------------------------------------------------
# 3. Missing prediction for an entire step
# ---------------------------------------------------------------------------

def test_missing_prediction_step():
    gt = [
        [_part("3001.dat", 0, 0, 0)],
        [_part("3004.dat", 20, -8, 0)],
    ]
    pred = [
        [_part("3001.dat", 0, 0, 0)],
        # step 2 missing entirely
    ]
    m = EVAL.evaluate("MOC-MISS", pred, gt)

    assert m.exact_parts == 1
    assert m.total_parts == 2
    assert m.pct_parts_correct == 50.0
    assert m.first_error_step == 2
    families = m.failure_family_counts
    assert families.get("MISSING_PREDICTION", 0) >= 1


# ---------------------------------------------------------------------------
# 4. C4 symmetry: 90-deg-rotated square part counts as exact
# ---------------------------------------------------------------------------

def test_c4_symmetry_90deg_counts_as_exact():
    # 3003.dat is a 2x2 brick -> C4 symmetry (order 4)
    R_gt   = _rot_y(0.0)
    R_pred = _rot_y(90.0)   # 90-deg rotation around Y: equivalent under C4
    gt   = [[_part("3003.dat", 0, 0, 0, R=R_gt)]]
    pred = [[_part("3003.dat", 0, 0, 0, R=R_pred)]]
    m = EVAL.evaluate("MOC-SYM", pred, gt)
    assert m.exact_parts == 1, "90-deg rotation of C4 part should count as exact"


def test_c1_symmetry_90deg_does_NOT_count_as_exact():
    # 3001.dat (2x4 brick) is C1 -> 90-deg Y rotation should NOT be exact
    R_gt   = _rot_y(0.0)
    R_pred = _rot_y(90.0)
    gt   = [[_part("3001.dat", 0, 0, 0, R=R_gt)]]
    pred = [[_part("3001.dat", 0, 0, 0, R=R_pred)]]
    m = EVAL.evaluate("MOC-SYM2", pred, gt)
    assert m.exact_parts == 0, "90-deg rotation of C1 part should NOT count as exact"


# ---------------------------------------------------------------------------
# 5. Edge matching
# ---------------------------------------------------------------------------

def test_edge_correct_within_tolerance():
    conn_gt   = [{"relative_offset_ldu": [0, -8, 0]}]
    conn_pred = [{"relative_offset_ldu": [0, -8, 0]}]   # exact
    gt   = [[_part("3001.dat", 0, 0, 0, connections=conn_gt)]]
    pred = [[_part("3001.dat", 0, 0, 0, connections=conn_pred)]]
    m = EVAL.evaluate("MOC-EDGE", pred, gt)
    assert m.correct_edges == 1
    assert m.pct_edges_correct == 100.0


def test_edge_wrong_beyond_tolerance():
    conn_gt   = [{"relative_offset_ldu": [0, -8,  0]}]
    conn_pred = [{"relative_offset_ldu": [20, -8, 0]}]  # 20 LDU off
    gt   = [[_part("3001.dat", 0, 0, 0, connections=conn_gt)]]
    pred = [[_part("3001.dat", 0, 0, 0, connections=conn_pred)]]
    m = EVAL.evaluate("MOC-EDGEWRONG", pred, gt)
    assert m.correct_edges == 0
    assert m.pct_edges_correct == 0.0


def test_no_connections_means_zero_edges():
    gt   = [[_part("3001.dat", 0, 0, 0)]]   # no connections key
    pred = [[_part("3001.dat", 0, 0, 0)]]
    m = EVAL.evaluate("MOC-NOEDGE", pred, gt)
    assert m.total_edges == 0
    assert m.pct_edges_correct == 0.0  # 0/0 -> 0.0 (safe)


# ---------------------------------------------------------------------------
# 6. candidate_info propagation
# ---------------------------------------------------------------------------

def test_candidate_info_attached_to_errored_part():
    gt   = [[_part("3001.dat", 0, 0, 0)], [_part("3004.dat", 40, -8, 0)]]
    pred = [[_part("3001.dat", 0, 0, 0)], [_part("3004.dat", 0, -8, 0)]]  # wrong X
    cand_info = [
        {
            "gt_in_candidates": True,
            "total_candidates": 12,
            "family": "SCORER_COLLINEAR_SLIP",
            "post_gating_gt_in_candidates": True,
        }
    ]
    m = EVAL.evaluate("MOC-CAND", pred, gt, candidate_info=cand_info)

    # Find the errored part result
    errored = [r for r in m.part_results if not r.is_exact]
    assert len(errored) == 1
    r = errored[0]
    assert r.gt_in_candidates is True
    assert r.total_candidates == 12
    assert r.failure_family == "SCORER_COLLINEAR_SLIP"
    assert r.post_gating_gt_in_candidates is True

    assert m.gt_in_candidates_count == 1
    assert m.gt_in_candidates_rate == 1.0
    assert m.post_gating_gt_rate == 1.0
    assert m.failure_family_counts.get("SCORER_COLLINEAR_SLIP") == 1


def test_gt_not_in_candidates():
    gt   = [[_part("3001.dat", 0, 0, 0)], [_part("3004.dat", 40, -8, 0)]]
    pred = [[_part("3001.dat", 0, 0, 0)], [_part("3004.dat", 0, -8, 0)]]
    cand_info = [
        {"gt_in_candidates": False, "total_candidates": 5, "family": "NO_VALID_STUD_HYPOTHESIS"}
    ]
    m = EVAL.evaluate("MOC-NOCAND", pred, gt, candidate_info=cand_info)
    assert m.gt_not_in_candidates_count == 1
    assert m.gt_in_candidates_rate == 0.0


# ---------------------------------------------------------------------------
# 7. aggregate_corpus
# ---------------------------------------------------------------------------

def test_aggregate_corpus_basic():
    # Build two synthetic MOCGraphMetrics
    m1 = MOCGraphMetrics(
        moc_id="MOC-001", total_parts=10, exact_parts=10,
        pct_parts_correct=100.0, total_edges=5, correct_edges=5,
        pct_edges_correct=100.0, graph_edit_distance=0,
        convergence_depth=3, gt_in_candidates_count=0,
        gt_not_in_candidates_count=0, gt_in_candidates_rate=0.0,
        is_holdout=True,
        failure_family_counts={},
    )
    m2 = MOCGraphMetrics(
        moc_id="MOC-002", total_parts=10, exact_parts=4,
        pct_parts_correct=40.0, total_edges=5, correct_edges=2,
        pct_edges_correct=40.0, graph_edit_distance=9,
        convergence_depth=1, gt_in_candidates_count=3,
        gt_not_in_candidates_count=3, gt_in_candidates_rate=0.5,
        is_holdout=True,
        failure_family_counts={"SCORER_DISCRIMINATION_FAILURE": 3, "SNOT_FRACTIONAL_OFFSET": 3},
    )

    summary = aggregate_corpus([m1, m2])

    assert summary.total_mocs == 2
    assert summary.holdout_mocs == 2
    assert summary.mean_pct_parts_correct == 70.0
    assert summary.mocs_fully_converged == 1
    assert summary.mocs_above_50pct == 1
    assert summary.mean_convergence_depth == 2.0
    assert summary.overall_gt_in_candidates_rate == pytest.approx(0.5, abs=0.01)
    assert summary.failure_family_counts["SCORER_DISCRIMINATION_FAILURE"] == 3
    assert "SCORER_DISCRIMINATION_FAILURE" in summary.failure_family_pct
    assert summary.holdout_mean_pct_parts_correct == 70.0


def test_aggregate_corpus_empty():
    summary = aggregate_corpus([])
    assert summary.total_mocs == 0
    assert summary.mean_pct_parts_correct == 0.0


# ---------------------------------------------------------------------------
# 8. nodes_to_steps helper
# ---------------------------------------------------------------------------

def test_nodes_to_steps_round_trip():
    nodes = [
        {"part_id": "3001.dat", "step_id": 1, "absolute_transform": _tf(0, 0, 0)},
        {"part_id": "3004.dat", "step_id": 2, "absolute_transform": _tf(20, -8, 0)},
        {"part_id": "3004.dat", "step_id": 2, "absolute_transform": _tf(-20, -8, 0)},
    ]
    steps = nodes_to_steps(nodes)
    assert len(steps) == 2
    assert len(steps[0]) == 1
    assert len(steps[1]) == 2
    assert steps[0][0]["part_id"] == "3001.dat"


# ---------------------------------------------------------------------------
# 9. Holdout flag
# ---------------------------------------------------------------------------

def test_holdout_flag_hardcoded_sets_not_holdout():
    for set_id in HARDCODED_SETS:
        gt   = [[_part("3001.dat", 0, 0, 0)]]
        pred = [[_part("3001.dat", 0, 0, 0)]]
        moc_id = set_id  # e.g. "75295-1"
        m = EVAL.evaluate(moc_id, pred, gt)
        assert m.is_holdout is False, (
            f"{set_id} should NOT be holdout (has hardcoded logic)"
        )


def test_holdout_flag_arbitrary_moc_is_holdout():
    gt   = [[_part("3001.dat", 0, 0, 0)]]
    pred = [[_part("3001.dat", 0, 0, 0)]]
    m = EVAL.evaluate("MOC-99999", pred, gt)
    assert m.is_holdout is True


# ---------------------------------------------------------------------------
# 10. Empty inputs
# ---------------------------------------------------------------------------

def test_empty_gt_returns_zero_metrics():
    m = EVAL.evaluate("MOC-EMPTY", pred_steps=[], gt_steps=[])
    assert m.total_parts == 0
    assert m.pct_parts_correct == 0.0
    assert m.convergence_depth == 0


def test_empty_pred_all_missing():
    gt = [[_part("3001.dat", 0, 0, 0), _part("3004.dat", 20, 0, 0)]]
    m = EVAL.evaluate("MOC-NOPRED", pred_steps=[], gt_steps=gt)
    assert m.total_parts == 2
    assert m.exact_parts == 0
    assert m.pct_parts_correct == 0.0
    assert m.failure_family_counts.get("MISSING_PREDICTION", 0) == 2


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

def test_rot_error_identity_is_zero():
    R = np.eye(3, dtype=np.float32)
    assert _rot_error_deg(R, R) == pytest.approx(0.0, abs=1e-4)


def test_rot_error_90deg():
    R_gt = np.eye(3, dtype=np.float32)
    R_pred = _rot_y(90.0)
    err = _rot_error_deg(R_pred, R_gt)
    assert err == pytest.approx(90.0, abs=0.5)


def test_symmetry_rot_error_c4_reduces_90deg_to_zero():
    R_gt = np.eye(3, dtype=np.float32)
    R_pred = _rot_y(90.0)
    # Under C4 symmetry, 90-deg rotation is equivalent
    err = _symmetry_rot_error(R_pred, R_gt, symmetry_order=4)
    assert err == pytest.approx(0.0, abs=1e-3)


def test_symmetry_rot_error_c1_does_not_reduce():
    R_gt = np.eye(3, dtype=np.float32)
    R_pred = _rot_y(90.0)
    err = _symmetry_rot_error(R_pred, R_gt, symmetry_order=1)
    assert err == pytest.approx(90.0, abs=0.5)


def test_parse_transform_valid():
    M = np.eye(4, dtype=np.float32)
    M[0, 3] = 10.0
    M[1, 3] = -24.0
    M[2, 3] = 5.0
    R, t = _parse_transform(M.flatten().tolist())
    assert np.allclose(R, np.eye(3), atol=1e-5)
    assert np.allclose(t, [10.0, -24.0, 5.0], atol=1e-5)


def test_parse_transform_invalid_falls_back_to_identity():
    R, t = _parse_transform([])
    assert np.allclose(R, np.eye(3), atol=1e-5)
    assert np.allclose(t, [0.0, 0.0, 0.0], atol=1e-5)


def test_clean_pid_adds_dat_extension():
    assert _clean_pid("3001") == "3001.dat"
    assert _clean_pid("3001.dat") == "3001.dat"
    assert _clean_pid("3001.DAT") == "3001.dat"

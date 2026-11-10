import os
import json
import numpy as np
import pytest

from src.parser import ParsedPart
from src.io_parser import export_parts_to_io
from src.oracle_step_debugger import (
    OracleStepDebugger,
    PatternClusterAnalyzer,
    GeneralizedFixBundle,
)
from scripts.run_moc_continuous_improvement_loop import run_continuous_improvement_loop


def _make_tf(x: float, y: float, z: float) -> np.ndarray:
    tf = np.eye(4, dtype=np.float32)
    tf[0, 3] = x
    tf[1, 3] = y
    tf[2, 3] = z
    return tf


def _build_sample_stepped_io(io_path: str) -> str:
    """
    Creates a 3-step physically connected LEGO assembly in .io format:
      Step 1: 3001.dat (2x4 brick) at (0, 0, 0)
      Step 2: 3003.dat (2x2 brick) at (0, -24, 0) on top of Step 1
      Step 3: 3005.dat (1x1 brick) at (10, -48, 10) on top of Step 2
    """
    parts = [
        ParsedPart("3001.dat", 4, _make_tf(0.0, 0.0, 0.0), step_id=1),
        ParsedPart("3003.dat", 14, _make_tf(0.0, -24.0, 0.0), step_id=2),
        ParsedPart("3005.dat", 1, _make_tf(10.0, -48.0, 10.0), step_id=3),
    ]
    return export_parts_to_io(parts, io_path, model_title="Oracle Test MOC")


def test_fail_fast_abort_and_root_cause_taxonomy(tmp_path):
    """
    Verifies that OracleStepDebugger:
      1. Aborts immediately at the first positioning error (Step 2) so Step 3 is never
         corrupted by cascading errors.
      2. Correctly classifies INVENTORY_OCR_MISMATCH, SUBSTRATE_PARENT_FILTERED,
         and VISUAL_SCORER_RANKING_ERROR.
    """
    io_path = str(tmp_path / "moc_test.io")
    _build_sample_stepped_io(io_path)

    debugger = OracleStepDebugger()

    # Case A: Clean run with no perturbations passes 100% of steps
    res_clean = debugger.evaluate_moc_fail_fast(
        moc_id="MOC-CLEAN", io_path=io_path, abort_on_first_error=True
    )
    assert res_clean["completed_all_steps"] is True
    assert res_clean["steps_survived"] == 3
    assert len(res_clean["incidents"]) == 0

    # Case B: Inventory OCR mismatch at Step 2 -> must abort at Step 2
    res_ocr = debugger.evaluate_moc_fail_fast(
        moc_id="MOC-OCR",
        io_path=io_path,
        abort_on_first_error=True,
        simulated_step_corruptions={
            2: {"ocr_step_parts": [("3002.dat", 14)]}  # Wrong part 2x3 instead of 2x2
        },
    )
    assert res_ocr["completed_all_steps"] is False
    assert res_ocr["aborted_at_step"] == 2
    assert res_ocr["steps_survived"] == 1
    assert len(res_ocr["incidents"]) == 1  # Aborted before Step 3!
    assert res_ocr["first_error_category"] == "INVENTORY_OCR_MISMATCH"

    # Case C: Substrate parent filter prunes parent 0 at Step 2 -> HYPOTHESIS_NOT_GENERATED :: SUBSTRATE_PARENT_FILTERED
    res_sub = debugger.evaluate_moc_fail_fast(
        moc_id="MOC-SUBSTRATE",
        io_path=io_path,
        abort_on_first_error=True,
        simulated_step_corruptions={
            2: {"substrate_parent_indices": [99]}  # Excludes actual parent index 0
        },
    )
    assert res_sub["completed_all_steps"] is False
    assert res_sub["aborted_at_step"] == 2
    assert res_sub["first_error_category"] == "HYPOTHESIS_NOT_GENERATED"
    assert res_sub["first_error_sub_cause"] == "SUBSTRATE_PARENT_FILTERED"


def test_closed_loop_self_improvement_across_moc_cohort(tmp_path):
    """
    Verifies that the continuous improvement loop:
      1. Detects recurring error patterns across a cohort of MOCs in Iteration 1.
      2. Synthesizes a GeneralizedFixBundle (recalibrating VisualPoseScorer weights
         and disabling over-restrictive substrate filtering).
      3. Automatically resolves the failures in Iteration 2 (raising pass rate to 100%).
    """
    io_path_1 = str(tmp_path / "moc_1.io")
    io_path_2 = str(tmp_path / "moc_2.io")
    io_path_3 = str(tmp_path / "moc_3.io")
    _build_sample_stepped_io(io_path_1)
    _build_sample_stepped_io(io_path_2)
    _build_sample_stepped_io(io_path_3)

    # Bias function simulating a Step 2 candidate where GT engages fewer studs (e.g. overhang)
    # than a centered distractor, so default w_edge=0.60, w_stud=0.40 ranks the distractor #1,
    # whereas GT has higher visual edge alignment (0.92 vs 0.72).
    def overhang_visual_bias(cand, is_gt_match: bool) -> float:
        if is_gt_match:
            cand.stud_count = 1
            return 0.92
        else:
            cand.stud_count = 4
            return 0.72

    moc_cohort = [
        {
            "moc_id": "MOC-001",
            "io_path": io_path_1,
            "simulated_step_corruptions": {
                2: {"visual_edge_scorer": overhang_visual_bias}
            },
        },
        {
            "moc_id": "MOC-002",
            "io_path": io_path_2,
            "simulated_step_corruptions": {
                2: {"visual_edge_scorer": overhang_visual_bias}
            },
        },
        {
            "moc_id": "MOC-003",
            "io_path": io_path_3,
            "simulated_step_corruptions": {},
        },
    ]

    out_dir = str(tmp_path / "loop_out")
    report = run_continuous_improvement_loop(
        moc_items=moc_cohort,
        iterations=2,
        abort_on_first_error=True,
        output_dir=out_dir,
        use_multiprocessing=False,
    )

    assert report["total_iterations_run"] == 2
    # In Iteration 1, MOC-001 and MOC-002 abort at Step 2 due to VISUAL_SCORER_RANKING_ERROR
    assert report["initial_pass_rate_percent"] < 50.0
    # PatternClusterAnalyzer recalibrates w_edge_score upward and Iteration 2 achieves 100% pass rate!
    assert report["final_pass_rate_percent"] == 100.0
    assert report["final_mean_survival_ratio"] == 1.0
    assert os.path.exists(os.path.join(out_dir, "moc_continuous_loop_report.json"))
    assert os.path.exists(os.path.join(out_dir, "moc_continuous_loop_summary.md"))

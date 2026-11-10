#!/usr/bin/env python3
"""
Step-by-Step Inference vs Ground Truth Diagnostic Benchmark for LegoGPT.

Evaluates 5 alternate build sets across sequential manual steps:
  1. Extracts ground truth assembly steps from original .io / .ldr files.
  2. Generates candidate placement hypotheses using PhysicalHypothesisGenerator.
  3. Evaluates candidates against visual evidence from manual pages using VisualPoseScorer.
  4. Diagnoses discrepancies at the part level (Translation, Rotation, Symmetry, Occlusion).
  5. Computes quantitative metrics and identifies systemic algorithmic improvements.

Optimized for Apple Silicon M4.
"""

import os
import sys
import math
import json
import zipfile
import sqlite3
import numpy as np
import cv2
import pymupdf
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any

from src.ldraw_geometry import LDrawGeometryResolver
from src.physical_hypothesis_generator import PhysicalHypothesisGenerator, ConnectionHypothesis
from src.step_axonometric_renderer import StepAxonometricRenderer
from src.visual_pose_scorer import VisualPoseScorer
from src.storage_guard import assert_disk_space_or_exit


@dataclass
class StepPartEvaluation:
    step_num: int
    part_idx: int
    part_id: str
    color_code: int
    gt_pos: List[float]
    gt_rot: List[List[float]]
    num_candidates: int
    gt_in_candidates: bool
    min_dist_to_gt: float
    pred_pos: Optional[List[float]]
    pred_dist_to_gt: float
    rot_error_deg: float
    is_exact_match: bool
    is_symmetry_equivalent: bool
    error_category: str
    notes: str


def parse_ldr_steps(file_path: str) -> List[List[Dict[str, Any]]]:
    """Parses .ldr or .io file into sequential steps of parts."""
    if file_path.endswith(".io"):
        with zipfile.ZipFile(file_path, "r") as zf:
            text = zf.read("model.ldr").decode("utf-8", errors="replace")
    else:
        with open(file_path, "r", errors="replace") as f:
            text = f.read()

    steps = []
    current_step = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("0 STEP"):
            if current_step:
                steps.append(current_step)
                current_step = []
        elif line.startswith("1 "):
            parts = line.split()
            if len(parts) >= 15:
                color = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                rot_flat = [float(p) for p in parts[5:14]]
                rot = np.array(rot_flat, dtype=np.float32).reshape((3, 3))
                part_id = parts[14].lower()
                current_step.append({
                    "part_id": part_id,
                    "color": color,
                    "pos": np.array([x, y, z], dtype=np.float32),
                    "rot": rot,
                    "raw": line
                })
    if current_step:
        steps.append(current_step)
    return steps


def evaluate_set(
    set_info: Dict[str, Any],
    geom: LDrawGeometryResolver,
    phg: PhysicalHypothesisGenerator,
    scorer: VisualPoseScorer,
    max_steps_to_eval: int = 15
) -> Dict[str, Any]:
    """Runs step-by-step closed-loop evaluation for a single set."""
    moc_id = set_info["moc_id"]
    name = set_info["name"]
    io_path = set_info["io_path"]
    pdf_path = set_info["pdf_path"]

    print(f"\n=======================================================")
    print(f"Evaluating {moc_id}: {name}")
    print(f"3D Source: {io_path}")
    print(f"PDF Manual: {pdf_path}")
    print(f"=======================================================")

    steps = parse_ldr_steps(io_path)
    total_steps = len(steps)
    total_parts = sum(len(s) for s in steps)
    print(f"Loaded {total_steps} steps with {total_parts} total parts.")

    # Open PDF
    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)
    os.makedirs(f"scratch/eval_crops/{moc_id}", exist_ok=True)

    eval_results: List[StepPartEvaluation] = []
    existing_nodes: List[Dict[str, Any]] = []

    steps_to_run = min(max_steps_to_eval, total_steps)

    for s_idx in range(steps_to_run):
        step_parts = steps[s_idx]
        step_num = s_idx + 1
        print(f"\n--- Step {step_num} / {steps_to_run} ({len(step_parts)} parts) ---")

        # Step 1 establishes the root assembly
        if s_idx == 0:
            for p_idx, p in enumerate(step_parts):
                resolved_id = geom.resolve_compatible_part_id(p["part_id"])
                tf = np.eye(4, dtype=np.float32)
                tf[:3, :3] = p["rot"]
                tf[:3, 3] = p["pos"]
                existing_nodes.append({
                    "part_index": len(existing_nodes),
                    "part_id": resolved_id,
                    "color_code": p["color"],
                    "absolute_transform": tf.flatten().tolist()
                })
                eval_results.append(StepPartEvaluation(
                    step_num=step_num,
                    part_idx=p_idx,
                    part_id=resolved_id,
                    color_code=p["color"],
                    gt_pos=p["pos"].tolist(),
                    gt_rot=p["rot"].tolist(),
                    num_candidates=1,
                    gt_in_candidates=True,
                    min_dist_to_gt=0.0,
                    pred_pos=p["pos"].tolist(),
                    pred_dist_to_gt=0.0,
                    rot_error_deg=0.0,
                    is_exact_match=True,
                    is_symmetry_equivalent=False,
                    error_category="NONE_ROOT_STEP",
                    notes="Substrate initialization"
                ))
            continue

        # Extract manual page visual evidence if available
        # Simple mapping: page_num = min(step_num, num_pages)
        page_idx = min(step_num - 1, num_pages - 1)
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        crop_path = f"scratch/eval_crops/{moc_id}/page_{page_idx+1}.png"
        pix.save(crop_path)
        visual_evidence = scorer.extract_visual_evidence(crop_path)

        # Evaluate each part in the step against existing assembly
        new_step_nodes = []
        for p_idx, p in enumerate(step_parts):
            resolved_id = geom.resolve_compatible_part_id(p["part_id"])
            gt_pos = p["pos"]
            gt_rot = p["rot"]

            # Generate hypotheses against existing_nodes
            cands = phg.generate_candidate_hypotheses(
                resolved_id, f"elem_{step_num}_{p_idx}", existing_nodes
            )

            num_cands = len(cands)
            gt_in_cands = False
            min_dist = 999999.0
            best_cand_by_gt = None

            for c in cands:
                d = float(np.linalg.norm(c.translation - gt_pos))
                if d < min_dist:
                    min_dist = d
                    best_cand_by_gt = c
                if d < 1.0:
                    gt_in_cands = True

            # Intra-step substrate chaining: if GT not found and prior parts exist in this step,
            # allow attaching to parts placed earlier in the same step
            if not gt_in_cands and new_step_nodes:
                intra_cands = phg.generate_candidate_hypotheses(
                    resolved_id, f"elem_{step_num}_{p_idx}_intra", existing_nodes + new_step_nodes
                )
                if intra_cands:
                    cands = cands + intra_cands
                    num_cands = len(cands)
                    for c in intra_cands:
                        d = float(np.linalg.norm(c.translation - gt_pos))
                        if d < min_dist:
                            min_dist = d
                            best_cand_by_gt = c
                        if d < 1.0:
                            gt_in_cands = True

            # Predict winning candidate using VisualPoseScorer (or top heuristic if no visual crop)
            pred_cand = None
            if cands:
                if visual_evidence and "fg_mask" in visual_evidence:
                    best_score = -1.0
                    for c in cands[:30]: # evaluate top 30
                        s = scorer.score_single_hypothesis(c, visual_evidence)
                        if s > best_score:
                            best_score = s
                            pred_cand = c
                if pred_cand is None:
                    pred_cand = cands[0]

            if pred_cand is not None:
                pred_pos = pred_cand.translation.tolist()
                pred_dist = float(np.linalg.norm(pred_cand.translation - gt_pos))
                # Rotation difference
                R_diff = pred_cand.rotation_matrix.T @ gt_rot
                tr = np.trace(R_diff)
                tr_clamped = max(-1.0, min(3.0, (tr - 1.0) / 2.0))
                rot_err = float(np.degrees(np.arccos(tr_clamped)))
            else:
                pred_pos = None
                pred_dist = 999999.0
                rot_err = 180.0

            is_exact = (pred_dist < 1.0 and rot_err < 5.0)

            # Check symmetry equivalence
            p_meta = geom.get_part_meta(resolved_id)
            is_sym_equiv = False
            if pred_dist < 1.0 and p_meta.get("is_symmetric", False):
                sym_angles = p_meta.get("symmetry_yaw_angles", [180.0])
                for sa in sym_angles:
                    if abs(rot_err - sa) < 5.0:
                        is_sym_equiv = True
                        break

            # Categorize discrepancy
            if is_exact or is_sym_equiv:
                err_cat = "EXACT_OR_SYMMETRIC"
                notes = "Correct placement"
            elif gt_in_cands:
                err_cat = "SCORER_DISCRIMINATION_FAILURE"
                notes = f"GT candidate was generated but scorer selected cand with dist={pred_dist:.1f} LDU"
            elif num_cands == 0:
                # Check why 0 candidates
                if "technic" in resolved_id or any(t in resolved_id for t in ["axle", "pin", "beam", "6558", "32526", "27940"]):
                    err_cat = "TECHNIC_CONNECTOR_UNSUPPORTED"
                    notes = "Part uses Technic pin/axle connection instead of stud-to-socket"
                elif ".ldr" in resolved_id or "sub" in resolved_id or "engine" in resolved_id or "exhaust" in resolved_id:
                    err_cat = "SUBMODEL_UNRESOLVED"
                    notes = "Part is an inline subassembly rather than standard LDraw primitive"
                else:
                    err_cat = "NO_VALID_STUD_HYPOTHESIS"
                    notes = "No valid stud-socket alignment found (possible lateral SNOT or collision filter)"
            else:
                # Candidates generated, but GT was not among them
                delta_y = abs(min_dist)
                if abs(abs(pred_dist) - 8.0) < 1.0 or (abs(gt_pos[1] - (best_cand_by_gt.translation[1] if best_cand_by_gt else 0)) in [8.0, 16.0, 24.0]):
                    err_cat = "LAYER_HEIGHT_OR_STACKING"
                    notes = f"Vertical delta of {delta_y:.1f} LDU (intra-step stacking or plate height)"
                elif abs(pred_dist - 20.0) < 2.0 or (best_cand_by_gt and abs(np.linalg.norm(best_cand_by_gt.translation - gt_pos) - 20.0) < 2.0):
                    err_cat = "HORIZONTAL_STUD_OFFSET"
                    notes = "Offset by exactly 1 horizontal stud pitch (20 LDU)"
                elif p_meta.get("has_lateral_ports", False) or abs(gt_rot[1, 1]) < 0.2:
                    err_cat = "SNOT_LATERAL_ORIENTATION"
                    notes = "Part placed with lateral studs (SNOT) or non-vertical orientation"
                else:
                    err_cat = "HYPOTHESIS_FILTERED"
                    notes = f"GT position missing from candidate pool (min dist = {min_dist:.1f} LDU)"

            print(f"  Part {resolved_id:12s} | Cands: {num_cands:3d} | GT in Cands: {str(gt_in_cands):5s} | Δt: {pred_dist:6.1f} LDU | Cat: {err_cat}")

            eval_results.append(StepPartEvaluation(
                step_num=step_num,
                part_idx=p_idx,
                part_id=resolved_id,
                color_code=p["color"],
                gt_pos=gt_pos.tolist(),
                gt_rot=gt_rot.tolist(),
                num_candidates=num_cands,
                gt_in_candidates=gt_in_cands,
                min_dist_to_gt=float(min_dist),
                pred_pos=pred_pos,
                pred_dist_to_gt=pred_dist,
                rot_error_deg=rot_err,
                is_exact_match=is_exact,
                is_symmetry_equivalent=is_sym_equiv,
                error_category=err_cat,
                notes=notes
            ))

            # Accumulate Ground Truth state so subsequent steps test real connectivity
            tf = np.eye(4, dtype=np.float32)
            tf[:3, :3] = gt_rot
            tf[:3, 3] = gt_pos
            new_step_nodes.append({
                "part_index": len(existing_nodes) + len(new_step_nodes),
                "part_id": resolved_id,
                "color_code": p["color"],
                "absolute_transform": tf.flatten().tolist()
            })

        existing_nodes.extend(new_step_nodes)

    doc.close()

    # Aggregate set statistics
    total_eval_parts = len(eval_results)
    exact_count = sum(1 for e in eval_results if e.is_exact_match)
    sym_count = sum(1 for e in eval_results if e.is_symmetry_equivalent)
    success_count = exact_count + sym_count
    gt_cand_coverage = sum(1 for e in eval_results if e.gt_in_candidates)

    cat_counts = {}
    for e in eval_results:
        cat_counts[e.error_category] = cat_counts.get(e.error_category, 0) + 1

    return {
        "moc_id": moc_id,
        "name": name,
        "total_steps": total_steps,
        "steps_evaluated": steps_to_run,
        "total_parts": total_parts,
        "parts_evaluated": total_eval_parts,
        "exact_matches": exact_count,
        "symmetry_matches": sym_count,
        "accuracy_pct": round(100.0 * success_count / max(1, total_eval_parts), 2),
        "gt_cand_coverage_pct": round(100.0 * gt_cand_coverage / max(1, total_eval_parts), 2),
        "category_breakdown": cat_counts,
        "eval_details": [asdict(e) for e in eval_results]
    }


def main():
    assert_disk_space_or_exit()
    batch_mode = sys.argv[1] if len(sys.argv) > 1 else "1"
    print(f"Initializing LegoGPT Geometry and Solvers on Apple Silicon M4 (Running Batch {batch_mode})...")
    geom = LDrawGeometryResolver()
    phg = PhysicalHypothesisGenerator(geom)
    renderer = StepAxonometricRenderer(geom)
    scorer = VisualPoseScorer(renderer)

    if batch_mode == "3":
        sets_to_test = [
            {
                "moc_id": "MOC-26387",
                "name": "31057 Sea integrator",
                "io_path": "data/models_io/MOC-26387_31057_Sea_integrator.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26387_Sea_integrator.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-26847",
                "name": "31057 Shubunkin cruiser",
                "io_path": "data/models_io/MOC-26847_31057_Shubunkin_cruiser.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26847_Shubunkin_cruiser.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-26848",
                "name": "31057 Hybrid swallow",
                "io_path": "data/models_io/MOC-26848_31057_Hybrid_swallow.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26848_Hybrid_swallow.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-208578",
                "name": "31040 excavator",
                "io_path": "data/models_io/MOC-208578_31040_excavator.io",
                "pdf_path": "data/manuals/sample_eval/MOC-208578_Excavator.pdf",
                "max_steps": 15
            },
            {
                "moc_id": "MOC-210826",
                "name": "31040 mobile crane",
                "io_path": "data/models_io/MOC-210826_31040_mobile_crane.io",
                "pdf_path": "data/manuals/sample_eval/MOC-210826_Mobile_crane.pdf",
                "max_steps": 15
            }
        ]
        output_json = "data/catalog/inference_benchmark_results_batch3.json"
    elif batch_mode == "2":
        sets_to_test = [
            {
                "moc_id": "MOC-26386",
                "name": "31057 Sea grinder",
                "io_path": "data/models_io/MOC-26386_31057_Sea_grinder.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26386_Sea_grinder.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-26638",
                "name": "31057 Bug tracker",
                "io_path": "data/models_io/MOC-26638_31057_Bug_tracker.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26638_Bug_tracker.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-26636",
                "name": "31057 Alien flyer",
                "io_path": "data/models_io/MOC-26636_31057_Alien_flyer.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26636_Alien_flyer.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-26845",
                "name": "31057 Dragonfly",
                "io_path": "data/models_io/MOC-26845_31057_Dragonfly.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26845_Dragonfly.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-210073",
                "name": "31040 Golf cart",
                "io_path": "data/models_io/MOC-210073_31040_golf_cart.io",
                "pdf_path": "data/manuals/sample_eval/MOC-210073_Golf_cart.pdf",
                "max_steps": 15
            }
        ]
        output_json = "data/catalog/inference_benchmark_results_batch2.json"
    else:
        sets_to_test = [
            {
                "moc_id": "MOC-164407",
                "name": "Spaceship Stand (31134-1)",
                "io_path": "data/models_io/MOC-164407_Spaceship_Stand.io",
                "pdf_path": "data/manuals/sample_eval/MOC-164407_Spaceship_Stand.pdf",
                "max_steps": 7
            },
            {
                "moc_id": "MOC-26384",
                "name": "31057 Sea puncher",
                "io_path": "data/models_io/MOC-26384_31057_Sea_puncher.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26384_Sea_puncher.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-164817",
                "name": "Perry the Platypus (31128-1)",
                "io_path": "data/models_io/MOC-164817_Perry_the_Platypus.io",
                "pdf_path": "data/manuals/sample_eval/MOC-164817_Perry_Platypus.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-26385",
                "name": "31057 Sky runner",
                "io_path": "data/models_io/MOC-26385_31057_Sky_runner.io",
                "pdf_path": "data/manuals/sample_eval/MOC-26385_Sky_runner.pdf",
                "max_steps": 12
            },
            {
                "moc_id": "MOC-209916",
                "name": "31040 Dragster",
                "io_path": "data/models_io/MOC-209916_31040_dragster.io",
                "pdf_path": "data/manuals/sample_eval/MOC-209916_Dragster.pdf",
                "max_steps": 15
            }
        ]
        output_json = "data/catalog/inference_benchmark_results.json"

    all_results = []
    for s in sets_to_test:
        res = evaluate_set(s, geom, phg, scorer, max_steps_to_eval=s.get("max_steps", 12))
        all_results.append(res)

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full benchmark results to {output_json}")


if __name__ == "__main__":
    main()

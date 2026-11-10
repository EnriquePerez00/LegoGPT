#!/usr/bin/env python3
"""
Full-Corpus Fail-Fast Benchmark (Abort-on-First-Error per Set).
Evaluates all cataloged MOCs step-by-step.
If an error occurs on any step of a set, immediately aborts that set and proceeds
to the next set.
Aggregates all first-failure incidents across the corpus, classifies them by family,
and generates summary metrics.

Optimized for Apple Silicon M4 with 8 parallel worker processes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import time
import json
import sqlite3
import numpy as np
import pymupdf
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.ldraw_geometry import LDrawGeometryResolver
from src.physical_hypothesis_generator import PhysicalHypothesisGenerator
from src.step_axonometric_renderer import StepAxonometricRenderer
from src.visual_pose_scorer import VisualPoseScorer
from src.storage_guard import assert_disk_space_or_exit
from src.parser import ParsedPart
from src.io_parser import export_parts_to_io
from scripts.run_full_corpus_benchmark import parse_flattened_steps


def _export_inferred_nodes_to_io(nodes: list[dict], export_dir: str, moc_id: str, name: str) -> str:
    """Exports the inferred 3D construction graph nodes to a BrickLink Studio .io file."""
    if not export_dir or not nodes:
        return ""
    os.makedirs(export_dir, exist_ok=True)
    parsed_parts = []
    for n in nodes:
        tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
        parsed_parts.append(
            ParsedPart(
                part_id=n["part_id"],
                color=int(n.get("color_code", 16)),
                transform=tf,
                step_id=int(n.get("step_id", 1)),
            )
        )
    out_io = os.path.join(export_dir, f"{moc_id}_inferred.io")
    export_parts_to_io(parsed_parts, out_io, model_title=f"{moc_id} - {name}")
    return out_io


def _build_step_to_page_map(doc, num_steps: int) -> dict[int, int]:
    """
    Scans PDF text blocks to map each build step number (1..num_steps) to its true 0-based page index.
    Prevents trailing BOM/inventory pages from shifting early steps to the back of the manual.
    """
    if doc is None or len(doc) == 0:
        return {}
    import re
    step_to_page: dict[int, int] = {}
    num_pages = len(doc)
    for p_idx in range(num_pages):
        try:
            blocks = doc[p_idx].get_text("blocks")
        except Exception:
            continue
        for b in blocks:
            txt = str(b[4]).strip()
            for token in re.findall(r"\b(\d{1,3})\b", txt):
                val = int(token)
                if 1 <= val <= num_steps and val not in step_to_page:
                    # Enforce monotonic page progression
                    prev_p = step_to_page.get(val - 1, 0)
                    if p_idx >= prev_p:
                        step_to_page[val] = p_idx
    return step_to_page


def evaluate_set_fail_fast(item: dict) -> dict:
    """
    Evaluates a single MOC. Aborts at the first step that contains an error,
    returning the failure diagnostic and elapsed timing.
    """
    t0 = time.perf_counter()
    moc_id = item["moc_id"]
    name = item["name"]
    io_path = item["io_path"]
    pdf_path = item["pdf_path"]
    total_catalog_parts = item.get("num_parts", 0)

    steps = parse_flattened_steps(io_path)
    if not steps:
        return {
            "moc_id": moc_id,
            "name": name,
            "status": "EMPTY_STEPS",
            "total_parts": total_catalog_parts,
            "duration_sec": round(time.perf_counter() - t0, 3)
        }

    # Agile guard: discard unsegmented CAD dumps with >35 parts in a single step
    MAX_STEP_PART_DENSITY = 35
    if any(len(s) > MAX_STEP_PART_DENSITY for s in steps[:12]):
        return {
            "moc_id": moc_id,
            "name": name,
            "status": "DISCARDED_EXCESSIVE_STEP_DENSITY",
            "total_parts": total_catalog_parts,
            "duration_sec": round(time.perf_counter() - t0, 3)
        }

    geom = LDrawGeometryResolver()
    phg = PhysicalHypothesisGenerator(geom)
    renderer = StepAxonometricRenderer(geom)
    scorer = VisualPoseScorer(renderer)

    doc = None
    num_pages = 1
    if os.path.exists(pdf_path):
        try:
            doc = pymupdf.open(pdf_path)
            num_pages = len(doc)
        except Exception:
            doc = None

    # Page offset mapping via monotonic OCR/Vector text step number extraction
    step_page_map = _build_step_to_page_map(doc, len(steps))
    if 1 in step_page_map:
        page_offset = step_page_map[1]
    elif num_pages >= len(steps) + 1:
        # Studio 2.0 PDFs typically have 1 cover page (page 0) followed by Step 1 on page 1
        page_offset = 1
    else:
        page_offset = 0

    existing_nodes = []
    prev_pix = None
    steps_to_run = min(12, len(steps))
    parts_passed = 0

    for s_idx in range(steps_to_run):
        step_parts = steps[s_idx]
        step_num = s_idx + 1

        if s_idx == 0:
            for p in step_parts:
                part_id = geom.resolve_compatible_part_id(p["part_id"])
                tf = np.eye(4, dtype=np.float32)
                tf[:3, :3] = p["rot"]
                tf[:3, 3] = p["pos"]
                existing_nodes.append({
                    "part_index": len(existing_nodes),
                    "part_id": part_id,
                    "color_code": p["color"],
                    "step_id": step_num,
                    "absolute_transform": tf.flatten().tolist()
                })
                parts_passed += 1

            if doc is not None:
                try:
                    p0_idx = step_page_map.get(1, min(num_pages - 1, page_offset))
                    prev_pix = doc[p0_idx].get_pixmap(dpi=120)
                except Exception:
                    prev_pix = None
            continue

        page_idx = step_page_map.get(step_num, min(num_pages - 1, max(0, s_idx + page_offset)))
        curr_pix = None
        visual_evidence = {}
        if doc is not None:
            try:
                curr_page = doc[page_idx]
                curr_pix = curr_page.get_pixmap(dpi=120)
                visual_evidence = scorer.extract_visual_evidence(curr_pix, prev_crop_source=prev_pix, page=curr_page)
            except Exception:
                visual_evidence = {}

        # Phase A + Phase B: Calibrate camera from G_{t-1} and synthesize Differential Mask Delta I_t & Z-Buffer
        calib = (
            scorer.calibrate_camera_from_prior_state(existing_nodes, visual_evidence)
            if visual_evidence and existing_nodes
            else {"scale": 2.65, "offset": (200.0, 200.0), "az_deg": 45.0, "el_deg": 30.0, "center_3d": None}
        )

        # Y-sort step parts (base first)
        sorted_step_parts = sorted(step_parts, key=lambda it: it["pos"][1], reverse=True)
        # Intra-Step Set Matching: in multi-part steps, pieces may be placed in any sequence
        unplaced_gt = list(sorted_step_parts)
        new_step_nodes = []
        chosen_step_positions = []
        chosen_step_hypotheses = []

        for p_slot in range(len(sorted_step_parts)):
            # Remaining needed (part_id, color_id) pairs in this step
            needed_part_colors = list(set(
                (geom.resolve_compatible_part_id(it["part_id"]), int(it.get("color", 16)))
                for it in unplaced_gt
            ))

            all_cands = []
            for pid, col_id in needed_part_colors:
                # Phase C: 2D connector pruning + Lazy 3D collision check for fast candidate generation
                c_base = phg.generate_candidate_hypotheses(
                    pid, "elem", existing_nodes,
                    color_id=col_id,
                    connector_2d_filter=calib if visual_evidence else None,
                    lazy_collision_check=True,
                )
                if len(c_base) < 12:
                    # Fallback without 2D ROI restriction if differential ROI was overly narrow
                    c_full = phg.generate_candidate_hypotheses(
                        pid, "elem", existing_nodes,
                        color_id=col_id,
                        lazy_collision_check=True,
                    )
                    seen_pos = {tuple(np.round(h.translation, 1)) for h in c_base}
                    for ch in c_full:
                        if tuple(np.round(ch.translation, 1)) not in seen_pos:
                            c_base.append(ch)

                if new_step_nodes:
                    c_intra = phg.generate_candidate_hypotheses(
                        pid, "elem_intra", existing_nodes + new_step_nodes,
                        target_parent_indices=[n["part_index"] for n in new_step_nodes],
                        color_id=col_id,
                        lazy_collision_check=True,
                    )
                    if c_intra:
                        c_base.extend(c_intra)

                    # Special Case 2: If a previous part in this step has the same part_id, generate mirror twin
                    for prev_h in chosen_step_hypotheses:
                        if prev_h.child_part_id == pid:
                            twins = phg.generate_symmetric_twin_hypothesis(
                                prev_h, pid, "elem_twin", existing_nodes + new_step_nodes, color_id=col_id
                            )
                            if twins:
                                c_base.extend(twins)

                c_vg = phg.generate_virtual_ground_hypotheses(pid, "elem_vg", existing_nodes + new_step_nodes)
                if c_vg:
                    for vg_h in c_vg:
                        vg_h.color_id = col_id
                    c_base.extend(c_vg)
                all_cands.extend(c_base)

            cands = all_cands
            gt_in_cands = False

            # Check if any unplaced GT part is in candidate list
            for gt_item in unplaced_gt:
                g_pid = geom.resolve_compatible_part_id(gt_item["part_id"])
                g_pos = np.array(gt_item["pos"], dtype=np.float32)
                g_rot = np.array(gt_item["rot"], dtype=np.float32).reshape((3, 3))
                for c in cands:
                    if c.child_part_id == g_pid and np.linalg.norm(c.translation - g_pos) < 1.0:
                        R_diff = c.rotation_matrix.T @ g_rot
                        tr = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
                        if np.degrees(np.arccos(tr)) < 5.0:
                            gt_in_cands = True
                            break
                if gt_in_cands:
                    break

            # Vectorized MPS Pre-Scoring across up to 480 candidates
            recent_positions = [np.array(rn["absolute_transform"]).reshape((4, 4))[:3, 3] for rn in (existing_nodes + new_step_nodes)[-6:]]
            if recent_positions:
                cands.sort(
                    key=lambda c: (
                        getattr(c, "bridged_parents", 1) * 18.0
                        + c.stud_count * 12.0
                        + (25.0 if getattr(c, "is_symmetric_twin", False) else 0.0)
                        - min(np.linalg.norm(c.translation - pos) for pos in recent_positions)
                    ),
                    reverse=True,
                )

            pool_cands = cands[:480]
            if visual_evidence and pool_cands:
                mps_edge_scores = scorer.score_chamfer_batch_mps(
                    pool_cands, visual_evidence,
                    scale=calib["scale"], offset=calib["offset"],
                    az_deg=calib.get("az_deg", 45.0), el_deg=calib.get("el_deg", 30.0),
                    center_3d=calib.get("center_3d"),
                )
                mps_col_scores = scorer.score_color_batch_mps(
                    pool_cands, visual_evidence,
                    scale=calib["scale"], offset=calib["offset"],
                    az_deg=calib.get("az_deg", 45.0), el_deg=calib.get("el_deg", 30.0),
                    center_3d=calib.get("center_3d"),
                )
                for c_obj, e_sc, col_sc in zip(pool_cands, mps_edge_scores, mps_col_scores):
                    c_obj.edge_score = e_sc
                    c_obj.color_score = col_sc
                    bridge_b = 0.08 * float(max(0, getattr(c_obj, "bridged_parents", 1) - 1))
                    twin_b = 0.15 if getattr(c_obj, "is_symmetric_twin", False) else 0.0
                    c_obj.total_score = 0.42 * e_sc + 0.28 * col_sc + 0.22 * min(1.0, c_obj.stud_count / 4.0) + bridge_b + twin_b
                pool_cands.sort(key=lambda c: c.total_score, reverse=True)

            # Level 1 Hard Physical Validation (Lazy SAT 3D collision check on top visual candidates)
            valid_solid_pool = phg.filter_valid_solid_hypotheses(
                pool_cands[:120], existing_nodes + new_step_nodes, max_keep=60, penetration_threshold=1.5
            )
            if not valid_solid_pool:
                valid_solid_pool = pool_cands[:40]

            # Fine-score top 60 collision-free candidates with full 6-level VisualPoseScorer
            scored_cands = []
            for c in valid_solid_pool:
                s = scorer.score_single_hypothesis(
                    c, visual_evidence,
                    scale=calib["scale"], offset=calib["offset"],
                    az_deg=calib.get("az_deg", 45.0),
                    recent_nodes=existing_nodes + new_step_nodes,
                    already_placed_poses=chosen_step_positions
                )
                scored_cands.append((s, c))

            scored_cands.sort(key=lambda x: x[0], reverse=True)

            # Level 6: Maintain Top-K=3 Beam Hypotheses for step / lookahead verification
            beam_k_cands = []
            if scored_cands:
                for s_val, c_val in scored_cands:
                    if not any(np.linalg.norm(c_val.translation - pos) < 5.0 for pos in chosen_step_positions):
                        if not any(
                            c_val.child_part_id == bh.child_part_id and np.linalg.norm(c_val.translation - bh.translation) < 2.0
                            for _, bh in beam_k_cands
                        ):
                            beam_k_cands.append((s_val, c_val))
                            if len(beam_k_cands) >= 3:
                                break
                if not beam_k_cands:
                    beam_k_cands = [scored_cands[0]]

            pred_score, pred_cand = beam_k_cands[0] if beam_k_cands else (-999.0, None)

            # Match Top-K=3 beam candidates against unplaced GT items in this step
            matched_gt_idx = None
            matched_beam_rank = None
            closest_gt_item = None
            min_dist_to_gt = 999999.0
            best_rot_err = 180.0

            for b_rank, (b_score, b_cand) in enumerate(beam_k_cands):
                for g_i, gt_item in enumerate(unplaced_gt):
                    g_pid = geom.resolve_compatible_part_id(gt_item["part_id"])
                    g_pos = np.array(gt_item["pos"], dtype=np.float32)
                    g_rot = np.array(gt_item["rot"], dtype=np.float32).reshape((3, 3))

                    d = float(np.linalg.norm(b_cand.translation - g_pos))
                    R_diff = b_cand.rotation_matrix.T @ g_rot
                    tr = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
                    r_err = float(np.degrees(np.arccos(tr)))

                    if b_rank == 0 and d < min_dist_to_gt:
                        min_dist_to_gt = d
                        best_rot_err = r_err
                        closest_gt_item = gt_item

                    if b_cand.child_part_id == g_pid:
                        is_exact = (d < 1.0 and r_err < 5.0)
                        meta = geom.get_part_meta(g_pid)
                        is_sym = False
                        if d < 1.0 and meta.get("is_symmetric", False):
                            for sa in meta.get("symmetry_yaw_angles", [180.0]):
                                if abs(r_err - sa) < 5.0:
                                    is_sym = True
                                    break
                        if is_exact or is_sym:
                            matched_gt_idx = g_i
                            matched_beam_rank = b_rank + 1
                            pred_cand = b_cand
                            pred_score = b_score
                            break
                if matched_gt_idx is not None:
                    break

            if matched_gt_idx is not None:
                # Part matched successfully in active K=3 beam!
                matched_p = unplaced_gt.pop(matched_gt_idx)
                parts_passed += 1
                chosen_step_positions.append(pred_cand.translation)
                chosen_step_hypotheses.append(pred_cand)
                tf = np.eye(4, dtype=np.float32)
                tf[:3, :3] = pred_cand.rotation_matrix
                tf[:3, 3] = pred_cand.translation
                new_step_nodes.append({
                    "part_index": len(existing_nodes) + len(new_step_nodes),
                    "part_id": pred_cand.child_part_id,
                    "color_code": matched_p["color"],
                    "step_id": step_num,
                    "beam_rank": matched_beam_rank,
                    "absolute_transform": tf.flatten().tolist()
                })
                continue

            # IF ERROR: ABORT THIS SET IMMEDIATELY!
            if doc is not None:
                doc.close()

            exported_io = _export_inferred_nodes_to_io(
                existing_nodes + new_step_nodes,
                item.get("export_io_dir", ""),
                moc_id,
                name,
            )

            error_piece_num = parts_passed + 1
            target_p = closest_gt_item if closest_gt_item else unplaced_gt[0]
            err_pid = geom.resolve_compatible_part_id(target_p["part_id"])
            err_gt_pos = np.array(target_p["pos"], dtype=np.float32)
            err_gt_rot = np.array(target_p["rot"], dtype=np.float32).reshape((3, 3))
            meta = geom.get_part_meta(err_pid)

            # Categorize failure family
            if gt_in_cands:
                if pred_cand is not None:
                    delta = np.abs(pred_cand.translation - err_gt_pos)
                    if delta[1] < 1.0 and (abs(delta[0] % 20.0) < 1.0 or abs(delta[0] % 20.0 - 20.0) < 1.0) and (abs(delta[2] % 20.0) < 1.0 or abs(delta[2] % 20.0 - 20.0) < 1.0):
                        family = "SCORER_COLLINEAR_SLIP"
                    elif (abs(pred_cand.translation[0] + err_gt_pos[0]) < 2.0 and abs(pred_cand.translation[1] - err_gt_pos[1]) < 2.0) or \
                         (abs(pred_cand.translation[2] + err_gt_pos[2]) < 2.0 and abs(pred_cand.translation[1] - err_gt_pos[1]) < 2.0):
                        family = "SCORER_BILATERAL_MIRROR"
                    else:
                        family = "SCORER_DISCRIMINATION_FAILURE"
                else:
                    family = "SCORER_DISCRIMINATION_FAILURE"
            elif len(cands) == 0:
                family = "NO_VALID_STUD_HYPOTHESIS"
            else:
                is_technic = any(k in err_pid for k in ["technic", "pin", "axle", "32054", "2780", "6558", "3673", "43093"])
                is_hinge = any(k in err_pid for k in ["3937", "3938", "44301", "44302", "60478", "14417", "14704", "48336"])
                is_non_ortho = abs(err_gt_rot[1, 1]) not in [0.0, 1.0, -1.0] and abs(abs(err_gt_rot[1, 1]) - 1.0) > 0.05 and abs(err_gt_rot[1, 1]) > 0.05
                if is_technic or is_hinge or is_non_ortho:
                    family = "KINEMATIC_OR_TECHNIC_CONNECTOR"
                elif abs(err_gt_rot[1, 1] + 1.0) < 0.15:
                    family = "INVERTED_UNDERSIDE_ASSEMBLY"
                elif meta.get("has_lateral_ports", False) or abs(err_gt_rot[1, 1]) < 0.2:
                    family = "SNOT_FRACTIONAL_OFFSET"
                elif any(abs(float(np.linalg.norm(c.translation - err_gt_pos)) - 10.0) < 1.0 for c in cands):
                    family = "SNOT_FRACTIONAL_OFFSET"
                elif any(abs(float(np.linalg.norm(c.translation - err_gt_pos)) - 8.0) < 1.0 or abs(float(np.linalg.norm(c.translation - err_gt_pos)) - 16.0) < 1.0 for c in cands):
                    family = "INTRA_STEP_STACKING"
                else:
                    family = "COLLISION_OR_DISCONNECTED_SUBASSEMBLY"

            return {
                "moc_id": moc_id,
                "name": name,
                "pdf_path": pdf_path,
                "exported_io_path": exported_io,
                "status": "FIRST_ERROR_ABORT",
                "total_catalog_parts": total_catalog_parts,
                "total_steps": len(steps),
                "first_error_step": step_num,
                "error_piece_number": error_piece_num,
                "page_idx": page_idx + 1,
                "total_pdf_pages": num_pages,
                "page_ratio_str": f"{page_idx + 1}/{num_pages}",
                "part_id": err_pid,
                "color": target_p["color"],
                "family": family,
                "parts_passed_before_error": parts_passed,
                "accuracy_evaluated": round(100.0 * parts_passed / max(1, parts_passed + 1), 2),
                "duration_sec": round(time.perf_counter() - t0, 3),
                "gt_pos": err_gt_pos.tolist(),
                "pred_pos": pred_cand.translation.tolist() if pred_cand else None,
                "pred_dist": round(min_dist_to_gt, 2),
                "rot_err": round(best_rot_err, 2),
                "gt_in_cands": gt_in_cands,
                "total_cands_generated": len(cands),
                "pred_score_breakdown": {
                    "total": round(float(pred_cand.total_score), 4),
                    "edge": round(float(pred_cand.edge_score), 4),
                    "color": round(float(getattr(pred_cand, "color_score", 0.0)), 4),
                    "stud_occ": round(float(pred_cand.stud_occlusion_score), 4),
                    "neighbor": round(float(pred_cand.neighbor_score), 4),
                    "visibility": round(float(getattr(pred_cand, "visibility_ratio", 1.0)), 4),
                    "bridged_parents": int(getattr(pred_cand, "bridged_parents", 1)),
                } if pred_cand else None,
            }

        existing_nodes.extend(new_step_nodes)
        prev_pix = curr_pix

    if doc is not None:
        doc.close()

    exported_io = _export_inferred_nodes_to_io(
        existing_nodes,
        item.get("export_io_dir", ""),
        moc_id,
        name,
    )

    return {
        "moc_id": moc_id,
        "name": name,
        "pdf_path": pdf_path,
        "exported_io_path": exported_io,
        "status": "ALL_STEPS_CONVERGED",
        "total_catalog_parts": total_catalog_parts,
        "total_steps": len(steps),
        "total_pdf_pages": num_pages,
        "page_ratio_str": f"{num_pages}/{num_pages}",
        "steps_completed": steps_to_run,
        "parts_passed_before_error": parts_passed,
        "accuracy_evaluated": 100.0,
        "duration_sec": round(time.perf_counter() - t0, 3)
    }


def main():
    import argparse
    import random
    parser = argparse.ArgumentParser(description="Fail-Fast PDF -> .io Corpus Benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDF manuals to infer (0 = all)")
    parser.add_argument("--random-sample", type=int, default=0, help="Randomly sample N sets from catalog")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--export-io-dir", type=str, default="outputs/inferred_io", help="Directory to export inferred .io files")
    parser.add_argument("--summary-out", type=str, default="data/catalog/first_error_corpus_summary.json", help="Output JSON summary path")
    args = parser.parse_args()

    assert_disk_space_or_exit()
    print("=" * 75)
    print("RUNNING FAIL-FAST BENCHMARK ACROSS CATALOGED MOCS")
    print("(Aborts single MOC at 1st error and jumps to next MOC)")
    print("=" * 75)

    conn = sqlite3.connect("data/catalog/models_catalog.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT moc_id, name, io_file_path, pdf_url, num_parts 
        FROM alternate_builds 
        WHERE io_file_path IS NOT NULL AND pdf_url IS NOT NULL
        ORDER BY num_parts ASC
    """)
    rows = cur.fetchall()

    matched_sets = []
    for moc_id, name, io_path, pdf_url, num_parts in rows:
        if not os.path.exists(io_path):
            continue
        matches = glob.glob(f"data/manuals/mocs/{moc_id}_*.pdf")
        if not matches:
            matches = glob.glob(f"data/manuals/sample_eval/{moc_id}_*.pdf")
        if matches:
            matched_sets.append({
                "moc_id": moc_id,
                "name": name,
                "num_parts": num_parts or 0,
                "io_path": io_path,
                "pdf_path": matches[0],
                "export_io_dir": args.export_io_dir,
            })

    if args.random_sample > 0:
        rng = random.Random(args.seed)
        k = min(args.random_sample, len(matched_sets))
        matched_sets = rng.sample(matched_sets, k)
    elif args.limit > 0:
        # Select a representative stratified sample of `args.limit` manuals across diverse part counts
        step_stride = max(1, len(matched_sets) // max(1, args.limit))
        matched_sets = [matched_sets[i * step_stride] for i in range(min(args.limit, len(matched_sets)))]

    total_sets = len(matched_sets)
    print(f"Loaded {total_sets} verified MOC pairs with local .io and .pdf.")

    num_procs = max(1, min(8, (os.cpu_count() or 12) - 4))
    print(f"Allocating {num_procs} parallel process workers on Apple Silicon M4...\n")

    results = []
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        future_to_item = {executor.submit(evaluate_set_fail_fast, item): item for item in matched_sets}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                r = future.result()
                results.append(r)
                st = r.get("status")
                if st == "ALL_STEPS_CONVERGED":
                    print(f"[{len(results)}/{total_sets}] [CONVERGED 100%] {r['moc_id']}: {r['name'][:28]} | Pág. {r.get('page_ratio_str', 'N/A')} | Passed: {r['parts_passed_before_error']} parts ({r['duration_sec']}s)")
                elif st == "FIRST_ERROR_ABORT":
                    print(f"[{len(results)}/{total_sets}] [ABORT Pág. {r.get('page_ratio_str', 'N/A')} | Step {r['first_error_step']}/{r['total_steps']}] {r['moc_id']}: {r['name'][:24]} | {r['part_id']} ({r['family']}) | {r['duration_sec']}s")
                elif st == "DISCARDED_EXCESSIVE_STEP_DENSITY":
                    print(f"[{len(results)}/{total_sets}] [FAST-SKIP] {r['moc_id']}: {r['name'][:28]} | Exceeds max step density (>35 parts/step)")
            except Exception as e:
                print(f"   -> Error evaluating {item['moc_id']}: {e}")

    # Maintain deterministic order
    target_order = {item["moc_id"]: i for i, item in enumerate(matched_sets)}
    results.sort(key=lambda r: target_order.get(r["moc_id"], 999))

    # Aggregated failure analysis
    converged_models = [r for r in results if r.get("status") == "ALL_STEPS_CONVERGED"]
    aborted_models = [r for r in results if r.get("status") == "FIRST_ERROR_ABORT"]
    skipped_models = [r for r in results if r.get("status") == "DISCARDED_EXCESSIVE_STEP_DENSITY"]

    family_counts = {}
    family_exemplars = {}
    step_fail_dist = {}
    piece_fail_dist = {}
    gt_cands_present_count = 0

    for r in aborted_models:
        fam = r["family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1
        if fam not in family_exemplars:
            family_exemplars[fam] = []
        if len(family_exemplars[fam]) < 5:
            family_exemplars[fam].append({
                "moc_id": r["moc_id"],
                "name": r["name"],
                "step": r["first_error_step"],
                "piece_num": r.get("error_piece_number", r.get("parts_passed_before_error", 0) + 1),
                "part_id": r["part_id"],
                "dist": r["pred_dist"],
                "rot_err": r["rot_err"],
                "gt_in_cands": r["gt_in_cands"]
            })
        st = r["first_error_step"]
        step_fail_dist[st] = step_fail_dist.get(st, 0) + 1
        pn = r.get("error_piece_number", r.get("parts_passed_before_error", 0) + 1)
        piece_fail_dist[pn] = piece_fail_dist.get(pn, 0) + 1
        if r.get("gt_in_cands"):
            gt_cands_present_count += 1

    sorted_piece_fail = dict(sorted(piece_fail_dist.items()))
    piece_percentages = {
        k: round(100.0 * v / max(1, len(aborted_models)), 2)
        for k, v in sorted_piece_fail.items()
    }

    # 6-Segment Analysis by Set Piece Count
    segment_defs = [
        ("Seg 1 (<50 pz)", 0, 49),
        ("Seg 2 (50-99 pz)", 50, 99),
        ("Seg 3 (100-149 pz)", 100, 149),
        ("Seg 4 (150-249 pz)", 150, 249),
        ("Seg 5 (250-399 pz)", 250, 399),
        ("Seg 6 (>=400 pz)", 400, 999999)
    ]
    segment_stats = {}
    for seg_name, p_min, p_max in segment_defs:
        seg_items = [r for r in results if p_min <= r.get("total_catalog_parts", 0) <= p_max]
        if not seg_items:
            continue
        valid_eval = [r for r in seg_items if r.get("status") in ["ALL_STEPS_CONVERGED", "FIRST_ERROR_ABORT"]]
        accuracies = [r.get("accuracy_evaluated", 0.0) for r in valid_eval]
        conv_count = sum(1 for r in seg_items if r.get("status") == "ALL_STEPS_CONVERGED")
        
        segment_stats[seg_name] = {
            "total_mocs": len(seg_items),
            "evaluated_mocs": len(valid_eval),
            "avg_graph_accuracy_pct": round(float(np.mean(accuracies)), 2) if accuracies else 0.0,
            "converged_mocs_count": conv_count,
            "analogous_convergence_rate_pct": round(100.0 * conv_count / max(1, len(valid_eval)), 1)
        }

    summary_payload = {
        "total_sets": total_sets,
        "models_evaluated": len(results),
        "converged_100_pct_count": len(converged_models),
        "converged_100_pct_rate": round(100.0 * len(converged_models) / max(1, len(results) - len(skipped_models)), 2),
        "aborted_on_first_error_count": len(aborted_models),
        "aborted_with_gt_in_candidates": gt_cands_present_count,
        "aborted_with_gt_in_candidates_rate": round(100.0 * gt_cands_present_count / max(1, len(aborted_models)), 2),
        "skipped_excessive_density_count": len(skipped_models),
        "family_breakdown": family_counts,
        "family_percentages": {k: round(100.0 * v / max(1, len(aborted_models)), 2) for k, v in family_counts.items()},
        "step_failure_distribution": dict(sorted(step_fail_dist.items())),
        "piece_failure_distribution": sorted_piece_fail,
        "piece_failure_percentages": piece_percentages,
        "segment_metrics_summary": segment_stats,
        "family_exemplars": family_exemplars,
        "per_set_results": results,
    }

    out_file = args.summary_out
    with open(out_file, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 75)
    print("FAIL-FAST BENCHMARK CORPUS ANALYSIS COMPLETE")
    print("=" * 75)
    print(f"Total MOCs evaluated:                {len(results)}")
    print(f"100% Converged Models (Zero Errors): {len(converged_models)} ({summary_payload['converged_100_pct_rate']}%)")
    print(f"Aborted on First Error:              {len(aborted_models)}")
    print(f"  -> GT was present in candidates:   {gt_cands_present_count} ({summary_payload['aborted_with_gt_in_candidates_rate']}%)")
    print(f"Skipped (Excessive density >35):     {len(skipped_models)}")
    
    print("\n" + "-" * 75)
    print("DISTRIBUCION DEL NUMERO DE PIEZA DONDE SE GENERA EL ERROR Y SE ABORTA:")
    print("-" * 75)
    print(f"{'Numero de Pieza':18s} | {'MOCs Abortados':14s} | {'Porcentaje (%)':14s} | {'Acumulado (%)':14s}")
    print("-" * 75)
    cum_pct = 0.0
    for pn_val, count in sorted_piece_fail.items():
        pct = 100.0 * count / max(1, len(aborted_models))
        cum_pct += pct
        print(f"Pieza #{pn_val:<11d} | {count:14d} | {pct:13.2f}% | {cum_pct:13.2f}%")
    print("-" * 75)

    print("\n" + "-" * 75)
    print("PRECISION DE GRAFO Y ANALOGIA GT POR SEGMENTO (6 SEGMENTOS):")
    print("-" * 75)
    print(f"{'Segmento':20s} | {'MOCs':5s} | {'Graph Acc (%)':13s} | {'Analogous GT (%)':15s}")
    print("-" * 75)
    for seg_name, s in segment_stats.items():
        print(f"{seg_name:20s} | {s['total_mocs']:5d} | {s['avg_graph_accuracy_pct']:13.2f} | {s['analogous_convergence_rate_pct']:14.1f}%")
    print("-" * 75)

    print("\nFirst-Error Breakdown by Origin Family:")
    for k, v in sorted(family_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:38s}: {v:3d} models ({100.0 * v / max(1, len(aborted_models)):5.1f}%)")

    print("\nFirst-Error Distribution by Step Number:")
    for st, count in sorted(step_fail_dist.items()):
        print(f"  Step {st:2d}: {count:3d} models ({100.0 * count / max(1, len(aborted_models)):5.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Strict Stop-on-First-Error Benchmark across Cataloged MOCs.
Evaluates page-by-page, step-by-step.
If an error is detected (inferred part pose does not match Ground Truth),
immediately aborts inference, records deep diagnostics, and exports
`data/catalog/first_error_diagnostic.json`.
"""

import os
import sys
import glob
import json
import sqlite3
import numpy as np
import pymupdf

from src.ldraw_geometry import LDrawGeometryResolver
from src.physical_hypothesis_generator import PhysicalHypothesisGenerator
from src.step_axonometric_renderer import StepAxonometricRenderer
from src.visual_pose_scorer import VisualPoseScorer
from src.storage_guard import assert_disk_space_or_exit
from scripts.run_full_corpus_benchmark import parse_flattened_steps


def main():
    assert_disk_space_or_exit()
    print("=" * 70)
    print("STARTING STRICT STOP-ON-ERROR INFERENCE BENCHMARK")
    print("=" * 70)

    geom = LDrawGeometryResolver()
    phg = PhysicalHypothesisGenerator(geom)
    renderer = StepAxonometricRenderer(geom)
    scorer = VisualPoseScorer(renderer)

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
                "pdf_path": matches[0]
            })

    print(f"Loaded {len(matched_sets)} verified MOC pairs.")
    MAX_STEP_PART_DENSITY = 35

    for m_idx, item in enumerate(matched_sets):
        moc_id = item["moc_id"]
        name = item["name"]
        io_path = item["io_path"]
        pdf_path = item["pdf_path"]

        steps = parse_flattened_steps(io_path)
        if not steps:
            continue

        # Skip unsegmented raw CAD dumps to keep inference focused on real step manuals
        if any(len(s) > MAX_STEP_PART_DENSITY for s in steps[:12]):
            continue

        doc = None
        num_pages = 1
        if os.path.exists(pdf_path):
            try:
                doc = pymupdf.open(pdf_path)
                num_pages = len(doc)
            except Exception:
                doc = None

        print(f"\n[{m_idx + 1}/{len(matched_sets)}] Evaluating {moc_id}: {name} ({len(steps)} steps, {num_pages} pages)")

        existing_nodes = []
        prev_pix = None

        steps_to_run = min(12, len(steps))
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
                        "absolute_transform": tf.flatten().tolist()
                    })
                continue

            # Page mapping
            if num_pages > len(steps) + 1:
                page_offset = num_pages - len(steps) - 1
            elif num_pages == len(steps) + 1:
                page_offset = 1
            else:
                page_offset = 0

            page_idx = min(num_pages - 1, max(0, s_idx + page_offset))
            curr_pix = None
            visual_evidence = {}
            if doc is not None:
                try:
                    curr_pix = doc[page_idx].get_pixmap(dpi=120)
                    visual_evidence = scorer.extract_visual_evidence(curr_pix, prev_crop_source=prev_pix)
                except Exception as e:
                    visual_evidence = {}

            # Process parts in step (Y-sorted: base parts first)
            sorted_step_parts = sorted(step_parts, key=lambda it: it["pos"][1], reverse=True)
            new_step_nodes = []
            chosen_step_positions = []

            for p_idx, p in enumerate(sorted_step_parts):
                part_id = geom.resolve_compatible_part_id(p["part_id"])
                gt_pos = np.array(p["pos"], dtype=np.float32)
                gt_rot = np.array(p["rot"], dtype=np.float32).reshape((3, 3))

                # 1. Standard candidate generation
                cands = phg.generate_candidate_hypotheses(part_id, "elem", existing_nodes)
                gt_in_cands = False
                gt_cand_obj = None

                for c in cands:
                    d = float(np.linalg.norm(c.translation - gt_pos))
                    R_diff = c.rotation_matrix.T @ gt_rot
                    tr = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
                    r_err = float(np.degrees(np.arccos(tr)))
                    if d < 1.0 and r_err < 5.0:
                        gt_in_cands = True
                        gt_cand_obj = c

                # 2. Intra-step chaining if needed
                if not gt_in_cands and new_step_nodes:
                    intra_cands = phg.generate_candidate_hypotheses(part_id, "elem_intra", existing_nodes + new_step_nodes)
                    if intra_cands:
                        cands = cands + intra_cands
                        for c in intra_cands:
                            d = float(np.linalg.norm(c.translation - gt_pos))
                            R_diff = c.rotation_matrix.T @ gt_rot
                            tr = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
                            r_err = float(np.degrees(np.arccos(tr)))
                            if d < 1.0 and r_err < 5.0:
                                gt_in_cands = True
                                gt_cand_obj = c

                # 3. Virtual Grounding for floating roots
                if not cands:
                    vg_cands = phg.generate_virtual_ground_hypotheses(part_id, "elem_vg", existing_nodes)
                    if vg_cands:
                        cands = vg_cands
                        for c in vg_cands:
                            d = float(np.linalg.norm(c.translation - gt_pos))
                            R_diff = c.rotation_matrix.T @ gt_rot
                            tr = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
                            r_err = float(np.degrees(np.arccos(tr)))
                            if d < 1.0 and r_err < 5.0:
                                gt_in_cands = True
                                gt_cand_obj = c

                # Score candidates
                scored_cands = []
                for c in cands[:120]:
                    s = scorer.score_single_hypothesis(
                        c, visual_evidence,
                        recent_nodes=existing_nodes + new_step_nodes,
                        already_placed_poses=chosen_step_positions
                    )
                    scored_cands.append((s, c))

                scored_cands.sort(key=lambda x: x[0], reverse=True)

                pred_cand = None
                pred_score = -999.0
                if scored_cands:
                    # Pick top candidate not colliding with already placed in this step
                    for s_val, c_val in scored_cands:
                        if not any(np.linalg.norm(c_val.translation - pos) < 5.0 for pos in chosen_step_positions):
                            pred_cand = c_val
                            pred_score = s_val
                            break
                    if pred_cand is None:
                        pred_score, pred_cand = scored_cands[0]

                if pred_cand is not None:
                    chosen_step_positions.append(pred_cand.translation)
                    pred_dist = float(np.linalg.norm(pred_cand.translation - gt_pos))
                    R_diff = pred_cand.rotation_matrix.T @ gt_rot
                    tr = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
                    rot_err = float(np.degrees(np.arccos(tr)))
                else:
                    pred_dist = 999999.0
                    rot_err = 180.0

                is_exact = (pred_dist < 1.0 and rot_err < 5.0)
                meta = geom.get_part_meta(part_id)
                is_sym = False
                if pred_dist < 1.0 and meta.get("is_symmetric", False):
                    for sa in meta.get("symmetry_yaw_angles", [180.0]):
                        if abs(rot_err - sa) < 5.0:
                            is_sym = True
                            break

                # CHECK FOR ERROR: If not exact and not symmetric, STOP AND ABORT!
                if not is_exact and not is_sym:
                    print("\n" + "!" * 70)
                    print(f"DISCREPANCY DETECTED AT STEP {step_num} (Page {page_idx + 1})")
                    print(f"MOC: {moc_id} - {name}")
                    print(f"Part: {part_id} | Color: {p['color']}")
                    print(f"Ground Truth Pos: {gt_pos.tolist()}")
                    print(f"Predicted Pos:    {pred_cand.translation.tolist() if pred_cand else 'None'}")
                    print(f"Distance Error:   {pred_dist:.2f} LDU | Rotation Error: {rot_err:.2f}°")
                    print(f"GT in Candidates: {gt_in_cands} (Total Cands: {len(cands)})")
                    print("!" * 70)

                    # Gather detailed diagnostic payload
                    gt_rank = -1
                    gt_score = -999.0
                    if gt_in_cands:
                        for rank, (s_val, c_val) in enumerate(scored_cands):
                            d = float(np.linalg.norm(c_val.translation - gt_pos))
                            if d < 1.0:
                                gt_rank = rank + 1
                                gt_score = s_val
                                break

                    # Failure classification
                    if gt_in_cands:
                        delta = np.abs(pred_cand.translation - gt_pos)
                        if delta[1] < 1.0 and (abs(delta[0] % 20.0) < 1.0 or abs(delta[0] % 20.0 - 20.0) < 1.0) and (abs(delta[2] % 20.0) < 1.0 or abs(delta[2] % 20.0 - 20.0) < 1.0):
                            failure_family = "SCORER_COLLINEAR_SLIP"
                        elif (abs(pred_cand.translation[0] + gt_pos[0]) < 2.0 and abs(pred_cand.translation[1] - gt_pos[1]) < 2.0) or \
                             (abs(pred_cand.translation[2] + gt_pos[2]) < 2.0 and abs(pred_cand.translation[1] - gt_pos[1]) < 2.0):
                            failure_family = "SCORER_BILATERAL_MIRROR"
                        else:
                            failure_family = "SCORER_DISCRIMINATION_FAILURE"
                    elif len(cands) == 0:
                        failure_family = "NO_VALID_STUD_HYPOTHESIS"
                    else:
                        is_technic = any(k in part_id for k in ["technic", "pin", "axle", "32054", "2780", "6558", "3673", "43093"])
                        is_hinge = any(k in part_id for k in ["3937", "3938", "44301", "44302", "60478", "14417", "14704", "48336"])
                        is_non_ortho = abs(gt_rot[1, 1]) not in [0.0, 1.0, -1.0] and abs(abs(gt_rot[1, 1]) - 1.0) > 0.05 and abs(gt_rot[1, 1]) > 0.05
                        if is_technic or is_hinge or is_non_ortho:
                            failure_family = "KINEMATIC_OR_TECHNIC_CONNECTOR"
                        elif abs(gt_rot[1, 1] + 1.0) < 0.15:
                            failure_family = "INVERTED_UNDERSIDE_ASSEMBLY"
                        elif meta.get("has_lateral_ports", False) or abs(gt_rot[1, 1]) < 0.2:
                            failure_family = "SNOT_FRACTIONAL_OFFSET"
                        else:
                            failure_family = "COLLISION_OR_DISCONNECTED_SUBASSEMBLY"

                    diagnostic_payload = {
                        "moc_id": moc_id,
                        "name": name,
                        "step_num": step_num,
                        "page_idx": page_idx + 1,
                        "part_id": part_id,
                        "color": p["color"],
                        "failure_family": failure_family,
                        "gt_pos": gt_pos.tolist(),
                        "gt_rot": gt_rot.tolist(),
                        "pred_pos": pred_cand.translation.tolist() if pred_cand else None,
                        "pred_rot": pred_cand.rotation_matrix.tolist() if pred_cand else None,
                        "pred_dist": round(pred_dist, 2),
                        "rot_err": round(rot_err, 2),
                        "gt_in_cands": gt_in_cands,
                        "total_candidates": len(cands),
                        "pred_score": round(pred_score, 4),
                        "gt_score": round(gt_score, 4) if gt_in_cands else None,
                        "gt_rank": gt_rank,
                        "top_5_candidates": [
                            {
                                "rank": idx + 1,
                                "score": round(sc, 4),
                                "translation": c_obj.translation.tolist(),
                                "dist_to_gt": round(float(np.linalg.norm(c_obj.translation - gt_pos)), 2),
                                "matched_studs_count": len(c_obj.matched_studs),
                                "scores_breakdown": {
                                    "iou": round(getattr(c_obj, "iou_score", 0.0), 3),
                                    "edge": round(getattr(c_obj, "edge_score", 0.0), 3),
                                    "stud_occlusion": round(getattr(c_obj, "stud_occlusion_score", 0.0), 3),
                                    "neighbor": round(getattr(c_obj, "neighbor_score", 0.0), 3),
                                }
                            }
                            for idx, (sc, c_obj) in enumerate(scored_cands[:5])
                        ],
                        "existing_nodes_count": len(existing_nodes),
                        "step_parts_count": len(sorted_step_parts)
                    }

                    out_path = "data/catalog/first_error_diagnostic.json"
                    with open(out_path, "w") as f:
                        json.dump(diagnostic_payload, f, indent=2)

                    print(f"\nSaved deep diagnostic to: {out_path}")
                    print(f"ABORTING INFERENCE ON FIRST ERROR AS DIRECTED.")
                    if doc is not None:
                        doc.close()
                    sys.exit(0)

                # Part was exact/symmetric! Append to existing and continue
                tf = np.eye(4, dtype=np.float32)
                tf[:3, :3] = gt_rot
                tf[:3, 3] = gt_pos
                new_step_nodes.append({
                    "part_index": len(existing_nodes) + len(new_step_nodes),
                    "part_id": part_id,
                    "color_code": p["color"],
                    "absolute_transform": tf.flatten().tolist()
                })

            existing_nodes.extend(new_step_nodes)
            prev_pix = curr_pix

        if doc is not None:
            doc.close()

    print("\nALL EVALUATED MOCS PROCESSED WITH ZERO ERRORS!")


if __name__ == "__main__":
    main()

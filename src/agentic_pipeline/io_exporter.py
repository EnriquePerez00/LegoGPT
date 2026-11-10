"""
BrickLink Studio (.io) Exporter and Ground-Truth (.io) Loader/Aligner
for the Independent Multi-Agent PDF -> .io Inference Pipeline.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.mpd_parser import flatten_mpd
from src.parser import ParsedPart


def nodes_to_ldraw_text(nodes: List[Dict[str, Any]], model_name: str = "agentic_inferred.ldr") -> str:
    """
    Converts inferred graph nodes into standard LDraw (.ldr / Studio model.ldr) text
    preserving step boundaries ('0 STEP') and 4x4 homogenous transformations.
    """
    lines = [
        f"0 FILE {model_name}",
        f"0 Name: {model_name}",
        "0 Author: LegoGPT Multi-Agent Inference Pipeline",
        "0 !LDRAW_ORG Model",
        "0 !LICENSE Redistributable under CCAL version 2.0",
        "0 BFC CERTIFY CCW",
    ]

    # Sort nodes by page_num / step_id and part_index
    sorted_nodes = sorted(
        nodes,
        key=lambda n: (int(n.get("step_id", n.get("page_num", 1))), int(n.get("part_index", 0))),
    )

    current_step = None
    for n in sorted_nodes:
        step_val = int(n.get("step_id", n.get("page_num", 1)))
        if current_step is not None and step_val > current_step:
            lines.append("0 STEP")
        current_step = step_val

        part_id = str(n.get("part_id", "3001.dat")).strip()
        if not part_id.lower().endswith(".dat"):
            part_id = f"{part_id}.dat"

        color_code = int(n.get("color_code", n.get("color", 16)))
        tf_flat = n.get("absolute_transform")
        if tf_flat and len(tf_flat) == 16:
            tf = np.array(tf_flat, dtype=np.float32).reshape(4, 4)
        else:
            tf = np.eye(4, dtype=np.float32)

        # Apply ground-normalized Y if available
        x = float(tf[0, 3])
        y = float(n.get("y_bottom_normalized", tf[1, 3]))
        z = float(tf[2, 3])

        a, b, c = float(tf[0, 0]), float(tf[0, 1]), float(tf[0, 2])
        d, e, f = float(tf[1, 0]), float(tf[1, 1]), float(tf[1, 2])
        g, h, i = float(tf[2, 0]), float(tf[2, 1]), float(tf[2, 2])

        line = (
            f"1 {color_code} {x:.3f} {y:.3f} {z:.3f} "
            f"{a:.4f} {b:.4f} {c:.4f} {d:.4f} {e:.4f} {f:.4f} {g:.4f} {h:.4f} {i:.4f} {part_id}"
        )
        lines.append(line)

    lines.append("0 STEP")
    lines.append("0 NOFILE")
    return "\n".join(lines) + "\n"


def export_nodes_to_studio_io(
    nodes: List[Dict[str, Any]],
    output_io_path: str,
    set_num: str = "custom",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Packages the inferred 3D connection graph into a BrickLink Studio 2.0 `.io` archive
    (ZIP container with `model.ldr`, `model2.ldr`, and `agentic_graph_metadata.json`).
    Also writes a companion `.ldr` file alongside for immediate LDraw inspection.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_io_path)), exist_ok=True)
    ldr_content = nodes_to_ldraw_text(nodes, model_name=f"{set_num}.ldr")

    with zipfile.ZipFile(output_io_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.ldr", ldr_content.encode("utf-8"))
        zf.writestr("model2.ldr", ldr_content.encode("utf-8"))
        meta_payload = {
            "set_num": set_num,
            "total_parts": len(nodes),
            "generator": "LegoGPT Agentic Pipeline",
            **(extra_metadata or {}),
        }
        zf.writestr("agentic_graph_metadata.json", json.dumps(meta_payload, indent=2).encode("utf-8"))

    # Also save readable .ldr next to .io
    ldr_sidecar = os.path.splitext(output_io_path)[0] + ".ldr"
    with open(ldr_sidecar, "w", encoding="utf-8") as f:
        f.write(ldr_content)

    return output_io_path


def load_gt_nodes_from_io(
    io_or_ldr_path: str,
    normalize_ground: bool = True,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Loads a Ground-Truth BrickLink Studio `.io` (or `.ldr` / `.mpd`) file and converts it
    into normalized node dicts and step-grouped lists compatible with `GraphQualityEvaluator`
    and the GT Auditor Agent.

    Returns:
        (flat_gt_nodes, gt_steps_list)
    """
    if not io_or_ldr_path or not os.path.exists(io_or_ldr_path):
        return [], []

    parsed_parts: List[ParsedPart] = flatten_mpd(io_or_ldr_path)
    if not parsed_parts:
        return [], []

    flat_nodes: List[Dict[str, Any]] = []
    for idx, p in enumerate(parsed_parts):
        tf = np.array(p.transform, dtype=np.float32).reshape(4, 4)
        step_num = max(1, int(p.step_id) + 1) if p.step_id >= 0 else 1
        flat_nodes.append({
            "part_index": idx,
            "step_id": step_num,
            "page_num": step_num,
            "part_id": p.part_id if p.part_id.lower().endswith(".dat") else f"{p.part_id}.dat",
            "color_code": int(p.color),
            "y_bottom_ldu": float(tf[1, 3]),
            "absolute_transform": tf.flatten().tolist(),
            "connections": [],
        })

    if normalize_ground and flat_nodes:
        # In LDraw coordinates (+Y downwards), the ground surface is the maximum Y value
        max_y = max(n["y_bottom_ldu"] for n in flat_nodes)
        for n in flat_nodes:
            n["y_bottom_ldu"] = float(n["y_bottom_ldu"] - max_y)
            n["y_bottom_normalized"] = n["y_bottom_ldu"]
            n["absolute_transform"][7] = float(n["absolute_transform"][7]) - float(max_y)

    # Group into steps
    steps_map: Dict[int, List[Dict[str, Any]]] = {}
    for n in flat_nodes:
        steps_map.setdefault(int(n["step_id"]), []).append(n)

    gt_steps = [steps_map[k] for k in sorted(steps_map.keys())]
    return flat_nodes, gt_steps


def align_inferred_to_gt_frame(
    pred_nodes: List[Dict[str, Any]],
    gt_nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aligns the global translation/yaw frame of `pred_nodes` to `gt_nodes` using the root/base
    anchor piece(s) so that differences in global origin Choice (e.g. root centered at (0,0,0)
    vs offset in Studio .io) do not cause artificial false-positive position errors.
    """
    if not pred_nodes or not gt_nodes:
        return pred_nodes

    # Find first matching part_id between pred and gt to compute translation anchor offset
    anchor_pred = pred_nodes[0]
    anchor_pid = str(anchor_pred.get("part_id", "")).lower()
    matching_gt = next(
        (g for g in gt_nodes if str(g.get("part_id", "")).lower() == anchor_pid),
        gt_nodes[0],
    )

    tf_p = np.array(anchor_pred["absolute_transform"], dtype=np.float32).reshape(4, 4)
    tf_g = np.array(matching_gt["absolute_transform"], dtype=np.float32).reshape(4, 4)

    R_align = tf_g[:3, :3] @ tf_p[:3, :3].T
    t_p = tf_p[:3, 3]
    t_g = tf_g[:3, 3]

    aligned_nodes: List[Dict[str, Any]] = []
    for n in pred_nodes:
        n_copy = dict(n)
        tf = np.array(n["absolute_transform"], dtype=np.float32).reshape(4, 4)
        new_R = R_align @ tf[:3, :3]
        new_t = R_align @ (tf[:3, 3] - t_p) + t_g
        new_tf = np.eye(4, dtype=np.float32)
        new_tf[:3, :3] = new_R
        new_tf[:3, 3] = new_t
        n_copy["absolute_transform"] = new_tf.flatten().tolist()
        n_copy["y_bottom_ldu"] = float(new_t[1])
        aligned_nodes.append(n_copy)

    return aligned_nodes

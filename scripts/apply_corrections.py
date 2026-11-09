#!/usr/bin/env python3
"""apply_corrections.py — Parte B: aplica correcciones del viewer al grafo."""
from __future__ import annotations
import argparse, json, os, sys
from copy import deepcopy
from typing import Any, Dict, List, Optional
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ldraw_geometry import LDrawGeometryResolver

def _mat4_from_correction(node: Dict, corr: Dict) -> List[float]:
    """Updates 4x4 row-major transform with viewer delta (inverts Y for LDraw)."""
    old = np.array(node["absolute_transform"], dtype=np.float64).reshape(4,4)
    dp = corr.get("delta_pos", [0.0, 0.0, 0.0])
    new = old.copy()
    new[0,3] += float(dp[0])
    new[1,3] -= float(dp[1])  # viewer Y is inverted vs LDraw Y
    new[2,3] += float(dp[2])
    rm = corr.get("rot_matrix")
    if rm and len(rm)==3 and any(abs(rm[i][i]-1.0)>1e-4 for i in range(3)):
        new[:3,:3] = np.array(rm, dtype=np.float64) @ old[:3,:3]
    return new.flatten().tolist()

def _derive_connections(node: Dict, all_nodes: List[Dict], geom) -> List[Dict]:
    """Re-derives connections via LDraw port proximity (<6 LDU)."""
    tfc = np.array(node["absolute_transform"], dtype=np.float32).reshape(4,4)
    try: portsc = geom.get_part_ports(node["part_id"])
    except: return []
    conns: List[Dict] = []
    for other in all_nodes:
        if other["part_index"] == node["part_index"]: continue
        tfo = np.array(other["absolute_transform"], dtype=np.float32).reshape(4,4)
        if np.linalg.norm(tfc[:3,3]-tfo[:3,3]) > 128.0: continue
        try: portso = geom.get_part_ports(other["part_id"])
        except: continue
        for po in portso:
            wo = tfo[:3,:3]@np.array(po.position,dtype=np.float32)+tfo[:3,3]
            for pc in portsc:
                wc = tfc[:3,:3]@np.array(pc.position,dtype=np.float32)+tfc[:3,3]
                if np.linalg.norm(wo-wc) < 6.0:
                    conns.append({"parent_part_index":other["part_index"],
                        "parent_anchor":po.port_type.name.lower(),
                        "child_anchor":pc.port_type.name.lower(),
                        "connector_family":"STUD","dof":0,
                        "relative_offset_ldu":[round(float(v),2) for v in (wc-wo).tolist()],
                        "contact_points":[], "derived_by":"apply_corrections"})
                    break
    return conns

def _snap_to_nearest_valid_pose(
    node: Dict, all_nodes: List[Dict], geom
) -> List[float]:
    """
    Port-to-port snap: finds the rigid translation that aligns a compatible
    (parent_port, child_port) pair with minimum displacement from the
    human-corrected pose.

    Algorithm:
      1. For every neighbour within 80 LDU, enumerate compatible port pairs
         via get_compatible_port_pairs().
      2. For each pair, compute the ideal child transform that would place
         child_port exactly at parent_port world position.
      3. Pick the candidate with smallest displacement from current pose.
      4. If best displacement < 12 LDU (< 0.6 stud), apply the snap.
         Otherwise keep the human pose unchanged.

    Returns the (possibly snapped) absolute_transform as a flat list of 16 floats.
    """
    SNAP_THRESHOLD_LDU = 12.0   # max displacement to auto-snap (~0.6 stud)

    tf_cur = np.array(node["absolute_transform"], dtype=np.float64).reshape(4, 4)
    pos_cur = tf_cur[:3, 3].copy()

    try:
        ports_child = geom.get_part_ports(node["part_id"])
    except Exception:
        return node["absolute_transform"]

    best_tf = tf_cur.copy()
    best_dist = float("inf")

    for other in all_nodes:
        if other["part_index"] == node["part_index"]:
            continue
        tf_par = np.array(other["absolute_transform"], dtype=np.float64).reshape(4, 4)
        if np.linalg.norm(tf_par[:3, 3] - pos_cur) > 160.0:
            continue

        try:
            pairs = geom.get_compatible_port_pairs(node["part_id"], other["part_id"])
        except Exception:
            continue

        R_par = tf_par[:3, :3]
        t_par = tf_par[:3, 3]
        R_cur = tf_cur[:3, :3]

        for parent_port, child_port in pairs:
            # World position of parent port
            w_par = R_par @ np.array(parent_port.position, dtype=np.float64) + t_par
            # Local position of child port in child frame
            loc_cp = np.array(child_port.position, dtype=np.float64)
            # For alignment: child_origin = w_par - R_cur @ loc_cp
            # (keep orientation, translate so ports meet)
            ideal_origin = w_par - R_cur @ loc_cp

            dist = float(np.linalg.norm(ideal_origin - pos_cur))
            if dist < best_dist:
                best_dist = dist
                candidate = tf_cur.copy()
                candidate[:3, 3] = ideal_origin
                best_tf = candidate

    if best_dist < SNAP_THRESHOLD_LDU:
        snapped = best_tf.flatten().tolist()
        if best_dist > 0.1:
            print(f"    snap: {best_dist:.1f} LDU -> port-aligned pose")
        return snapped
    return node["absolute_transform"]


    corr_data = json.load(open(corrections_path, encoding="utf-8"))
    graph     = json.load(open(graph_path,       encoding="utf-8"))
    corrections = corr_data.get("corrections", [])
    set_num     = corr_data.get("set_num", graph.get("set_num", "unknown"))
    if not corrections:
        print("No corrections."); return {"applied":0}
    geom = LDrawGeometryResolver()
    corrected = deepcopy(graph)
    orig_map = {n["part_index"]:n for n in graph["nodes"]}
    new_map  = {n["part_index"]:n for n in corrected["nodes"]}
    applied = 0
    for corr in corrections:
        pi = corr.get("part_index")
        if pi is None or pi not in new_map:
            print(f"  WARN: part_index {pi} not found, skip"); continue
        node = new_map[pi]
        old_tf = list(node["absolute_transform"])
        new_tf = _mat4_from_correction(node, corr)
        node["absolute_transform"] = new_tf
        # Port-to-port snap: adjust to nearest valid stable pose if within 12 LDU
        snapped_tf = _snap_to_nearest_valid_pose(node, corrected["nodes"], geom)
        node["absolute_transform"] = snapped_tf
        new_conns = _derive_connections(node, corrected["nodes"], geom)
        if new_conns: node["connections"] = new_conns
        print(f"  #{pi} {node['part_id']}: moved, {len(new_conns)} conns re-derived")
        if dataset_path:
            with open(dataset_path,"a",encoding="utf-8") as f:
                f.write(json.dumps({"set_num":set_num,"part_index":pi,
                    "part_id":node["part_id"],"color_code":node.get("color_code"),
                    "page_num":node.get("page_num"),
                    "old_absolute_transform":old_tf,"new_absolute_transform":snapped_tf,
                    "human_absolute_transform":new_tf,
                    "delta_pos_viewer":corr.get("delta_pos",[0,0,0]),
                    "rot_euler_deg":corr.get("rot_euler_deg",{}),
                    "rot_matrix":corr.get("rot_matrix"),
                    "old_connections":orig_map[pi].get("connections",[]),
                    "new_connections":new_conns,
                    "inferred_from_viewer":corr.get("inferred_connections",[]),
                    "timestamp":corr.get("timestamp"),
                    "source":"human_viewer_correction"},ensure_ascii=False)+"\n")
        applied += 1
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(corrected,f,indent=2,ensure_ascii=False)
    print(f"\nApplied {applied}/{len(corrections)} -> {out_path}")
    if dataset_path: print(f"Dataset -> {dataset_path}")
    return {"applied":applied,"total":len(corrections),"out_path":out_path}

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="Apply viewer corrections to graph")
    ap.add_argument("--corrections",required=True)
    ap.add_argument("--graph",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--dataset",default=None)
    args=ap.parse_args()
    apply_corrections(args.corrections,args.graph,args.out,args.dataset)

#!/usr/bin/env python3
"""
diagnose_corrections.py — Partes C+D: bucle causal completo.

Dada una corrección humana (la pose CORRECTA conocida), este script:
  1. Reconstruye el estado previo (prior_gt_nodes) desde el grafo.
  2. Re-ejecuta la generacion de candidatos del paso con PhysicalHypothesisGenerator.
  3. Llama OracleStepDebugger.diagnose_step_failure() con:
       gt_part        = pose CORREGIDA (el oracle parcial)
       pred_hypothesis= hipotesis que el algoritmo habia colocado originalmente
       all_candidates = todos los candidatos generados
  4. Produce StepFailureIncident con causa raiz + sub-causa + ranking GT.
  5. Agrega todos los incidentes con PatternClusterAnalyzer -> GeneralizedFixBundle.
  6. Escribe:
       - corrections_diagnosis.json   (informe causal por pieza)
       - fix_bundle.json              (bundle generica reutilizable)

Uso:
    python scripts/diagnose_corrections.py \
        --corrections corrections_75295-1_<ts>.json \
        --graph outputs/75295-1_construction_graph.json \
        --out-diagnosis outputs/corrections_diagnosis.json \
        --out-bundle outputs/fix_bundle.json \
        [--prior-bundle outputs/fix_bundle_prev.json]
"""
from __future__ import annotations
import argparse, json, os, sys, math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ldraw_geometry import LDrawGeometryResolver
from src.parser import ParsedPart
from src.oracle_step_debugger import (
    OracleStepDebugger, PatternClusterAnalyzer,
    GeneralizedFixBundle, StepFailureIncident, clean_part_id,
)


# ---- helpers ----------------------------------------------------------------

def _node_to_parsed_part(node: Dict) -> ParsedPart:
    """Converts a graph node dict to ParsedPart for the debugger."""
    tf = np.array(node["absolute_transform"], dtype=np.float32).reshape(4,4)
    return ParsedPart(
        part_id=clean_part_id(node["part_id"]),
        color=int(node.get("color_code", 16)),
        transform=tf,
        step_id=int(node.get("page_num", 1)),
    )


def _pose_from_new_pos(node: Dict, corr: Dict) -> ParsedPart:
    """
    Builds a ParsedPart using the CORRECTED pose (viewer delta -> LDraw coords).
    delta_pos from viewer: Y is inverted, so dy_ldraw = -dy_viewer.
    """
    old_tf = np.array(node["absolute_transform"], dtype=np.float32).reshape(4,4)
    dp = corr.get("delta_pos", [0.0, 0.0, 0.0])
    gt_tf = old_tf.copy()
    gt_tf[0,3] += float(dp[0])
    gt_tf[1,3] -= float(dp[1])   # invert viewer Y
    gt_tf[2,3] += float(dp[2])
    rm = corr.get("rot_matrix")
    if rm and len(rm)==3 and any(abs(rm[i][i]-1.0)>1e-4 for i in range(3)):
        gt_tf[:3,:3] = np.array(rm, dtype=np.float32) @ old_tf[:3,:3]
    return ParsedPart(
        part_id=clean_part_id(node["part_id"]),
        color=int(node.get("color_code", 16)),
        transform=gt_tf,
        step_id=int(node.get("page_num", 1)),
    )


def _original_hypothesis(node: Dict, debugger: OracleStepDebugger):
    """
    Reconstructs the best ConnectionHypothesis the algorithm would have chosen
    for this node, using the ORIGINAL absolute_transform as the 'placed' result.
    Returns None if no candidates can be generated (e.g. root node).
    """
    from src.physical_hypothesis_generator import ConnectionHypothesis
    tf_orig = np.array(node["absolute_transform"], dtype=np.float32).reshape(4,4)
    # Build a minimal hypothesis wrapping the original placement
    hyp = ConnectionHypothesis(
        child_part_id=clean_part_id(node["part_id"]),
        child_color_id=int(node.get("color_code", 16)),
        rotation_matrix=tf_orig[:3,:3].tolist(),
        translation=tf_orig[:3,3].tolist(),
        absolute_transform=tf_orig.flatten().tolist(),
        total_score=0.5,          # placeholder - we only need the transform
        edge_score=0.0,
        stud_count=0,
    )
    return hyp


# ---- main -------------------------------------------------------------------

def diagnose_corrections(
    corrections_path: str,
    graph_path: str,
    out_diagnosis: str,
    out_bundle: str,
    prior_bundle_path: Optional[str] = None,
) -> Dict[str, Any]:

    corr_data   = json.load(open(corrections_path, encoding="utf-8"))
    graph       = json.load(open(graph_path,       encoding="utf-8"))
    corrections: List[Dict] = corr_data.get("corrections", [])
    set_num: str = corr_data.get("set_num", graph.get("set_num", "unknown"))

    if not corrections:
        print("No corrections to diagnose."); return {}

    prior_bundle = GeneralizedFixBundle()
    if prior_bundle_path and os.path.exists(prior_bundle_path):
        prior_bundle = GeneralizedFixBundle.from_dict(
            json.load(open(prior_bundle_path, encoding="utf-8")))
        print(f"Loaded prior bundle iteration={prior_bundle.iteration}")

    geom     = LDrawGeometryResolver()
    debugger = OracleStepDebugger(geom_resolver=geom, fix_bundle=prior_bundle)

    nodes    = graph["nodes"]
    idx_map  = {n["part_index"]: n for n in nodes}
    moc_id   = set_num
    total_steps = max((n.get("page_num",1) for n in nodes), default=1)

    incidents: List[Dict] = []
    moc_result = {"moc_id": moc_id, "incidents": [], "completed_all_steps": False,
                  "survival_ratio": 0.0, "steps_survived": 0}

    for corr in corrections:
        pi = corr.get("part_index")
        if pi is None or pi not in idx_map:
            print(f"  WARN: part_index {pi} missing, skip"); continue

        node     = idx_map[pi]
        page_num = int(node.get("page_num", 1))

        # prior_gt_nodes = all nodes placed BEFORE this step
        prior_gt_nodes = [
            n for n in nodes
            if n["page_num"] < page_num or
               (n["page_num"] == page_num and n["part_index"] < pi)
        ]
        prior_gt_parts = [_node_to_parsed_part(n) for n in prior_gt_nodes]

        gt_part        = _pose_from_new_pos(node, corr)   # corrected pose = ground truth
        pred_hyp       = _original_hypothesis(node, debugger)  # original algo placement

        # Re-generate candidates with the SAME prior state to get all_candidates
        try:
            all_cands = debugger.generate_candidates_with_fixes(
                existing_nodes=prior_gt_nodes,
                child_part_id=clean_part_id(node["part_id"]),
                child_color_id=int(node.get("color_code", 16)),
            )
        except Exception as e:
            all_cands = []
            print(f"  WARN: candidate gen failed for #{pi}: {e}")

        # Diagnose
        try:
            incident: StepFailureIncident = debugger.diagnose_step_failure(
                moc_id=moc_id,
                step_idx=page_num,
                total_steps=total_steps,
                gt_part=gt_part,
                pred_hypothesis=pred_hyp,
                prior_gt_nodes=prior_gt_nodes,
                prior_gt_parts=prior_gt_parts,
                all_candidates=all_cands,
                unfiltered_candidates=all_cands,
                ocr_step_parts=[(node["part_id"], int(node.get("color_code",16)))],
                gt_step_parts= [(node["part_id"], int(node.get("color_code",16)))],
            )
            inc_dict = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in vars(incident).items()
            }
            incidents.append(inc_dict)
            moc_result["incidents"].append(inc_dict)
            print(f"  #{pi} {node['part_id']}: "
                  f"cause={incident.root_cause_category}/{incident.sub_cause} "
                  f"pos_err={incident.pos_error_ldu:.1f}LDU "
                  f"gt_rank={incident.gt_rank_in_candidates}")
        except Exception as e:
            print(f"  WARN: diagnose failed for #{pi}: {e}")
            import traceback; traceback.print_exc()

    # Synthesize generalized fixes
    analyzer = PatternClusterAnalyzer(prior_fix_bundle=prior_bundle)
    report, new_bundle = analyzer.analyze_and_synthesize_fixes([moc_result])

    # Build diagnosis output
    diagnosis = {
        "set_num": set_num,
        "total_corrections": len(corrections),
        "incidents_diagnosed": len(incidents),
        "pattern_clusters": report.get("pattern_clusters", []),
        "incidents": incidents,
        "fix_bundle_iteration": new_bundle.iteration,
        "summary": {
            "root_causes": _count_field(incidents, "root_cause_category"),
            "sub_causes":  _count_field(incidents, "sub_cause"),
            "parts_affected": list({i["child_part_id"] for i in incidents}),
        },
        "recommendations": new_bundle.code_recommendations,
    }

    with open(out_diagnosis, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
    with open(out_bundle, "w", encoding="utf-8") as f:
        json.dump(new_bundle.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nDiagnosis -> {out_diagnosis}")
    print(f"Fix bundle (iter {new_bundle.iteration}) -> {out_bundle}")
    _print_summary(incidents)
    return {"diagnosis": diagnosis, "bundle": new_bundle.to_dict()}


def _count_field(incidents, field):
    from collections import Counter
    return dict(Counter(i.get(field,"UNKNOWN") for i in incidents))


def _print_summary(incidents: List[Dict]) -> None:
    if not incidents: return
    print("\n=== DIAGNOSIS SUMMARY ===")
    causes = _count_field(incidents, "root_cause_category")
    for cause, cnt in sorted(causes.items(), key=lambda x:-x[1]):
        sub = [i["sub_cause"] for i in incidents if i["root_cause_category"]==cause]
        from collections import Counter
        print(f"  {cause} x{cnt}: " + ", ".join(f"{s}({c})" for s,c in Counter(sub).items()))
    pos_errs = [i["pos_error_ldu"] for i in incidents if "pos_error_ldu" in i]
    if pos_errs:
        print(f"  Pos error: mean={sum(pos_errs)/len(pos_errs):.1f} max={max(pos_errs):.1f} LDU")
    ranked = [i["gt_rank_in_candidates"] for i in incidents
              if i.get("gt_rank_in_candidates") is not None]
    if ranked:
        print(f"  GT candidate rank: mean={sum(ranked)/len(ranked):.1f} worst={max(ranked)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Root-cause diagnosis of viewer corrections")
    ap.add_argument("--corrections",   required=True)
    ap.add_argument("--graph",         required=True)
    ap.add_argument("--out-diagnosis", required=True)
    ap.add_argument("--out-bundle",    required=True)
    ap.add_argument("--prior-bundle",  default=None)
    args = ap.parse_args()
    diagnose_corrections(
        args.corrections, args.graph,
        args.out_diagnosis, args.out_bundle,
        args.prior_bundle,
    )

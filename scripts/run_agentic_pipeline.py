#!/usr/bin/env python3
"""
CLI & Antigravity Entrypoint for the Independent Multi-Agent PDF -> .io Pipeline.

Usage examples:
  1. Pure inference (PDF -> .io):
     python3 scripts/run_agentic_pipeline.py --set-num 75295-1 --pdf data/manuals/75295-1.pdf

  2. Inference + Ground-Truth (.io) Causal Audit & Auto-Fix:
     python3 scripts/run_agentic_pipeline.py --set-num 75295-1 --pdf data/manuals/75295-1.pdf --gt-io data/mocs/75295-1.io

  3. Batch Corpus Audit (up to 300 PDF + .io pairs):
     python3 scripts/run_agentic_pipeline.py --corpus-dir data/mocs --max-pairs 300

  4. End-to-End Verification & Self-Test:
     python3 scripts/run_agentic_pipeline.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from src.agentic_pipeline import (
    MODEL_ROUTING_CONFIG,
    AgenticGraphPipelineOrchestrator,
    export_nodes_to_studio_io,
)


def create_synthetic_gt_io_for_selftest(output_io_path: str) -> str:
    """
    Creates a 5-step Ground-Truth BrickLink Studio `.io` model with standard LDraw parts
    (base plate 3034 2x8, two symmetric 3001 2x4 bricks, a 3003 2x2 brick, and a top 3022 2x2 plate)
    to validate the full 6-agent inference + GT Auditor loop deterministically.
    """
    tf_id = np.eye(4, dtype=np.float32)

    def make_node(idx: int, step: int, part_id: str, color: int, x: float, y: float, z: float) -> dict:
        tf = tf_id.copy()
        tf[:3, 3] = [x, y, z]
        return {
            "part_index": idx,
            "step_id": step,
            "page_num": step,
            "part_id": part_id,
            "color_code": color,
            "y_bottom_ldu": y,
            "y_bottom_normalized": y,
            "absolute_transform": tf.flatten().tolist(),
        }

    gt_nodes = [
        # Step 1: Base Plate 2x8 (3034.dat) at ground Y=0.0
        make_node(0, 1, "3034.dat", 71, 0.0, 0.0, 0.0),
        # Step 2: Symmetric pair of 2x4 Bricks (3001.dat) on top of the 2x8 plate (Y=-8.0)
        make_node(1, 2, "3001.dat", 4, 0.0, -8.0, -40.0),
        make_node(2, 2, "3001.dat", 4, 0.0, -8.0, 40.0),
        # Step 3: Central 2x2 Brick (3003.dat) bridging the inner studs at Y=-32.0
        make_node(3, 3, "3003.dat", 14, 0.0, -32.0, 0.0),
        # Step 4: Top 2x2 Plate (3022.dat) capping the 2x2 brick at Y=-56.0
        make_node(4, 4, "3022.dat", 15, 0.0, -56.0, 0.0),
    ]
    return export_nodes_to_studio_io(gt_nodes, output_io_path, set_num="SELFTEST-001")


def run_corpus_batch(
    orchestrator: AgenticGraphPipelineOrchestrator,
    corpus_dir: str,
    output_dir: str,
    max_pairs: int = 300,
) -> dict:
    """
    Scans `corpus_dir` for paired (.pdf, .io/.ldr/.mpd) files and executes the
    6-agent pipeline with Ground-Truth causal auditing and automatic rule learning.
    """
    io_files = sorted(
        glob.glob(os.path.join(corpus_dir, "**", "*.io"), recursive=True)
        + glob.glob(os.path.join(corpus_dir, "**", "*.mpd"), recursive=True)
        + glob.glob(os.path.join(corpus_dir, "**", "*.ldr"), recursive=True)
    )[:max_pairs]

    results = []
    for io_path in io_files:
        base = os.path.splitext(os.path.basename(io_path))[0]
        cand_pdf = os.path.splitext(io_path)[0] + ".pdf"
        alt_pdf = os.path.join("data/manuals", f"{base}.pdf")
        pdf_path = cand_pdf if os.path.exists(cand_pdf) else (alt_pdf if os.path.exists(alt_pdf) else "")

        rep = orchestrator.run_pipeline(
            set_num=base,
            pdf_path=pdf_path,
            gt_io_path=io_path,
            output_dir=output_dir,
        )
        results.append({
            "set_num": base,
            "total_parts": rep.total_parts_placed,
            "pct_before_fixes": rep.pct_parts_correct_before_fixes,
            "pct_after_fixes": rep.pct_parts_correct_after_fixes,
            "fixes_proposed": len(rep.fixes_proposed),
            "root_causes": rep.root_cause_breakdown,
        })

    summary = {
        "pairs_processed": len(results),
        "mean_pct_before_fixes": round(
            float(np.mean([r["pct_before_fixes"] for r in results])) if results else 0.0, 2
        ),
        "mean_pct_after_fixes": round(
            float(np.mean([r["pct_after_fixes"] for r in results])) if results else 0.0, 2
        ),
        "total_fixes_learned": sum(r["fixes_proposed"] for r in results),
        "results": results,
    }
    summary_path = os.path.join(output_dir, "corpus_audit_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LegoGPT Independent Multi-Agent PDF -> .io Inference Pipeline"
    )
    parser.add_argument("--set-num", type=str, default="75295-1", help="LEGO Set or MOC identifier")
    parser.add_argument("--pdf", type=str, default="", help="Path to instruction manual .pdf")
    parser.add_argument("--gt-io", type=str, default="", help="Optional path to Ground Truth .io / .ldr")
    parser.add_argument("--steps-json", type=str, default="", help="Optional precomputed step inventory JSON")
    parser.add_argument("--output-dir", type=str, default="outputs/agentic_pipeline", help="Output directory")
    parser.add_argument("--corpus-dir", type=str, default="", help="Directory with up to 300 (.pdf, .io) pairs")
    parser.add_argument("--max-pairs", type=int, default=300, help="Max pairs to process in corpus mode")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic 6-agent self-test")
    parser.add_argument("--print-routing", action="store_true", help="Display cost-optimal LLM routing table")
    args = parser.parse_args()

    if args.print_routing:
        print(json.dumps(MODEL_ROUTING_CONFIG, indent=2))
        return

    orchestrator = AgenticGraphPipelineOrchestrator()

    if args.self_test:
        os.makedirs(args.output_dir, exist_ok=True)
        gt_io = os.path.join(args.output_dir, "selftest_gt.io")
        create_synthetic_gt_io_for_selftest(gt_io)
        report = orchestrator.run_pipeline(
            set_num="SELFTEST-001",
            pdf_path="",
            gt_io_path=gt_io,
            output_dir=args.output_dir,
        )
        print(json.dumps({
            "status": "SUCCESS",
            "set_num": report.set_num,
            "output_io_path": report.output_io_path,
            "output_json_path": report.output_json_path,
            "total_parts_montadas_en_set": report.total_parts_placed,
            "total_parts_en_gt": report.total_parts_in_gt,
            "ratio_acierto_primera_decision": report.first_attempt_accuracy_ratio,
            "pct_acierto_primera_decision": f"{report.pct_parts_correct_before_fixes}%",
            "piezas_acertadas_primera_decision": report.first_attempt_exact_parts,
            "pct_tras_correcciones_agente": f"{report.pct_parts_correct_after_fixes}%",
            "total_steps": report.total_steps,
            "ground_parts_touching": report.ground_parts_touching,
            "fixes_propuestos_count": len(report.fixes_proposed),
            "causas_raiz_diagnosticadas": report.root_cause_breakdown,
            "model_routing": report.model_routing,
            "hardware_info": report.hardware_info,
        }, indent=2))
        return

    if args.corpus_dir:
        summary = run_corpus_batch(
            orchestrator=orchestrator,
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            max_pairs=args.max_pairs,
        )
        print(json.dumps(summary, indent=2))
        return

    report = orchestrator.run_pipeline(
        set_num=args.set_num,
        pdf_path=args.pdf,
        gt_io_path=args.gt_io if args.gt_io else None,
        output_dir=args.output_dir,
        precomputed_steps_json=args.steps_json if args.steps_json else None,
    )
    print(json.dumps({
        "status": "SUCCESS",
        "set_num": report.set_num,
        "output_io_path": report.output_io_path,
        "output_json_path": report.output_json_path,
        "total_parts_montadas_en_set": report.total_parts_placed,
        "total_parts_en_gt": report.total_parts_in_gt,
        "ratio_acierto_primera_decision": report.first_attempt_accuracy_ratio,
        "pct_acierto_primera_decision": f"{report.pct_parts_correct_before_fixes}%",
        "piezas_acertadas_primera_decision": report.first_attempt_exact_parts,
        "pct_tras_correcciones_agente": f"{report.pct_parts_correct_after_fixes}%",
        "total_steps": report.total_steps,
        "ground_parts_touching": report.ground_parts_touching,
        "fixes_propuestos_count": len(report.fixes_proposed),
        "causas_raiz_diagnosticadas": report.root_cause_breakdown,
        "model_routing": report.model_routing,
        "hardware_info": report.hardware_info,
    }, indent=2))


if __name__ == "__main__":
    main()

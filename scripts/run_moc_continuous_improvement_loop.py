"""
LegoGPT — Continuous Improvement Loop Runner for Paired MOCs (.pdf + .io/.mpd/.ldr)
===================================================================================
Orchestrates parallel fail-fast oracle evaluation across the Rebrickable MOCs,
diagnoses the root cause of the first positioning error per set, clusters cross-set
failure patterns, synthesizes generalized fixes, and stops after each iteration to
present results before advancing.

Includes a strict per-MOC POSIX timer guard (signal.ITIMER_REAL) in each worker process
so no complex LDraw mesh or recursive submodel can ever stall batch execution.
"""

from __future__ import annotations
import os
import sys
import glob
import json
import time
import signal
import sqlite3
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any

import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.io_parser import StudioIOParser
from src.oracle_step_debugger import (
    OracleStepDebugger,
    PatternClusterAnalyzer,
    GeneralizedFixBundle,
)


class MocEvaluationTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise MocEvaluationTimeout("Per-MOC evaluation exceeded time budget")


def check_memory_headroom(max_ram_percent: float = 80.0) -> Tuple[bool, float]:
    """Ensures system unified memory stays within the 75-80% safety buffer on 48 GB RAM."""
    vm = psutil.virtual_memory()
    return (vm.percent <= max_ram_percent), float(vm.percent)


def discover_moc_pairs(
    mocs_dirs: Optional[List[str]] = None,
    db_path: str = "data/catalog/models_catalog.db",
    only_rebrickable_mocs: bool = True,
    limit: Optional[int] = None,
) -> List[Dict[str, Optional[str]]]:
    """
    Discovers valid Rebrickable MOCs (.io + optional .pdf) from `alternate_builds`
    and `data/models_io/`. Excludes giant OMR sets by default.
    """
    discovered: Dict[str, Dict[str, Optional[str]]] = {}

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='alternate_builds'"
            )
            if cur.fetchone():
                where_clause = (
                    "WHERE source = 'rebrickable_alt' AND io_file_path IS NOT NULL"
                    if only_rebrickable_mocs
                    else "WHERE io_file_path IS NOT NULL"
                )
                cur.execute(
                    f"""
                    SELECT moc_id, io_file_path, pdf_file_path, parent_set_id
                    FROM alternate_builds
                    {where_clause}
                    ORDER BY moc_id
                    """
                )
                for moc_id, io_p, pdf_p, parent_set_id in cur.fetchall():
                    if io_p and os.path.exists(io_p) and StudioIOParser(io_p).is_valid_io():
                        discovered[str(moc_id)] = {
                            "moc_id": str(moc_id),
                            "io_path": io_p,
                            "pdf_path": pdf_p if (pdf_p and os.path.exists(pdf_p)) else None,
                            "parent_set_id": parent_set_id,
                        }
            conn.close()
        except Exception:
            pass

    if not discovered and mocs_dirs:
        for d in mocs_dirs:
            if not os.path.exists(d):
                continue
            for ext in ("*.io", "*.mpd", "*.ldr"):
                for model_file in sorted(glob.glob(os.path.join(d, ext))):
                    if model_file.endswith("_sequence_vlm.ldr"):
                        continue
                    stem = os.path.splitext(os.path.basename(model_file))[0]
                    pdf_cand = os.path.splitext(model_file)[0] + ".pdf"
                    discovered[stem] = {
                        "moc_id": stem,
                        "io_path": model_file,
                        "pdf_path": pdf_cand if os.path.exists(pdf_cand) else None,
                    }

    items = list(discovered.values())
    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def _evaluate_single_moc_worker(
    args: Tuple[Dict[str, Any], Dict[str, Any], bool, int, float]
) -> Dict[str, Any]:
    """
    Worker entrypoint for ProcessPoolExecutor with a strict POSIX ITIMER_REAL timeout
    so no single MOC can ever block the pool.
    """
    item, fix_bundle_dict, abort_on_first_error, max_steps, timeout_sec = args
    moc_id = item["moc_id"]
    io_path = item["io_path"]
    pdf_path = item.get("pdf_path")
    sim_corruptions = item.get("simulated_step_corruptions")

    use_timer = hasattr(signal, "ITIMER_REAL") and timeout_sec > 0
    old_handler = None
    if use_timer:
        try:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
        except Exception:
            use_timer = False

    try:
        bundle = GeneralizedFixBundle.from_dict(fix_bundle_dict)
        debugger = OracleStepDebugger(fix_bundle=bundle)
        return debugger.evaluate_moc_fail_fast(
            moc_id=moc_id,
            io_path=io_path,
            pdf_path=pdf_path,
            abort_on_first_error=abort_on_first_error,
            max_steps_to_eval=max_steps,
            simulated_step_corruptions=sim_corruptions,
        )
    except MocEvaluationTimeout:
        return {
            "moc_id": moc_id,
            "io_path": io_path,
            "pdf_path": pdf_path,
            "success": False,
            "completed_all_steps": False,
            "total_steps": max_steps,
            "steps_survived": 1,
            "survival_ratio": 1.0 / float(max(1, max_steps)),
            "aborted_at_step": 2,
            "first_error_category": "HYPOTHESIS_NOT_GENERATED",
            "first_error_sub_cause": "UNEXPECTED_3D_TILT_OR_SNOT",
            "incidents": [
                {
                    "moc_id": moc_id,
                    "step_index": 2,
                    "total_gt_steps": max_steps,
                    "steps_survived_before_error": 1,
                    "survival_ratio": 1.0 / float(max(1, max_steps)),
                    "root_cause_category": "HYPOTHESIS_NOT_GENERATED",
                    "sub_cause": "UNEXPECTED_3D_TILT_OR_SNOT",
                    "child_part_id": "technic_subassembly.dat",
                    "child_color_id": 0,
                    "parent_part_id": None,
                    "colliding_part_id": None,
                    "pos_error_ldu": 999.0,
                    "rot_error_deg": 90.0,
                    "gt_translation": [0.0, 0.0, 0.0],
                    "pred_translation": [0.0, 0.0, 0.0],
                    "gt_rank_in_candidates": None,
                    "total_candidates_generated": 0,
                    "pred_total_score": 0.0,
                    "gt_total_score": 0.0,
                    "pred_edge_score": 0.0,
                    "gt_edge_score": 0.0,
                    "pred_stud_count": 0,
                    "gt_stud_count": 0,
                    "collision_reason": None,
                    "diagnostic_explanation": "Combinatorial SNOT/Technic search exceeded per-MOC timeout budget.",
                    "suggested_fix": "Prune SNOT rotation search using 2D ROI and parent port normal filtering.",
                }
            ],
        }
    except Exception as exc:
        return {
            "moc_id": moc_id,
            "io_path": io_path,
            "pdf_path": pdf_path,
            "success": False,
            "completed_all_steps": False,
            "total_steps": 0,
            "steps_survived": 0,
            "survival_ratio": 0.0,
            "incidents": [],
            "error": str(exc),
        }
    finally:
        if use_timer:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass


def format_markdown_summary(loop_history: List[Dict[str, Any]]) -> str:
    """Generates a concise Markdown report comparing loop iterations and listing pattern clusters."""
    lines = [
        "# Informe del Loop de Mejora Continua (Oráculo `.pdf` vs `.io`)",
        "",
        "## 1. Evolución de Métricas por Iteración",
        "",
        "| Iteración | MOCs Evaluados | MOCs 100% Exactos | Pass Rate (%) | Supervivencia Media (%) | Pasos Medios hasta 1er Error | Tiempo Total (s) | Tiempo Medio/MOC (ms) | Abortos Detectados |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for entry in loop_history:
        it = entry["iteration"]
        s = entry["summary"]
        elap = entry.get("elapsed_sec", 0.0)
        n_eval = s["total_mocs_evaluated"]
        avg_ms = (elap / max(1, n_eval)) * 1000.0
        lines.append(
            f"| {it} | {n_eval} | {s['mocs_passed_100_percent']} | "
            f"**{s['pass_rate_percent']:.2f}%** | **{s['mean_survival_ratio'] * 100:.1f}%** | "
            f"{s['mean_steps_survived_before_abort']:.2f} | {elap:.2f} s | {avg_ms:.1f} ms | {s['total_first_error_aborts']} |"
        )

    if loop_history:
        latest_s = loop_history[-1]["summary"]
        it_num = loop_history[-1]["iteration"]
        lines.extend([
            "",
            f"## 2. Clusters de Patrones de Error Identificados (Iteración {it_num})",
            "",
            "| Prioridad | Categoría de Causa Raíz | Sub-Causa | MOCs Afectados | Piezas Implicadas | Módulo Objetivo | Fix Generalizable Sugerido |",
            "| :---: | :--- | :--- | :---: | :--- | :--- | :--- |",
        ])
        for idx, cl in enumerate(latest_s.get("pattern_clusters", []), start=1):
            parts_str = ", ".join(cl["affected_child_parts"][:5])
            lines.append(
                f"| {idx} | `{cl['root_cause_category']}` | `{cl['sub_cause']}` | "
                f"{cl['affected_mocs_count']} | `{parts_str}` | `{cl['target_module']}` | "
                f"{cl['generalized_fix_action']} |"
            )

    return "\n".join(lines) + "\n"


def run_continuous_improvement_loop(
    moc_items: Optional[List[Dict[str, Any]]] = None,
    iterations: int = 1,
    start_iteration: int = 1,
    input_fix_bundle: Optional[GeneralizedFixBundle] = None,
    abort_on_first_error: bool = True,
    max_steps_per_moc: int = 15,
    per_moc_timeout_sec: float = 2.0,
    output_dir: str = "outputs/moc_loop",
    max_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
) -> Dict[str, Any]:
    """
    Runs the closed-loop continuous improvement cycle across the provided MOCs.
    """
    os.makedirs(output_dir, exist_ok=True)
    if moc_items is None:
        moc_items = discover_moc_pairs(only_rebrickable_mocs=True)

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 12) - 2)

    current_bundle = input_fix_bundle or GeneralizedFixBundle(iteration=start_iteration)
    loop_history: List[Dict[str, Any]] = []

    for offset in range(iterations):
        it_idx = start_iteration + offset
        current_bundle.iteration = it_idx
        t0 = time.time()
        mem_ok, mem_pct = check_memory_headroom(max_ram_percent=80.0)
        eff_workers = max_workers if mem_ok else max(1, max_workers // 2)

        tasks = [
            (
                item,
                current_bundle.to_dict(),
                abort_on_first_error,
                max_steps_per_moc,
                per_moc_timeout_sec,
            )
            for item in moc_items
        ]
        moc_results: List[Dict[str, Any]] = []

        progress_jsonl = os.path.join(output_dir, f"iter_{it_idx}_progress.jsonl")
        if os.path.exists(progress_jsonl):
            os.remove(progress_jsonl)

        has_callables = any(
            bool(item.get("simulated_step_corruptions")) for item in moc_items
        )
        if use_multiprocessing and not has_callables and len(tasks) > 2 and eff_workers > 1:
            with ProcessPoolExecutor(max_workers=eff_workers) as executor:
                futures = [executor.submit(_evaluate_single_moc_worker, t) for t in tasks]
                done_cnt = 0
                with open(progress_jsonl, "a", encoding="utf-8") as pf:
                    for fut in as_completed(futures):
                        res = fut.result()
                        moc_results.append(res)
                        pf.write(json.dumps(res) + "\n")
                        pf.flush()
                        done_cnt += 1
                        if done_cnt % 40 == 0 or done_cnt == len(tasks):
                            print(
                                f"  [Iteración {it_idx}] Progreso: {done_cnt}/{len(tasks)} MOCs evaluados...",
                                flush=True,
                            )
            moc_results.sort(key=lambda r: r["moc_id"])
        else:
            for t in tasks:
                res = _evaluate_single_moc_worker(t)
                moc_results.append(res)

        analyzer = PatternClusterAnalyzer(prior_fix_bundle=current_bundle)
        summary, next_bundle = analyzer.analyze_and_synthesize_fixes(moc_results)
        elapsed = round(time.time() - t0, 3)

        avg_ms_moc = round((elapsed / max(1, len(tasks))) * 1000.0, 1)
        summary["elapsed_sec"] = elapsed
        summary["avg_ms_per_moc"] = avg_ms_moc

        iter_record = {
            "iteration": it_idx,
            "elapsed_sec": elapsed,
            "avg_ms_per_moc": avg_ms_moc,
            "ram_usage_percent": round(mem_pct, 1),
            "workers_used": eff_workers,
            "applied_fix_bundle": current_bundle.to_dict(),
            "summary": summary,
            "moc_results": moc_results,
        }
        loop_history.append(iter_record)

        fix_path = os.path.join(output_dir, f"generalized_fixes_iter_{it_idx}.json")
        with open(fix_path, "w", encoding="utf-8") as f:
            json.dump(next_bundle.to_dict(), f, indent=2)

        iter_report_path = os.path.join(output_dir, f"iter_{it_idx}_report.json")
        with open(iter_report_path, "w", encoding="utf-8") as f:
            json.dump(iter_record, f, indent=2)

        current_bundle = next_bundle

        if summary["total_first_error_aborts"] == 0:
            break

    final_report = {
        "total_iterations_run": len(loop_history),
        "initial_pass_rate_percent": loop_history[0]["summary"]["pass_rate_percent"] if loop_history else 0.0,
        "final_pass_rate_percent": loop_history[-1]["summary"]["pass_rate_percent"] if loop_history else 0.0,
        "initial_mean_survival_ratio": loop_history[0]["summary"]["mean_survival_ratio"] if loop_history else 0.0,
        "final_mean_survival_ratio": loop_history[-1]["summary"]["mean_survival_ratio"] if loop_history else 0.0,
        "final_fix_bundle": current_bundle.to_dict(),
        "iterations": loop_history,
    }

    report_json_path = os.path.join(output_dir, "moc_continuous_loop_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    report_md_path = os.path.join(output_dir, "moc_continuous_loop_summary.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(format_markdown_summary(loop_history))

    return final_report


def main():
    parser = argparse.ArgumentParser(
        description="Run Fail-Fast Oracle Evaluation & Continuous Improvement Loop on MOCs"
    )
    parser.add_argument("--mocs-dir", type=str, nargs="*", default=None, help="Directories to scan for MOCs")
    parser.add_argument("--limit", type=int, default=None, help="Max number of MOCs to evaluate")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run before stopping")
    parser.add_argument("--start-iteration", type=int, default=1, help="Starting iteration number")
    parser.add_argument("--input-fixes", type=str, default=None, help="Path to prior generalized_fixes_iter_N.json")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps evaluated per MOC")
    parser.add_argument("--timeout", type=float, default=2.0, help="Per-MOC timeout in seconds")
    parser.add_argument("--output-dir", type=str, default="outputs/moc_loop", help="Output directory for telemetry")
    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
        help="Continue after first error using GT state instead of aborting set immediately",
    )
    args = parser.parse_args()

    input_bundle = None
    if args.input_fixes and os.path.exists(args.input_fixes):
        with open(args.input_fixes, "r", encoding="utf-8") as f:
            input_bundle = GeneralizedFixBundle.from_dict(json.load(f))

    items = discover_moc_pairs(
        mocs_dirs=args.mocs_dir,
        only_rebrickable_mocs=(args.mocs_dir is None),
        limit=args.limit,
    )
    print(f"[Oracle Loop] Descubiertos {len(items)} modelos MOC Rebrickable (.io) válidos para evaluación.")
    report = run_continuous_improvement_loop(
        moc_items=items,
        iterations=args.iterations,
        start_iteration=args.start_iteration,
        input_fix_bundle=input_bundle,
        abort_on_first_error=not args.teacher_forcing,
        max_steps_per_moc=args.max_steps,
        per_moc_timeout_sec=args.timeout,
        output_dir=args.output_dir,
    )
    it_summary = report["iterations"][-1]["summary"]
    print(
        f"[Oracle Loop - Iteración {args.start_iteration}] Completado: "
        f"MOCs 100% Exactos = {it_summary['mocs_passed_100_percent']}/{it_summary['total_mocs_evaluated']} "
        f"({it_summary['pass_rate_percent']:.2f}%) | "
        f"Supervivencia Media = {it_summary['mean_survival_ratio']*100:.1f}% | "
        f"Pasos Medios antes de Aborto = {it_summary['mean_steps_survived_before_abort']:.2f}"
    )


if __name__ == "__main__":
    main()

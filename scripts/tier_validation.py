#!/usr/bin/env python3
"""
LegoGPT — Tier Validation Engine (Validación Masiva Progresiva por Tier)
=======================================================================

Estrategia de validación masiva en 3 fases:

  FASE 1 (probe)   → 1 set por cada tier/categoría. Detecta fallos, mide tiempos.
  FASE 2 (fix)     → Sintetiza bundle de fixes a partir de los incidentes del probe.
  FASE 3 (scale)   → Escala progresivamente: más sets por tier.

Tiers (complejidad por nº de piezas) — misma definición que
scripts/stratify_corpus_dataset.py:

    MICRO 2-50 | SMALL 51-150 | MEDIUM 151-400
    COMPLEX 401-1000 | EXPERT 1001-3000 | MONUMENTAL >3000

Categorías: MOC (propios) y OFFICIAL_OMR (manuales oficiales).

Uso:
    python3 scripts/tier_validation.py --phase probe
    python3 scripts/tier_validation.py --phase fix
    python3 scripts/tier_validation.py --phase scale --per-tier 5
    python3 scripts/tier_validation.py --phase all
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.apple_silicon import P_CORES
from src.oracle_step_debugger import (
    GeneralizedFixBundle,
    OracleStepDebugger,
    PatternClusterAnalyzer,
)

# --------------------------------------------------------------------------- #
# Configuración de tiers y categorías
# --------------------------------------------------------------------------- #

COMPLEXITY_TIERS: List[Tuple[str, int, int]] = [
    ("MICRO", 0, 50),
    ("SMALL", 51, 150),
    ("MEDIUM", 151, 400),
    ("COMPLEX", 401, 1000),
    ("EXPERT", 1001, 3000),
    ("MONUMENTAL", 3001, 999999),
]

TIER_ORDER = [t[0] for t in COMPLEXITY_TIERS]
CATEGORIES = ["MOC", "OFFICIAL_OMR"]

CORPUS_PATH = "data/dataset_splits/corpus_all_pairs.json"
DEFAULT_OUTDIR = "outputs/tier_validation"

# Tope duro por set. Sin esto, un set patológico cuelga el pool entero
# (bug histórico que dejó jobs colgados durante 11 h consumiendo un core).
TIMEOUT_PER_MOC = 120


def assign_tier(parts: int) -> str:
    """Mapea nº de piezas → tier de complejidad (misma lógica que el stratifier)."""
    for code, low, high in COMPLEXITY_TIERS:
        if low <= parts <= high:
            return code
    return "UNKNOWN"


# --------------------------------------------------------------------------- #
# Worker (module-level para pickling con ProcessPoolExecutor)
# --------------------------------------------------------------------------- #


class _EvalTimeout(Exception):
    """Se lanza cuando un set excede el tiempo máximo permitido."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _EvalTimeout(f"timeout tras {TIMEOUT_PER_MOC}s")


def _eval_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evalúa un set contra su ground truth. Aísla fallos por set.

    El timeout se aplica con signal.alarm (el worker corre en el hilo principal
    de su proceso, así que SIGALRM es seguro aquí). Sin esto, un set problemático
    cuelga el pool entero indefinidamente.
    """
    t0 = time.time()
    timeout = task.get("timeout_sec") or TIMEOUT_PER_MOC
    try:
        from src.geometry_cache import GeometryCache

        bundle = GeneralizedFixBundle.from_dict(task["bundle_dict"])
        geom = GeometryCache.get()
        debugger = OracleStepDebugger(geom_resolver=geom, fix_bundle=bundle)

        # Instala el temporizador solo si hay un timeout configurado
        alarm_installed = False
        if timeout and timeout > 0:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(int(timeout))
            alarm_installed = True

        try:
            result = debugger.evaluate_moc_fail_fast(
                moc_id=task["moc_id"],
                io_path=task["io_path"],
                pdf_path=task.get("pdf_path"),
                abort_on_first_error=False,
                max_steps_to_eval=task["max_steps"],
            )
        finally:
            if alarm_installed:
                signal.alarm(0)

        elapsed = time.time() - t0
        result["elapsed_sec"] = round(elapsed, 3)
        result["worker_status"] = "OK"
        result["complexity_tier"] = task.get("complexity_tier")
        result["category"] = task.get("category")
        result["parts_count"] = task.get("parts_count")
        result["steps_count"] = task.get("steps_count")
        return result

    except _EvalTimeout as e:
        return {
            "moc_id": task["moc_id"],
            "io_path": task["io_path"],
            "success": False,
            "completed_all_steps": False,
            "total_steps": task.get("steps_count", 0),
            "steps_survived": 0,
            "survival_ratio": 0.0,
            "incidents": [],
            "error": f"TIMEOUT: {e}",
            "elapsed_sec": round(time.time() - t0, 3),
            "worker_status": "TIMEOUT",
            "complexity_tier": task.get("complexity_tier"),
            "category": task.get("category"),
            "parts_count": task.get("parts_count"),
            "steps_count": task.get("steps_count"),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "moc_id": task["moc_id"],
            "io_path": task["io_path"],
            "success": False,
            "completed_all_steps": False,
            "total_steps": task.get("steps_count", 0),
            "steps_survived": 0,
            "survival_ratio": 0.0,
            "incidents": [],
            "error": f"{type(e).__name__}: {e}"[:300],
            "elapsed_sec": round(time.time() - t0, 3),
            "worker_status": "ERROR",
            "complexity_tier": task.get("complexity_tier"),
            "category": task.get("category"),
            "parts_count": task.get("parts_count"),
            "steps_count": task.get("steps_count"),
        }


# --------------------------------------------------------------------------- #
# Selección de sets
# --------------------------------------------------------------------------- #


def load_corpus(path: str = CORPUS_PATH) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def group_by_tier_category(
    corpus: List[Dict[str, Any]], only_existing: bool = True
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Agrupa los pares (category, tier) -> lista de sets ordenados por nº piezas."""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in corpus:
        if only_existing and not os.path.exists(item.get("io_path", "")):
            continue
        tier = item.get("complexity_tier") or assign_tier(int(item.get("parts_count", 0)))
        buckets[(item.get("category", "UNKNOWN"), tier)].append(item)

    for key in buckets:
        buckets[key].sort(key=lambda x: int(x.get("parts_count", 0)))
    return dict(buckets)


def select_sets(
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]],
    per_tier: int,
    categories: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Selecciona `per_tier` sets por cada combinación (categoría, tier).

    Si el bucket tiene más elementos que `per_tier`, se toma una muestra
    ESPACIADA sobre el orden por nº de piezas, para cubrir todo el rango de
    complejidad del tier en lugar de solo los más pequeños.
    """
    cats = categories or CATEGORIES
    tier_list = tiers or TIER_ORDER
    selected: List[Dict[str, Any]] = []

    for cat in cats:
        for tier in tier_list:
            pool = buckets.get((cat, tier), [])
            if not pool:
                continue
            if len(pool) <= per_tier:
                chosen = pool
            else:
                step = len(pool) / float(per_tier)
                idxs = sorted({int(i * step) for i in range(per_tier)})
                chosen = [pool[i] for i in idxs]
            selected.extend(chosen)

    return selected


def to_eval_item(item: Dict[str, Any], max_steps: Optional[int]) -> Dict[str, Any]:
    """Normaliza un par del corpus al formato de item de evaluación."""
    steps = int(item.get("steps_count", 0) or 0)
    return {
        "moc_id": item.get("set_id", "unknown"),
        "io_path": item["io_path"],
        "pdf_path": item.get("pdf_path"),
        "num_parts": int(item.get("parts_count", 0) or 0),
        "num_steps": steps if steps > 0 else 30,
        "parts_per_step": 0,
        "complexity_tier": item.get("complexity_tier", "UNKNOWN"),
        "category": item.get("category", "UNKNOWN"),
    }


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #


def evaluate_sets(
    items: List[Dict[str, Any]],
    bundle: GeneralizedFixBundle,
    max_steps: Optional[int] = None,
    workers: int = P_CORES,
    timeout_sec: int = TIMEOUT_PER_MOC,
) -> List[Dict[str, Any]]:
    """Evalúa una lista de sets en paralelo. Devuelve un resultado por set."""
    if not items:
        return []

    bundle_dict = bundle.to_dict()
    tasks = []
    for it in items:
        task = dict(it)
        task["bundle_dict"] = bundle_dict
        task["max_steps"] = max_steps or it.get("num_steps", 30)
        task["parts_count"] = it.get("num_parts", 0)
        task["steps_count"] = it.get("num_steps", 0)
        task["timeout_sec"] = timeout_sec
        tasks.append(task)

    results: List[Dict[str, Any]] = []
    n_workers = max(1, min(workers, len(tasks)))
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_eval_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {
                    "moc_id": t["moc_id"],
                    "io_path": t["io_path"],
                    "success": False,
                    "completed_all_steps": False,
                    "total_steps": t.get("num_steps", 0),
                    "steps_survived": 0,
                    "survival_ratio": 0.0,
                    "incidents": [],
                    "error": f"WorkerFailure: {e}"[:300],
                    "elapsed_sec": 0.0,
                    "worker_status": "ERROR",
                    "complexity_tier": t.get("complexity_tier"),
                    "category": t.get("category"),
                }
            results.append(res)
            done += 1
            status = "OK  " if res.get("completed_all_steps") else "FAIL"
            print(
                f"  [{done}/{len(tasks)}] {status} {str(res.get('moc_id','?'))[:16]:<16} "
                f"{str(res.get('complexity_tier','?'))[:11]:<11} "
                f"{str(res.get('category','?'))[:13]:<13} "
                f"{res.get('steps_survived',0):>4}/{res.get('total_steps',0):<4} steps "
                f"({res.get('elapsed_sec',0):.2f}s)",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"  → {len(results)} sets evaluados en {elapsed:.1f}s "
        f"({elapsed/max(1,len(results)):.2f}s/set, {n_workers} workers)",
        flush=True,
    )
    return results


# --------------------------------------------------------------------------- #
# Análisis
# --------------------------------------------------------------------------- #


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrega resultados globales y por (categoría, tier)."""
    if not results:
        return {"total": 0, "passed": 0, "pass_rate_percent": 0.0, "by_combo": {}}

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(rows)
        passed = sum(1 for r in rows if r.get("completed_all_steps"))
        times = [r.get("elapsed_sec", 0.0) for r in rows]
        survival = [r.get("survival_ratio", 0.0) for r in rows]
        return {
            "n": n,
            "passed": passed,
            "pass_rate_percent": round(passed / n * 100, 2),
            "mean_survival_ratio": round(sum(survival) / n, 4),
            "mean_elapsed_sec": round(sum(times) / n, 3),
            "max_elapsed_sec": round(max(times), 3) if times else 0.0,
            "total_incidents": sum(len(r.get("incidents", [])) for r in rows),
            "worker_errors": sum(1 for r in rows if r.get("worker_status") == "ERROR"),
        }

    by_combo: Dict[str, Dict[str, Any]] = {}
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(r.get("category", "?"), r.get("complexity_tier", "?"))].append(r)
    for (cat, tier), rows in sorted(groups.items()):
        by_combo[f"{cat}/{tier}"] = _agg(rows)

    passed_all = sum(1 for r in results if r.get("completed_all_steps"))
    return {
        "total": len(results),
        "passed": passed_all,
        "pass_rate_percent": round(passed_all / len(results) * 100, 2),
        "mean_elapsed_sec": round(
            sum(r.get("elapsed_sec", 0.0) for r in results) / len(results), 3
        ),
        "total_elapsed_sec": round(sum(r.get("elapsed_sec", 0.0) for r in results), 2),
        "total_incidents": sum(len(r.get("incidents", [])) for r in results),
        "by_combo": by_combo,
    }


def analyze_timings(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analiza tiempos por tier para dimensionar la validación masiva."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(r.get("category", "?"), r.get("complexity_tier", "?"))].append(r)

    per_tier: Dict[str, Any] = {}
    for (cat, tier), rows in sorted(groups.items()):
        times = sorted(r.get("elapsed_sec", 0.0) for r in rows)
        steps = [r.get("total_steps", 0) for r in rows]
        parts = [r.get("parts_count", 0) or 0 for r in rows]
        mean_t = sum(times) / len(times)
        mean_steps = sum(steps) / len(steps)
        per_tier[f"{cat}/{tier}"] = {
            "n_sets": len(rows),
            "mean_sec_per_set": round(mean_t, 3),
            "min_sec": round(times[0], 3),
            "max_sec": round(times[-1], 3),
            "mean_steps": round(mean_steps, 1),
            "mean_parts": round(sum(parts) / len(parts), 1) if parts else 0,
            "sec_per_step": round(mean_t / max(1.0, mean_steps), 4),
        }

    corpus = load_corpus()
    buckets = group_by_tier_category(corpus)
    projection: Dict[str, Any] = {}
    for key, stats in per_tier.items():
        cat, tier = key.split("/")
        population = len(buckets.get((cat, tier), []))
        projected = population * stats["mean_sec_per_set"]
        projection[key] = {
            "population_sets": population,
            "sampled": stats["n_sets"],
            "projected_total_sec": round(projected, 1),
            "projected_total_min": round(projected / 60, 1),
            "projected_parallel_min": round(projected / 60 / P_CORES, 1),
        }

    return {"per_tier": per_tier, "projection_full_corpus": projection}


def root_causes(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extrae las causas raíz de los fallos para la fase de fix."""
    causes: Dict[str, int] = defaultdict(int)
    sub_causes: Dict[str, int] = defaultdict(int)
    parts: Dict[str, int] = defaultdict(int)
    failing = 0
    for r in results:
        if not r.get("completed_all_steps"):
            failing += 1
        for inc in r.get("incidents", []):
            causes[inc.get("root_cause_category", "UNKNOWN")] += 1
            sub_causes[
                f"{inc.get('root_cause_category','?')}/{inc.get('sub_cause','?')}"
            ] += 1
            parts[inc.get("child_part_id", "?")] += 1
    return {
        "failing_sets": failing,
        "causes": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
        "sub_causes": dict(sorted(sub_causes.items(), key=lambda kv: -kv[1])),
        "top_failing_parts": dict(sorted(parts.items(), key=lambda kv: -kv[1])[:15]),
    }


# --------------------------------------------------------------------------- #
# Utilidades de persistencia
# --------------------------------------------------------------------------- #


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_bundle(path: Optional[str]) -> GeneralizedFixBundle:
    if path and os.path.exists(path):
        with open(path) as f:
            return GeneralizedFixBundle.from_dict(json.load(f))
    return GeneralizedFixBundle()


def print_tier_table(summary: Dict[str, Any]) -> None:
    """Imprime la tabla de resultados por tier/categoría."""
    print("\n--- RESULTADOS POR TIER/CATEGORÍA ---")
    for key, agg in summary.get("by_combo", {}).items():
        flag = "✅" if agg["pass_rate_percent"] == 100 else "⚠️ "
        print(
            f"  {flag} {key:<26} {agg['passed']}/{agg['n']} ok "
            f"({agg['pass_rate_percent']:>5.1f}%) | "
            f"survival {agg['mean_survival_ratio']*100:>5.1f}% | "
            f"{agg['mean_elapsed_sec']:>6.2f}s/set | "
            f"{agg['total_incidents']} inc"
        )
    print(
        f"\n  Global: {summary['passed']}/{summary['total']} "
        f"({summary['pass_rate_percent']:.1f}%) | "
        f"{summary['total_elapsed_sec']:.1f}s total | "
        f"{summary['total_incidents']} incidentes"
    )


# --------------------------------------------------------------------------- #
# FASE 1 — PROBE
# --------------------------------------------------------------------------- #


def phase_probe(args) -> Dict[str, Any]:
    """FASE 1: valida 1 set por cada tier/categoría."""
    print("\n" + "=" * 78)
    print("FASE 1 — PROBE: 1 set por tier/categoría")
    print("=" * 78)

    corpus = load_corpus(args.corpus)
    buckets = group_by_tier_category(corpus)
    items = [
        to_eval_item(it, args.max_steps)
        for it in select_sets(
            buckets, per_tier=1, categories=args.categories, tiers=args.tiers
        )
    ]

    print(f"Corpus: {len(corpus)} pares | Seleccionados para probe: {len(items)}")
    print(f"Workers: {P_CORES} | Max steps: {args.max_steps or 'auto'}\n")

    bundle = load_bundle(args.bundle)
    results = evaluate_sets(items, bundle, args.max_steps, args.workers, args.timeout)

    summary = summarize(results)
    timings = analyze_timings(results)
    causes = root_causes(results)

    report = {
        "phase": "probe",
        "timestamp": datetime.now().isoformat(),
        "items_evaluated": len(items),
        "summary": summary,
        "timings": timings,
        "root_causes": causes,
        "results": results,
    }
    out = os.path.join(args.output_dir, "probe_report.json")
    save_json(out, report)

    print_tier_table(summary)

    if causes["causes"]:
        print("\n--- CAUSAS RAÍZ ---")
        for cause, cnt in list(causes["causes"].items())[:8]:
            print(f"  {cnt:>4}x  {cause}")

    print(f"\n✓ Reporte: {out}")
    return report


# --------------------------------------------------------------------------- #
# FASE 2 — FIX
# --------------------------------------------------------------------------- #


def phase_fix(args) -> Dict[str, Any]:
    """FASE 2: sintetiza un bundle de fixes a partir de los incidentes del probe."""
    print("\n" + "=" * 78)
    print("FASE 2 — FIX: síntesis de correcciones generalizadas")
    print("=" * 78)

    probe_path = os.path.join(args.output_dir, "probe_report.json")
    if not os.path.exists(probe_path):
        print(f"✗ No se encontró {probe_path}. Ejecuta primero --phase probe.")
        return {}

    with open(probe_path) as f:
        probe = json.load(f)

    results = probe.get("results", [])
    prior = load_bundle(args.bundle)

    analyzer = PatternClusterAnalyzer(prior_fix_bundle=prior)
    summary, next_bundle = analyzer.analyze_and_synthesize_fixes(results)

    bundle_path = os.path.join(args.output_dir, "fix_bundle.json")
    save_json(bundle_path, next_bundle.to_dict())

    print("\n--- BUNDLE SINTETIZADO ---")
    b = next_bundle
    print(f"  iteration                   : {b.iteration}")
    print(f"  disable_substrate_filter    : {b.disable_substrate_filter}")
    print(f"  auto_retry_flip_on_dead_end : {b.auto_retry_flip_on_dead_end}")
    print(f"  penetration_threshold_ldu   : {b.penetration_threshold_ldu}")
    print(f"  beam_width                  : {b.beam_width}")
    print(f"  roi_max_keep                : {b.roi_max_keep}")
    print(f"  snot_and_hinge_parts        : {len(b.snot_and_hinge_parts)}")
    print(f"  fractional_offset_parts     : {len(b.fractional_offset_parts)}")
    print(f"  socket_coverage_parts       : {len(b.socket_coverage_parts)}")
    print(f"  multi_box_hollow_parts      : {len(b.multi_box_hollow_parts)}")
    print(f"  missing_port_parts          : {len(b.missing_port_parts)}")

    changes = []
    for field in (
        "disable_substrate_filter", "auto_retry_flip_on_dead_end",
        "penetration_threshold_ldu", "w_edge_score", "w_stud_score",
        "symmetry_pair_bonus", "beam_width", "roi_max_keep",
    ):
        old = getattr(prior, field, None)
        new = getattr(b, field, None)
        if old != new:
            changes.append(f"    {field}: {old} → {new}")
    for field in (
        "snot_and_hinge_parts", "fractional_offset_parts", "socket_coverage_parts",
        "multi_box_hollow_parts", "missing_port_parts",
    ):
        old_set = set(getattr(prior, field, []) or [])
        new_set = set(getattr(b, field, []) or [])
        if new_set - old_set:
            changes.append(
                f"    {field} +{len(new_set - old_set)}: {sorted(new_set - old_set)[:5]}"
            )

    if changes:
        print("\n--- CAMBIOS RESPECTO AL BUNDLE PREVIO ---")
        for c in changes:
            print(c)
    else:
        print("\n  (sin cambios respecto al bundle previo)")

    fix_report = {
        "phase": "fix",
        "timestamp": datetime.now().isoformat(),
        "source_probe": probe_path,
        "cluster_summary": summary,
        "bundle": next_bundle.to_dict(),
        "changes_vs_prior": changes,
    }
    save_json(os.path.join(args.output_dir, "fix_report.json"), fix_report)
    print(f"\n✓ Bundle: {bundle_path}")
    return fix_report


# --------------------------------------------------------------------------- #
# FASE 3 — SCALE
# --------------------------------------------------------------------------- #


def phase_scale(args) -> Dict[str, Any]:
    """FASE 3: escala progresivamente a más sets por tier."""
    print("\n" + "=" * 78)
    print(f"FASE 3 — SCALE: {args.per_tier} sets por tier/categoría")
    print("=" * 78)

    corpus = load_corpus(args.corpus)
    buckets = group_by_tier_category(corpus)

    # Excluir los sets ya validados en el probe (para no repetir)
    exclude: set = set()
    probe_path = os.path.join(args.output_dir, "probe_report.json")
    if os.path.exists(probe_path) and not args.reinclude_probe:
        with open(probe_path) as f:
            for r in json.load(f).get("results", []):
                if r.get("io_path"):
                    exclude.add(r["io_path"])

    # per_tier + 1 para compensar la exclusión del probe
    selected = select_sets(
        buckets, per_tier=args.per_tier + 1,
        categories=args.categories, tiers=args.tiers
    )
    selected = [s for s in selected if s["io_path"] not in exclude]
    items = [to_eval_item(it, args.max_steps) for it in selected]

    print(
        f"Seleccionados para scale: {len(items)} sets "
        f"(excluidos {len(exclude)} ya validados en probe)"
    )

    default_bundle = os.path.join(args.output_dir, "fix_bundle.json")
    bundle_path = args.bundle or default_bundle
    bundle = load_bundle(bundle_path)
    has_fixes = bool(
        bundle.disable_substrate_filter
        or bundle.auto_retry_flip_on_dead_end
        or bundle.snot_and_hinge_parts
        or bundle.fractional_offset_parts
        or bundle.socket_coverage_parts
        or bundle.multi_box_hollow_parts
        or bundle.missing_port_parts
    )
    print(f"Bundle en uso: {bundle_path} ({'con fixes' if has_fixes else 'default'})")

    results = evaluate_sets(items, bundle, args.max_steps, args.workers, args.timeout)

    summary = summarize(results)
    timings = analyze_timings(results)
    causes = root_causes(results)

    report = {
        "phase": "scale",
        "timestamp": datetime.now().isoformat(),
        "per_tier_requested": args.per_tier,
        "items_evaluated": len(items),
        "bundle_used": bundle_path,
        "summary": summary,
        "timings": timings,
        "root_causes": causes,
        "results": results,
    }
    out = os.path.join(args.output_dir, f"scale_report_{args.per_tier}.json")
    save_json(out, report)

    print_tier_table(summary)

    proj = timings.get("projection_full_corpus", {})
    if proj:
        print("\n--- PROYECCIÓN CORPUS COMPLETO (con estos tiempos) ---")
        total_min = 0.0
        for key, p in sorted(proj.items()):
            print(
                f"  {key:<26} {p['population_sets']:>4} sets → "
                f"{p['projected_total_min']:>8.1f} min secuencial "
                f"({p['projected_parallel_min']:>6.1f} min en {P_CORES} workers)"
            )
            total_min += p["projected_parallel_min"]
        print(f"  {'TOTAL':<26} {'':>4}      → {total_min:>8.1f} min en paralelo")

    print(f"\n✓ Reporte: {out}")
    return report


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #


def phase_all(args) -> None:
    """Ciclo completo: probe → fix → scale."""
    phase_probe(args)
    phase_fix(args)
    phase_scale(args)

    probe_p = os.path.join(args.output_dir, "probe_report.json")
    scale_p = os.path.join(args.output_dir, f"scale_report_{args.per_tier}.json")
    if os.path.exists(probe_p) and os.path.exists(scale_p):
        with open(probe_p) as f:
            p = json.load(f)
        with open(scale_p) as f:
            s = json.load(f)
        print("\n" + "=" * 78)
        print("COMPARATIVA PROBE vs SCALE")
        print("=" * 78)
        print(
            f"  Probe : {p['summary']['passed']}/{p['summary']['total']} "
            f"({p['summary']['pass_rate_percent']:.1f}%)"
        )
        print(
            f"  Scale : {s['summary']['passed']}/{s['summary']['total']} "
            f"({s['summary']['pass_rate_percent']:.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LegoGPT — Validación masiva progresiva por tier"
    )
    parser.add_argument(
        "--phase", choices=["probe", "fix", "scale", "all"], default="probe",
        help="Fase del ciclo de validación"
    )
    parser.add_argument(
        "--per-tier", type=int, default=5,
        help="Sets por tier en la fase scale"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Máximo de pasos a evaluar por set"
    )
    parser.add_argument(
        "--workers", type=int, default=P_CORES,
        help="Workers del pool de evaluación"
    )
    parser.add_argument(
        "--timeout", type=int, default=TIMEOUT_PER_MOC,
        help="Timeout por set en segundos (0 = sin timeout)"
    )
    parser.add_argument(
        "--categories", nargs="*", default=None,
        choices=["MOC", "OFFICIAL_OMR"],
        help="Categorías a validar (default: ambas)"
    )
    parser.add_argument(
        "--tiers", nargs="*", default=None,
        help="Tiers a validar (default: todos)"
    )
    parser.add_argument(
        "--bundle", type=str, default=None,
        help="Ruta a un bundle de fixes previo"
    )
    parser.add_argument(
        "--corpus", type=str, default=CORPUS_PATH,
        help="Ruta al corpus de pares"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTDIR,
        help="Directorio de salida"
    )
    parser.add_argument(
        "--reinclude-probe", action="store_true",
        help="Reincluir los sets ya validados en el probe"
    )
    args = parser.parse_args()

    print("=" * 78)
    print("LEGOGPT — TIER VALIDATION ENGINE")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 78)

    if args.phase == "probe":
        phase_probe(args)
    elif args.phase == "fix":
        phase_fix(args)
    elif args.phase == "scale":
        phase_scale(args)
    else:
        phase_all(args)


if __name__ == "__main__":
    main()








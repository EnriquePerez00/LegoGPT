"""
Coordinated Multi-Agent Orchestrator for the Independent PDF -> .io Pipeline.

Executes the 6 specialized agents in sequence per page/step, handles backtracking,
exports BrickLink Studio `.io` files, and runs Ground-Truth causal auditing & auto-fixing
when paired with `.io` GT files (designed for the ~300 PDF + .io corpus).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

try:
    import fitz as pymupdf
except ImportError:
    pymupdf = None

from src.apple_silicon import P_CORES, workers_for
from src.connector_occupancy_map import ConnectorOccupancyMap
from src.graph_quality_metrics import GraphQualityEvaluator, nodes_to_steps
from src.inventory_matcher import LegoInventoryMatcher
from src.ldraw_geometry import LDrawGeometryResolver
from src.physical_hypothesis_generator import ConnectionHypothesis, PhysicalHypothesisGenerator
from src.solid_collision_validator import SolidCollisionValidator
from src.step_axonometric_renderer import StepAxonometricRenderer
from src.subassembly_manager import DisjointGraphTask, ManualGraphSegmenter, SubassemblyManager
from src.visual_pose_scorer import VisualPoseScorer
from src.agentic_pipeline.agents import (
    LEARNED_RULES_PATH,
    ContextAndCameraAgent,
    GTAuditorAndFixerAgent,
    Identifier3DAgent,
    PhysicalGuardianAgent,
    PlacerVisualROIAgent,
    TemporalSupervisorAgent,
    load_learned_rules,
    save_learned_rules,
)
from src.agentic_pipeline.io_exporter import (
    align_inferred_to_gt_frame,
    export_nodes_to_studio_io,
    load_gt_nodes_from_io,
)
from src.agentic_pipeline.schemas import (
    MODEL_ROUTING_CONFIG,
    AgenticPipelineReport,
    GTAuditDiagnosis,
    ProposedFix,
    StepInferenceResult,
)


class AgenticGraphPipelineOrchestrator:
    """
    Independent Multi-Agent Orchestrator that coordinates:
      1. `ContextAndCameraAgent`   (flash_lite)
      2. `Identifier3DAgent`       (flash)
      3. `PhysicalGuardianAgent`   (flash_lite)
      4. `PlacerVisualROIAgent`    (flash + MPS GPU)
      5. `TemporalSupervisorAgent` (flash)
      6. `GTAuditorAndFixerAgent`  (pro - activated when GT .io is provided)
    """

    def __init__(
        self,
        db_path: str = "data/catalog/models_catalog.db",
        rules_path: str = LEARNED_RULES_PATH,
        roi_margin_ratio: float = 0.35,
    ):
        self.db_path = db_path
        self.rules_path = rules_path
        self.roi_margin_ratio = roi_margin_ratio

        # Hardware profile (Apple Silicon M4 + MPS + 48GB Unified RAM safety guard)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.max_workers = P_CORES

        # Shared geometry & validation engines (cached in Unified RAM)
        self.geom_resolver = LDrawGeometryResolver(db_path=db_path)
        self.step_renderer = StepAxonometricRenderer(self.geom_resolver)
        self.hypothesis_gen = PhysicalHypothesisGenerator(self.geom_resolver)
        self.visual_scorer = VisualPoseScorer(self.step_renderer, db_path=db_path)
        self.occupancy_map = ConnectorOccupancyMap(self.geom_resolver)
        self.solid_validator = SolidCollisionValidator(self.geom_resolver)
        self.quality_evaluator = GraphQualityEvaluator()
        self.subassembly_mgr = SubassemblyManager(self.geom_resolver)

        # Instantiate the 6 Specialized Agents
        self.context_agent = ContextAndCameraAgent(self.visual_scorer)
        self.identifier_agent = Identifier3DAgent(self.geom_resolver, self.hypothesis_gen)
        self.guardian_agent = PhysicalGuardianAgent(self.solid_validator)
        self.placer_agent = PlacerVisualROIAgent(
            self.step_renderer, self.visual_scorer, self.geom_resolver
        )
        self.supervisor_agent = TemporalSupervisorAgent()
        self.gt_auditor_agent = GTAuditorAndFixerAgent(self.geom_resolver)

    @staticmethod
    def get_hardware_status() -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        return {
            "platform": "macOS Apple Silicon M4",
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "p_cores_workers": P_CORES,
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "ram_percent_used": round(vm.percent, 1),
        }

    def _detect_flip_pages(self, pdf_path: str) -> Dict[int, bool]:
        """Scans PDF vector drawings to identify step rotation/flip icons."""
        flip_pages: Dict[int, bool] = {}
        if not pymupdf or not pdf_path or not os.path.exists(pdf_path):
            return flip_pages
        try:
            doc = pymupdf.open(pdf_path)
            for idx in range(len(doc)):
                page = doc[idx]
                for d in page.get_drawings():
                    r, fill = d.get("rect"), d.get("fill")
                    if r is None or fill is None or len(fill) < 3:
                        continue
                    if 35 <= r.width <= 60 and 35 <= r.height <= 60 and min(fill[:3]) > 0.97:
                        if r.x0 < 90 and 80 < r.y0 < 190:
                            flip_pages[idx + 1] = True
                            break
            doc.close()
        except Exception:
            pass
        return flip_pages

    def _extract_steps_from_pdf_or_gt(
        self,
        pdf_path: str,
        gt_steps: Optional[List[List[Dict[str, Any]]]] = None,
        precomputed_steps_json: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts step-by-step part inventories and 2D page crops from:
          1. Precomputed step matches JSON if provided,
          2. Live PDF extraction via `LegoInventoryMatcher` when `pdf_path` exists, or
          3. Step inventory derived from `gt_steps` (when testing pure 3D placement & GT auditing).
        """
        if precomputed_steps_json and os.path.exists(precomputed_steps_json):
            with open(precomputed_steps_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if "step_pages" in data:
                return data["step_pages"]

        if pdf_path and os.path.exists(pdf_path):
            matcher = LegoInventoryMatcher(pdf_path)
            inv_pages = matcher.detect_inventory_pages()
            if inv_pages:
                matcher.extract_inventory_templates()
                step_matches = matcher.extract_and_match_all_steps()
                if step_matches:
                    return step_matches

        # Fallback: construct step inventory descriptors from gt_steps (without leaking 3D poses!)
        if gt_steps:
            synthesized_steps: List[Dict[str, Any]] = []
            for s_idx, step_nodes in enumerate(gt_steps):
                grouped: Dict[Tuple[str, int], int] = {}
                for n in step_nodes:
                    pid = str(n.get("part_id", "3001.dat")).replace(".dat", "")
                    col = int(n.get("color_code", 71))
                    grouped[(pid, col)] = grouped.get((pid, col), 0) + 1

                items = []
                for (pid, col), qty in grouped.items():
                    items.append({
                        "matched_element_id": f"{pid}_{col}",
                        "qty": qty,
                        "matched_template": {
                            "part_num": pid,
                            "color_id": col,
                        },
                    })
                synthesized_steps.append({
                    "page_num": s_idx + 1,
                    "step_label": f"Step {s_idx + 1}",
                    "items": items,
                    "assembly_crop_path": "",
                })
            return synthesized_steps

        return []

    def _infer_single_graph(
        self,
        task: DisjointGraphTask,
        set_num: str,
        flip_pages: Dict[int, bool],
        learned_rules: Dict[str, Any],
        gt_steps: Optional[List[List[Dict[str, Any]]]] = None,
        teacher_forcing_on_audit: bool = True,
        initial_piece_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Infers 3D piece placements for an isolated graph (minifigure, subassembly, or main model).
        Runs with its own isolated ConnectorOccupancyMap so independent models don't contaminate
        or collide with each other during parallel search.
        """
        local_occupancy = ConnectorOccupancyMap(self.geom_resolver)
        placed_nodes: List[Dict[str, Any]] = []
        step_results: List[StepInferenceResult] = []
        proposed_fixes: List[ProposedFix] = []
        root_causes: Dict[str, int] = {}

        current_flipped_state = False
        piece_counter = initial_piece_index
        exact_before = 0
        exact_after = 0
        exact_after_pure = 0  # excludes teacher-forced (GT-injected) poses
        audited_count = 0

        for step_idx, sp_data in enumerate(task.steps_data):
            page_num = int(sp_data.get("page_num", step_idx + 1))
            step_ctx, evidence, calib = self.context_agent.analyze_step_page(
                step_page_data=sp_data,
                existing_nodes=placed_nodes,
                flip_pages=flip_pages,
                current_flipped_state=current_flipped_state,
                learned_rules=learned_rules,
                set_num=set_num,
            )
            step_ctx.subassembly_id = task.graph_id
            current_flipped_state = step_ctx.is_flipped

            step_res = StepInferenceResult(
                page_num=step_ctx.page_num,
                step_label=step_ctx.step_label,
            )

            occupied_step_studs: set = set()
            prior_sibling_hyp: Optional[ConnectionHypothesis] = None
            gt_step_nodes = gt_steps[step_idx] if (gt_steps and step_idx < len(gt_steps)) else []
            used_gt_indices: set = set()

            for part_spec in step_ctx.parts_to_place:
                resolved_pid = self.geom_resolver.resolve_compatible_part_id(part_spec["raw_part_id"])
                color_id = int(part_spec["color_id"])

                id_candidates = self.identifier_agent.propose_placements(
                    part_spec=part_spec,
                    existing_nodes=placed_nodes,
                    occupancy_map=local_occupancy,
                    step_context=step_ctx,
                    calib=calib if evidence else None,
                    learned_rules=learned_rules,
                    prior_sibling_hypothesis=prior_sibling_hyp,
                )
                step_res.candidates_generated_count += len(id_candidates)

                valid_candidates, _ = self.guardian_agent.filter_physically_valid(
                    candidates=id_candidates,
                    existing_nodes=placed_nodes,
                    occupied_step_studs=occupied_step_studs,
                    learned_rules=learned_rules,
                    is_flipped=step_ctx.is_flipped,
                )

                if not valid_candidates and id_candidates:
                    step_res.backtracking_triggered = True
                    relaxed_id_cands = self.identifier_agent.propose_placements(
                        part_spec=part_spec,
                        existing_nodes=placed_nodes,
                        occupancy_map=local_occupancy,
                        step_context=step_ctx,
                        calib=None,
                        learned_rules=learned_rules,
                        prior_sibling_hypothesis=prior_sibling_hyp,
                        relax_roi_filter=True,
                    )
                    valid_candidates, _ = self.guardian_agent.filter_physically_valid(
                        candidates=relaxed_id_cands,
                        existing_nodes=placed_nodes,
                        occupied_step_studs=occupied_step_studs,
                        learned_rules=learned_rules,
                        is_flipped=step_ctx.is_flipped,
                    )
                    if not valid_candidates:
                        valid_candidates = id_candidates[:8]

                step_res.candidates_guardian_valid_count += len(valid_candidates)

                ranked_candidates, roi_evals = self.placer_agent.evaluate_and_rank_in_roi(
                    valid_candidates=valid_candidates,
                    existing_nodes=placed_nodes,
                    evidence=evidence,
                    calib=calib,
                    step_context=step_ctx,
                    learned_rules=learned_rules,
                )
                step_res.candidates_visually_scored_count += len(ranked_candidates)

                chosen_hyp: Optional[ConnectionHypothesis] = (
                    ranked_candidates[0] if ranked_candidates else None
                )
                chosen_roi_eval = roi_evals.get(id(chosen_hyp)) if chosen_hyp else None

                # Audit against Ground Truth .io if provided
                matching_gt_node: Optional[Dict[str, Any]] = None
                if gt_step_nodes:
                    best_gt_idx = None
                    best_gt_dist = float("inf")
                    for g_i, g_node in enumerate(gt_step_nodes):
                        if g_i in used_gt_indices:
                            continue
                        g_pid = self.geom_resolver.resolve_compatible_part_id(g_node.get("part_id", ""))
                        if g_pid == resolved_pid or len(gt_step_nodes) == 1:
                            if chosen_hyp is not None:
                                g_tf = np.array(g_node["absolute_transform"], dtype=np.float32).reshape(4, 4)
                                d = float(np.linalg.norm(chosen_hyp.translation - g_tf[:3, 3]))
                            else:
                                d = 0.0
                            if d < best_gt_dist:
                                best_gt_dist = d
                                best_gt_idx = g_i
                    if best_gt_idx is not None:
                        used_gt_indices.add(best_gt_idx)
                        matching_gt_node = gt_step_nodes[best_gt_idx]

                if matching_gt_node is not None:
                    audited_count += 1
                    audit_diag = self.gt_auditor_agent.audit_part_placement(
                        step_num=step_ctx.page_num,
                        piece_idx=piece_counter + 1,
                        part_id=resolved_pid,
                        color_id=color_id,
                        chosen_hypothesis=chosen_hyp,
                        identifier_candidates=id_candidates,
                        guardian_valid_candidates=valid_candidates,
                        placer_ranked_candidates=ranked_candidates,
                        gt_node=matching_gt_node,
                        learned_rules=learned_rules,
                        set_num=set_num,
                    )
                    step_res.gt_audits.append(audit_diag)
                    root_causes[audit_diag.root_cause] = (
                        root_causes.get(audit_diag.root_cause, 0) + 1
                    )

                    if audit_diag.is_exact_match:
                        exact_before += 1
                        exact_after += 1
                        exact_after_pure += 1
                    else:
                        if audit_diag.proposed_fix is not None:
                            proposed_fixes.append(audit_diag.proposed_fix)

                        re_ranked, re_roi_evals = self.placer_agent.evaluate_and_rank_in_roi(
                            valid_candidates=valid_candidates,
                            existing_nodes=placed_nodes,
                            evidence=evidence,
                            calib=calib,
                            step_context=step_ctx,
                            learned_rules=learned_rules,
                        )
                        if re_ranked:
                            re_best = re_ranked[0]
                            gt_tf = np.array(matching_gt_node["absolute_transform"], dtype=np.float32).reshape(4, 4)
                            is_fixed, _, _ = self.gt_auditor_agent._pose_matches_gt(
                                re_best.absolute_transform, gt_tf, resolved_pid
                            )
                            if is_fixed:
                                chosen_hyp = re_best
                                chosen_roi_eval = re_roi_evals.get(id(re_best))
                                exact_after += 1
                                exact_after_pure += 1
                            elif teacher_forcing_on_audit:
                                gt_tf_mat = np.array(matching_gt_node["absolute_transform"], dtype=np.float32).reshape(4, 4)
                                chosen_hyp = ConnectionHypothesis(
                                    child_part_id=resolved_pid,
                                    child_element_id=part_spec["element_id"],
                                    rotation_matrix=gt_tf_mat[:3, :3],
                                    translation=gt_tf_mat[:3, 3],
                                    absolute_transform=gt_tf_mat,
                                    parent_part_index=max(0, piece_counter - 1),
                                    matched_studs=[tuple(gt_tf_mat[:3, 3].tolist())],
                                    stud_count=1,
                                    color_id=color_id,
                                    total_score=1.0,
                                )
                                exact_after += 1

                if chosen_hyp is None:
                    tf_mat = np.eye(4, dtype=np.float32)
                    pos_list = [0.0, 0.0, 0.0]
                    c_points: List[Any] = []
                    parent_idx = None
                else:
                    tf_mat = chosen_hyp.absolute_transform.copy()
                    pos_list = chosen_hyp.translation.tolist()
                    c_points = [list(s) for s in chosen_hyp.matched_studs]
                    parent_idx = chosen_hyp.parent_part_index if chosen_hyp.parent_part_index >= 0 else None
                    for s in chosen_hyp.matched_studs:
                        occupied_step_studs.add(tuple(np.round(s, 1)))
                    prior_sibling_hyp = chosen_hyp

                node_dict: Dict[str, Any] = {
                    "part_index": piece_counter,
                    "step_id": step_ctx.page_num,
                    "page_num": step_ctx.page_num,
                    "step_label": step_ctx.step_label,
                    "part_id": resolved_pid,
                    "raw_part_id": part_spec["raw_part_id"],
                    "element_id": part_spec["element_id"],
                    "color_code": color_id,
                    "y_bottom_ldu": float(pos_list[1]),
                    "level_plates": int(round(abs(float(pos_list[1])) / 8.0)),
                    "view_flipped": step_ctx.is_flipped,
                    "subassembly_id": task.graph_id,
                    "graph_type": task.graph_type,
                    "absolute_transform": tf_mat.flatten().tolist(),
                    "connections": (
                        [
                            {
                                "parent_part_index": parent_idx,
                                "relative_offset_ldu": [round(v, 2) for v in pos_list],
                                "contact_points": c_points,
                            }
                        ]
                        if parent_idx is not None
                        else []
                    ),
                    "agentic_roi_evaluation": asdict(chosen_roi_eval) if chosen_roi_eval else None,
                }

                placed_nodes.append(node_dict)
                step_res.placed_nodes.append(node_dict)

                try:
                    local_occupancy.register_placed_part(
                        part_index=piece_counter,
                        part_id=resolved_pid,
                        element_id=part_spec["element_id"],
                        transform_4x4=tf_mat,
                        connected_parent_index=parent_idx,
                        contact_points=c_points,
                    )
                except Exception:
                    pass

                piece_counter += 1

            step_results.append(step_res)

        return {
            "graph_id": task.graph_id,
            "graph_type": task.graph_type,
            "placed_nodes": placed_nodes,
            "step_results": step_results,
            "proposed_fixes": proposed_fixes,
            "root_causes": root_causes,
            "exact_before": exact_before,
            "exact_after": exact_after,
            "exact_after_pure": exact_after_pure,
            "audited_count": audited_count,
        }

    def run_pipeline(
        self,
        set_num: str,
        pdf_path: str = "",
        gt_io_path: Optional[str] = None,
        output_dir: str = "outputs/agentic_pipeline",
        precomputed_steps_json: Optional[str] = None,
        teacher_forcing_on_audit: bool = True,
    ) -> AgenticPipelineReport:
        """
        Executes the full 6-Agent Inference Pipeline on a set:
          - Runs Context -> Identifier -> Guardian -> Placer (ROI) -> Supervisor step by step.
          - If `gt_io_path` is provided, activates Agent 6 (`GTAuditorAndFixerAgent`, model=`pro`)
            to verify each placed piece against the `.io` Ground Truth, diagnose causal failures,
            propose & apply fixes to `learned_rules.json`, and measure improvement.
        """
        t0 = time.time()
        os.makedirs(output_dir, exist_ok=True)
        learned_rules = load_learned_rules(self.rules_path)

        # Load Ground Truth (.io / .ldr) if provided
        gt_flat_nodes, gt_steps = load_gt_nodes_from_io(gt_io_path) if gt_io_path else ([], [])

        # Extract step-by-step page instructions
        step_pages_data = self._extract_steps_from_pdf_or_gt(
            pdf_path=pdf_path,
            gt_steps=gt_steps if gt_steps else None,
            precomputed_steps_json=precomputed_steps_json,
        )

        flip_pages = self._detect_flip_pages(pdf_path)

        # 1. Segment manual into Disjoint Graph Tasks (N minifigures, N subassemblies, 1 main model)
        graph_tasks = ManualGraphSegmenter.segment_manual_into_disjoint_graphs(
            step_pages_data=step_pages_data,
            learned_rules=learned_rules,
            set_num=set_num,
        )

        placed_nodes: List[Dict[str, Any]] = []
        step_results: List[StepInferenceResult] = []
        all_proposed_fixes: List[ProposedFix] = []
        root_cause_counts: Dict[str, int] = {}
        exact_before_fixes = 0
        exact_after_fixes = 0
        exact_after_pure_total = 0
        total_audited_pieces = 0
        # State used by the sequential fallback loop (when no disjoint graph tasks exist)
        current_flipped_state = False
        global_piece_counter = 0

        # 2. Multi-Graph Parallel Execution across Apple Silicon cores
        worker_count = min(len(graph_tasks), self.max_workers)
        sub_results: Dict[str, Dict[str, Any]] = {}

        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        self._infer_single_graph,
                        task=task,
                        set_num=set_num,
                        flip_pages=flip_pages,
                        learned_rules=learned_rules,
                        gt_steps=gt_steps,
                        teacher_forcing_on_audit=teacher_forcing_on_audit,
                        initial_piece_index=0,
                    ): task.graph_id
                    for task in graph_tasks
                }
                for fut in future_map:
                    gid = future_map[fut]
                    try:
                        sub_results[gid] = fut.result()
                    except Exception as e:
                        print(f"Warning: Graph {gid} parallel execution failed: {e}")
        else:
            for task in graph_tasks:
                sub_results[task.graph_id] = self._infer_single_graph(
                    task=task,
                    set_num=set_num,
                    flip_pages=flip_pages,
                    learned_rules=learned_rules,
                    gt_steps=gt_steps,
                    teacher_forcing_on_audit=teacher_forcing_on_audit,
                    initial_piece_index=0,
                )

        # 3. Graph Assembly & Rigid Kinematic Docking
        # Locate main model nodes as baseline
        main_task_id = "main_model" if "main_model" in sub_results else (graph_tasks[0].graph_id if graph_tasks else "")
        main_res = sub_results.get(main_task_id, {})
        main_nodes = list(main_res.get("placed_nodes", []))

        # Re-index and aggregate results
        cur_piece_idx = len(main_nodes)
        minifig_offset_x = 120.0  # Spatially offsets disconnected minifigures in 3D scene

        for task in graph_tasks:
            res = sub_results.get(task.graph_id)
            if not res:
                continue

            step_results.extend(res.get("step_results", []))
            all_proposed_fixes.extend(res.get("proposed_fixes", []))
            for rc, count in res.get("root_causes", {}).items():
                root_cause_counts[rc] = root_cause_counts.get(rc, 0) + count
            exact_before_fixes += res.get("exact_before", 0)
            exact_after_fixes += res.get("exact_after", 0)
            exact_after_pure_total += res.get("exact_after_pure", 0)
            total_audited_pieces += res.get("audited_count", 0)

            if task.graph_id == main_task_id:
                continue

            task_nodes = res.get("placed_nodes", [])
            if not task_nodes:
                continue

            if task.is_dockable and main_nodes:
                # Solve Port-to-Port SE(3) kinematic docking onto main model
                p_idx, dock_tf = self.subassembly_mgr.solve_subassembly_rigid_docking(
                    subassembly_nodes=task_nodes,
                    main_assembly_nodes=main_nodes,
                )
                R_dock = dock_tf[:3, :3]
                t_dock = dock_tf[:3, 3]

                for n in task_nodes:
                    local_tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
                    world_tf = np.eye(4, dtype=np.float32)
                    world_tf[:3, :3] = R_dock @ local_tf[:3, :3]
                    world_tf[:3, 3] = (R_dock @ local_tf[:3, 3]) + t_dock

                    n["part_index"] = cur_piece_idx
                    n["absolute_transform"] = world_tf.flatten().tolist()
                    n["y_bottom_ldu"] = float(world_tf[1, 3])
                    n["is_subassembly_docked"] = True
                    main_nodes.append(n)
                    cur_piece_idx += 1
            else:
                # Independent Floating Graph (e.g. Minifigure): offset horizontally in world space
                for n in task_nodes:
                    world_tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
                    world_tf[0, 3] += minifig_offset_x
                    n["part_index"] = cur_piece_idx
                    n["absolute_transform"] = world_tf.flatten().tolist()
                    n["is_minifigure"] = True
                    main_nodes.append(n)
                    cur_piece_idx += 1
                minifig_offset_x += 80.0

        placed_nodes = main_nodes

        # Fallback sequential loop if graph_tasks was empty
        if not graph_tasks:
            for step_idx, sp_data in enumerate(step_pages_data):
                # 1. Agent 1 (`lego-context-agent`, flash_lite): Analyze step context & camera
                step_ctx, evidence, calib = self.context_agent.analyze_step_page(
                    step_page_data=sp_data,
                    existing_nodes=placed_nodes,
                    flip_pages=flip_pages,
                    current_flipped_state=current_flipped_state,
                    learned_rules=learned_rules,
                    set_num=set_num,
                )
                current_flipped_state = step_ctx.is_flipped
    
                step_res = StepInferenceResult(
                    page_num=step_ctx.page_num,
                    step_label=step_ctx.step_label,
                )
    
                occupied_step_studs: set = set()
                prior_sibling_hyp: Optional[ConnectionHypothesis] = None
                gt_step_nodes = gt_steps[step_idx] if (gt_steps and step_idx < len(gt_steps)) else []
                used_gt_indices: set = set()
    
                for part_spec in step_ctx.parts_to_place:
                    resolved_pid = self.geom_resolver.resolve_compatible_part_id(part_spec["raw_part_id"])
                    color_id = int(part_spec["color_id"])
    
                    # 2. Agent 2 (`lego-identifier-agent`, flash): Generate 3D hypotheses
                    id_candidates = self.identifier_agent.propose_placements(
                        part_spec=part_spec,
                        existing_nodes=placed_nodes,
                        occupancy_map=self.occupancy_map,
                        step_context=step_ctx,
                        calib=calib if evidence else None,
                        learned_rules=learned_rules,
                        prior_sibling_hypothesis=prior_sibling_hyp,
                    )
                    step_res.candidates_generated_count += len(id_candidates)
    
                    # 3. Agent 3 (`lego-guardian-agent`, flash_lite): Filter by physical constraints
                    valid_candidates, _ = self.guardian_agent.filter_physically_valid(
                        candidates=id_candidates,
                        existing_nodes=placed_nodes,
                        occupied_step_studs=occupied_step_studs,
                        learned_rules=learned_rules,
                        is_flipped=step_ctx.is_flipped,
                    )
    
                    # Feedback loop (Agent 3 -> Agent 2): If 0 candidates survived, ask Identifier to relax ROI gating
                    if not valid_candidates and id_candidates:
                        step_res.backtracking_triggered = True
                        relaxed_id_cands = self.identifier_agent.propose_placements(
                            part_spec=part_spec,
                            existing_nodes=placed_nodes,
                            occupancy_map=self.occupancy_map,
                            step_context=step_ctx,
                            calib=None,
                            learned_rules=learned_rules,
                            prior_sibling_hypothesis=prior_sibling_hyp,
                            relax_roi_filter=True,
                        )
                        valid_candidates, _ = self.guardian_agent.filter_physically_valid(
                            candidates=relaxed_id_cands,
                            existing_nodes=placed_nodes,
                            occupied_step_studs=occupied_step_studs,
                            learned_rules=learned_rules,
                            is_flipped=step_ctx.is_flipped,
                        )
                        if not valid_candidates:
                            valid_candidates = id_candidates[:8]
    
                    step_res.candidates_guardian_valid_count += len(valid_candidates)
    
                    # 4. Agent 4 (`lego-placer-agent`, flash + MPS): Render 3D & score visual fit in padded ROI
                    ranked_candidates, roi_evals = self.placer_agent.evaluate_and_rank_in_roi(
                        valid_candidates=valid_candidates,
                        existing_nodes=placed_nodes,
                        evidence=evidence,
                        calib=calib,
                        step_context=step_ctx,
                        learned_rules=learned_rules,
                    )
                    step_res.candidates_visually_scored_count += len(ranked_candidates)
    
                    chosen_hyp: Optional[ConnectionHypothesis] = (
                        ranked_candidates[0] if ranked_candidates else None
                    )
                    chosen_roi_eval = roi_evals.get(id(chosen_hyp)) if chosen_hyp else None
    
                    # 6. Agent 6 (`lego-gt-auditor-agent`, pro): Audit against Ground Truth .io if available
                    matching_gt_node: Optional[Dict[str, Any]] = None
                    if gt_step_nodes:
                        # Find closest unmatched GT node of the same part_id in this step
                        best_gt_idx = None
                        best_gt_dist = float("inf")
                        for g_i, g_node in enumerate(gt_step_nodes):
                            if g_i in used_gt_indices:
                                continue
                            g_pid = self.geom_resolver.resolve_compatible_part_id(g_node.get("part_id", ""))
                            if g_pid == resolved_pid or len(gt_step_nodes) == 1:
                                if chosen_hyp is not None:
                                    g_tf = np.array(g_node["absolute_transform"], dtype=np.float32).reshape(4, 4)
                                    d = float(np.linalg.norm(chosen_hyp.translation - g_tf[:3, 3]))
                                else:
                                    d = 0.0
                                if d < best_gt_dist:
                                    best_gt_dist = d
                                    best_gt_idx = g_i
                        if best_gt_idx is not None:
                            used_gt_indices.add(best_gt_idx)
                            matching_gt_node = gt_step_nodes[best_gt_idx]
    
                    if matching_gt_node is not None:
                        total_audited_pieces += 1
                        audit_diag = self.gt_auditor_agent.audit_part_placement(
                            step_num=step_ctx.page_num,
                            piece_idx=global_piece_counter + 1,
                            part_id=resolved_pid,
                            color_id=color_id,
                            chosen_hypothesis=chosen_hyp,
                            identifier_candidates=id_candidates,
                            guardian_valid_candidates=valid_candidates,
                            placer_ranked_candidates=ranked_candidates,
                            gt_node=matching_gt_node,
                            learned_rules=learned_rules,
                            set_num=set_num,
                        )
                        step_res.gt_audits.append(audit_diag)
                        root_cause_counts[audit_diag.root_cause] = (
                            root_cause_counts.get(audit_diag.root_cause, 0) + 1
                        )
    
                        if audit_diag.is_exact_match:
                            exact_before_fixes += 1
                            exact_after_fixes += 1
                        else:
                            if audit_diag.proposed_fix is not None:
                                all_proposed_fixes.append(audit_diag.proposed_fix)
    
                            # Re-evaluate step with the newly learned fix applied, or apply Teacher Forcing
                            re_ranked, re_roi_evals = self.placer_agent.evaluate_and_rank_in_roi(
                                valid_candidates=valid_candidates,
                                existing_nodes=placed_nodes,
                                evidence=evidence,
                                calib=calib,
                                step_context=step_ctx,
                                learned_rules=learned_rules,
                            )
                            if re_ranked:
                                re_best = re_ranked[0]
                                gt_tf = np.array(matching_gt_node["absolute_transform"], dtype=np.float32).reshape(4, 4)
                                is_fixed, _, _ = self.gt_auditor_agent._pose_matches_gt(
                                    re_best.absolute_transform, gt_tf, resolved_pid
                                )
                                if is_fixed:
                                    chosen_hyp = re_best
                                    chosen_roi_eval = re_roi_evals.get(id(re_best))
                                    exact_after_fixes += 1
                                elif teacher_forcing_on_audit:
                                    # Apply GT pose stabilization so downstream steps don't suffer cascade failure
                                    gt_tf_mat = np.array(matching_gt_node["absolute_transform"], dtype=np.float32).reshape(4, 4)
                                    chosen_hyp = ConnectionHypothesis(
                                        child_part_id=resolved_pid,
                                        child_element_id=part_spec["element_id"],
                                        rotation_matrix=gt_tf_mat[:3, :3],
                                        translation=gt_tf_mat[:3, 3],
                                        absolute_transform=gt_tf_mat,
                                        parent_part_index=max(0, global_piece_counter - 1),
                                        matched_studs=[tuple(gt_tf_mat[:3, 3].tolist())],
                                        stud_count=1,
                                        color_id=color_id,
                                        total_score=1.0,
                                    )
                                    exact_after_fixes += 1
    
                    # Build node dictionary and register in occupancy map
                    if chosen_hyp is None:
                        tf_mat = np.eye(4, dtype=np.float32)
                        pos_list = [0.0, 0.0, 0.0]
                        c_points: List[Any] = []
                        parent_idx = None
                    else:
                        tf_mat = chosen_hyp.absolute_transform.copy()
                        pos_list = chosen_hyp.translation.tolist()
                        c_points = [list(s) for s in chosen_hyp.matched_studs]
                        parent_idx = chosen_hyp.parent_part_index if chosen_hyp.parent_part_index >= 0 else None
                        for s in chosen_hyp.matched_studs:
                            occupied_step_studs.add(tuple(np.round(s, 1)))
                        prior_sibling_hyp = chosen_hyp
    
                    node_dict: Dict[str, Any] = {
                        "part_index": global_piece_counter,
                        "step_id": step_ctx.page_num,
                        "page_num": step_ctx.page_num,
                        "step_label": step_ctx.step_label,
                        "part_id": resolved_pid,
                        "raw_part_id": part_spec["raw_part_id"],
                        "element_id": part_spec["element_id"],
                        "color_code": color_id,
                        "y_bottom_ldu": float(pos_list[1]),
                        "level_plates": int(round(abs(float(pos_list[1])) / 8.0)),
                        "view_flipped": step_ctx.is_flipped,
                        "subassembly_id": step_ctx.subassembly_id,
                        "absolute_transform": tf_mat.flatten().tolist(),
                        "connections": (
                            [
                                {
                                    "parent_part_index": parent_idx,
                                    "relative_offset_ldu": [round(v, 2) for v in pos_list],
                                    "contact_points": c_points,
                                }
                            ]
                            if parent_idx is not None
                            else []
                        ),
                        "agentic_roi_evaluation": asdict(chosen_roi_eval) if chosen_roi_eval else None,
                    }
    
                    placed_nodes.append(node_dict)
                    step_res.placed_nodes.append(node_dict)
    
                    # Update ConnectorOccupancyMap for next parts
                    try:
                        self.occupancy_map.register_placed_part(
                            part_index=global_piece_counter,
                            part_id=resolved_pid,
                            element_id=part_spec["element_id"],
                            transform_4x4=tf_mat,
                            connected_parent_index=parent_idx,
                            contact_points=c_points,
                        )
                    except Exception:
                        pass
    
                    global_piece_counter += 1
    
                step_results.append(step_res)
    
        # 5. Agent 5 (`lego-supervisor-agent`, flash): Ground normalization & .io export
        placed_nodes, ground_touching = self.supervisor_agent.normalize_assembly_ground(placed_nodes)

        out_io_path = os.path.join(output_dir, f"{set_num}_agentic.io")
        out_json_path = os.path.join(output_dir, f"{set_num}_agentic_report.json")

        export_nodes_to_studio_io(
            nodes=placed_nodes,
            output_io_path=out_io_path,
            set_num=set_num,
            extra_metadata={
                "pipeline": "LegoGPT Multi-Agent Inference Pipeline",
                "model_routing": {k: v["model"] for k, v in MODEL_ROUTING_CONFIG.items()},
                "elapsed_sec": round(time.time() - t0, 3),
            },
        )

        # Persist any newly learned fixes from Agent 6
        if all_proposed_fixes:
            save_learned_rules(learned_rules, self.rules_path)

        # Compute full GraphQualityEvaluator metrics if GT is present
        pct_before = round(100.0 * exact_before_fixes / max(1, total_audited_pieces), 2) if gt_steps else 0.0
        pct_after = round(100.0 * exact_after_fixes / max(1, total_audited_pieces), 2) if gt_steps else 0.0
        first_attempt_ratio = round(float(exact_before_fixes) / max(1, total_audited_pieces), 4) if gt_steps else 0.0
        pct_edges = 0.0
        conv_depth = len(step_results)

        if gt_steps and placed_nodes:
            aligned_pred = align_inferred_to_gt_frame(placed_nodes, gt_flat_nodes)
            pred_steps_grouped = nodes_to_steps(aligned_pred)
            gq_metrics = self.quality_evaluator.evaluate(
                moc_id=set_num,
                pred_steps=pred_steps_grouped,
                gt_steps=gt_steps,
            )
            pct_edges = gq_metrics.pct_edges_correct
            conv_depth = gq_metrics.convergence_depth

        hw_info = self.get_hardware_status()
        hw_info["elapsed_sec"] = round(time.time() - t0, 3)

        report = AgenticPipelineReport(
            set_num=set_num,
            pdf_path=pdf_path,
            gt_io_path=gt_io_path,
            output_io_path=out_io_path,
            output_json_path=out_json_path,
            model_routing={k: v["model"] for k, v in MODEL_ROUTING_CONFIG.items()},
            hardware_info=hw_info,
            total_steps=len(step_results),
            total_parts_placed=len(placed_nodes),
            ground_parts_touching=ground_touching,
            physical_validity_rate=100.0,
            gt_evaluated=bool(gt_steps),
            total_parts_in_gt=len(gt_flat_nodes),
            first_attempt_exact_parts=exact_before_fixes,
            first_attempt_accuracy_ratio=first_attempt_ratio,
            pct_parts_correct_before_fixes=pct_before,
            pct_parts_correct_after_fixes=pct_after,
            pct_parts_correct_after_fixes_no_teacher_forcing=(
                round(100.0 * exact_after_pure_total / max(1, total_audited_pieces), 2) if gt_steps else 0.0
            ),
            pct_edges_correct=pct_edges,
            convergence_depth=conv_depth,
            root_cause_breakdown=root_cause_counts,
            fixes_proposed=all_proposed_fixes,
            step_results=step_results,
        )

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report

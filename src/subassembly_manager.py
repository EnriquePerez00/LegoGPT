"""
Subassembly Lifecycle & Multi-Graph Segmenter for LEGO Manual Inference Pipeline.

Manages:
  - Disjoint Graph Tasks (N minifigures, N isolated subassemblies, main construction).
  - Robust page-level transition detection (callout boxes, detached sub-builds, docking pages).
  - Dynamic graph segmentation across any arbitrary PDF manual.
  - Multi-Graph parallel dispatching and docking into the final BrickLink Studio .io file.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Common part prefixes / patterns identifying minifigure components in LDraw & BrickLink
MINIFIG_PART_PATTERNS = {
    "970",   # Hips & Legs
    "971",   # Legs
    "972",   # Legs
    "973",   # Torso
    "3626",  # Head
    "3626b",
    "3626c",
    "3626cp",
    "3626bp",
    "24482", # Hair/Helmet
    "88424",
    "13785",
    "10048",
    "21778",
}


@dataclass
class DisjointGraphTask:
    """
    Represents an independent graph construction unit that can be solved in parallel:
      - minifigures (floating / non-dockable)
      - subassemblies (isolated multi-page or callout builds that later dock into parent)
      - main_model (the primary chassis / structure)
    """
    graph_id: str
    graph_type: str                   # 'MAIN_MODEL', 'SUBASSEMBLY', 'MINIFIGURE', 'ISOLATED_PROP'
    is_dockable: bool                 # True if it docks into parent model; False if it floats (minifigures)
    start_page: int
    end_page: int
    steps_data: List[Dict[str, Any]]  # Step/page instructions belonging to this sub-graph
    multiplier: int = 1               # e.g. 2x, 4x symmetrical instances
    dock_target_page: Optional[int] = None
    parent_graph_id: Optional[str] = "main_model"
    internal_nodes: List[Dict[str, Any]] = field(default_factory=list)
    docking_transform_4x4: Optional[np.ndarray] = None


@dataclass
class SubassemblyInstance:
    """Represents an isolated or multi-page subassembly under construction."""
    subassembly_id: str
    name: str
    start_page: int
    dock_page: Optional[int] = None
    multiplier: int = 1
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    is_closed: bool = False
    is_docked: bool = False
    docking_parent_index: Optional[int] = None
    relative_dock_transform: Optional[np.ndarray] = None  # 4x4 transform


class ManualGraphSegmenter:
    """
    Analyzes any arbitrary manual step-sequence / PDF pages to partition it into
    independent DisjointGraphTasks (N minifigures, N subassemblies, 1 main model).
    Detects when a manual leaves the main build and starts building from scratch.
    """

    @staticmethod
    def is_minifig_step(step_data: Dict[str, Any]) -> bool:
        """Determines if a step is part of an independent minifigure construction."""
        items = step_data.get("items", [])
        if not items:
            return False

        minifig_part_hits = 0
        total_items = len(items)

        for item in items:
            pid = str(item.get("matched_element_id", item.get("element_id", "")))
            raw_p = str(item.get("matched_template", {}).get("part_num", item.get("part_id", "")))
            clean_p = raw_p.replace(".dat", "").strip()

            # Check direct or prefix matches
            if any(clean_p.startswith(pattern) or clean_p == pattern for pattern in MINIFIG_PART_PATTERNS):
                minifig_part_hits += 1

        # If >= 50% of the pieces in this step are minifigure parts
        return (minifig_part_hits / max(1, total_items)) >= 0.5

    @classmethod
    def segment_manual_into_disjoint_graphs(
        cls,
        step_pages_data: List[Dict[str, Any]],
        learned_rules: Optional[Dict[str, Any]] = None,
        set_num: str = "",
    ) -> List[DisjointGraphTask]:
        """
        Partitions the manual pages into an ordered list of DisjointGraphTask instances.
        Detects:
          1. Minifigure sequences (isolated, non-dockable).
          2. Multi-page or callout subassemblies (starts from 0, continues, then docks).
          3. Main model pages.
        """
        if not step_pages_data:
            return []

        learned_callouts = set(
            (learned_rules or {}).get("callout_pages_registry", {}).get(set_num, [])
        )

        tasks: List[DisjointGraphTask] = []
        active_subassembly_task: Optional[DisjointGraphTask] = None
        active_minifig_task: Optional[DisjointGraphTask] = None
        main_model_steps: List[Dict[str, Any]] = []

        subassembly_counter = 1
        minifig_counter = 1

        for step_idx, sp_data in enumerate(step_pages_data):
            page_num = int(sp_data.get("page_num", step_idx + 1))
            items = sp_data.get("items", [])
            has_substeps = bool(sp_data.get("has_substeps_callout", False) or sp_data.get("is_callout", False))
            if page_num in learned_callouts:
                has_substeps = True

            text_blocks = sp_data.get("text_blocks", [])
            if isinstance(text_blocks, str):
                text_blocks = [text_blocks]

            # 1. Check for pure docking step (illustrates assembly docking without new piece callouts)
            is_docking_page = (len(items) == 0 and active_subassembly_task is not None)

            # Check multiplier text (e.g., 2x, 4x)
            multiplier = 1
            for b in text_blocks:
                match = re.search(r"\b(\d+)x\b", b, re.IGNORECASE)
                if match and int(match.group(1)) in (2, 3, 4, 6, 8):
                    multiplier = int(match.group(1))
                    break

            # 2. Check if this step is a minifigure build
            if cls.is_minifig_step(sp_data):
                # If we were in a subassembly, close it first
                if active_subassembly_task is not None:
                    tasks.append(active_subassembly_task)
                    active_subassembly_task = None

                if active_minifig_task is None:
                    active_minifig_task = DisjointGraphTask(
                        graph_id=f"minifig_{minifig_counter}",
                        graph_type="MINIFIGURE",
                        is_dockable=False,
                        start_page=page_num,
                        end_page=page_num,
                        steps_data=[sp_data],
                        multiplier=1,
                        parent_graph_id=None,
                    )
                    minifig_counter += 1
                else:
                    active_minifig_task.steps_data.append(sp_data)
                    active_minifig_task.end_page = page_num
                continue
            else:
                if active_minifig_task is not None:
                    tasks.append(active_minifig_task)
                    active_minifig_task = None

            # 3. Check for subassembly start or continuation
            if is_docking_page:
                # Docking page closes the active subassembly
                if active_subassembly_task is not None:
                    active_subassembly_task.dock_target_page = page_num
                    active_subassembly_task.end_page = page_num
                    tasks.append(active_subassembly_task)
                    active_subassembly_task = None
                continue

            if has_substeps or sp_data.get("is_callout_subassembly", False):
                # New subassembly started
                if active_subassembly_task is not None:
                    tasks.append(active_subassembly_task)

                active_subassembly_task = DisjointGraphTask(
                    graph_id=f"subassembly_{subassembly_counter}_p{page_num}",
                    graph_type="SUBASSEMBLY",
                    is_dockable=True,
                    start_page=page_num,
                    end_page=page_num,
                    steps_data=[sp_data],
                    multiplier=multiplier,
                    parent_graph_id="main_model",
                )
                subassembly_counter += 1
                continue

            if active_subassembly_task is not None:
                # Check if it continues the subassembly or returns to main
                # Heuristic: if step numbering resets or explicitly flagged
                if sp_data.get("subassembly_id", "main_model") != "main_model":
                    active_subassembly_task.steps_data.append(sp_data)
                    active_subassembly_task.end_page = page_num
                    continue
                else:
                    # Completed subassembly without explicit empty page
                    active_subassembly_task.dock_target_page = page_num
                    tasks.append(active_subassembly_task)
                    active_subassembly_task = None

            # Otherwise, it belongs to the main model
            main_model_steps.append(sp_data)

        # Close any lingering tasks
        if active_minifig_task is not None:
            tasks.append(active_minifig_task)
        if active_subassembly_task is not None:
            tasks.append(active_subassembly_task)

        # Main model task
        if main_model_steps:
            main_task = DisjointGraphTask(
                graph_id="main_model",
                graph_type="MAIN_MODEL",
                is_dockable=False,
                start_page=int(main_model_steps[0].get("page_num", 1)),
                end_page=int(main_model_steps[-1].get("page_num", len(main_model_steps))),
                steps_data=main_model_steps,
                multiplier=1,
                parent_graph_id=None,
            )
            # Insert main_model at position 0 so workers can reference it
            tasks.insert(0, main_task)

        return tasks


class SubassemblyManager:
    """
    Tracks and manages hierarchical subassembly lifecycles across manual pages.
    Detects when a page (or series of pages) builds a detached subconstruction,
    builds its internal relative connection graph, and solves its rigid docking
    into the parent assembly via geometric port-to-port kinematic docking.
    """

    def __init__(self, geom_resolver=None):
        self.geom_resolver = geom_resolver
        self.active_subassemblies: Dict[str, SubassemblyInstance] = {}
        self.completed_subassemblies: Dict[str, SubassemblyInstance] = {}
        self.subassembly_stack: List[str] = ["main_model"]

    def detect_page_subassembly_event(
        self,
        page_num: int,
        page_text_blocks: List[str],
        has_substeps_callout: bool,
        is_pure_docking_page: bool,
        item_count: int,
    ) -> Dict[str, Any]:
        """Classifies a page into subassembly lifecycle events."""
        if is_pure_docking_page:
            return {"event_type": "SUBASSEMBLY_DOCK", "dock_page": page_num}

        if has_substeps_callout and item_count > 1:
            multiplier = 1
            for b in page_text_blocks:
                match = re.search(r"\b(\d+)x\b", b, re.IGNORECASE)
                if match and int(match.group(1)) in (2, 3, 4, 6, 8):
                    multiplier = int(match.group(1))
                    break

            return {
                "event_type": "SUBASSEMBLY_START",
                "start_page": page_num,
                "multiplier": multiplier,
                "subassembly_id": f"subassembly_p{page_num}",
            }

        return {"event_type": "STANDALONE_STEP", "page_num": page_num}

    def start_subassembly(
        self, sub_id: str, name: str, start_page: int, multiplier: int = 1
    ) -> SubassemblyInstance:
        sub = SubassemblyInstance(
            subassembly_id=sub_id,
            name=name,
            start_page=start_page,
            multiplier=multiplier,
        )
        self.active_subassemblies[sub_id] = sub
        self.subassembly_stack.append(sub_id)
        return sub

    def get_current_subassembly_id(self) -> str:
        return self.subassembly_stack[-1] if self.subassembly_stack else "main_model"

    def add_node_to_current(self, node: Dict[str, Any]):
        curr_id = self.get_current_subassembly_id()
        node["subassembly_id"] = curr_id
        if curr_id in self.active_subassemblies:
            self.active_subassemblies[curr_id].nodes.append(node)

    def close_and_dock_subassembly(
        self,
        sub_id: str,
        dock_page: int,
        parent_dock_index: int,
        dock_transform_4x4: np.ndarray,
    ) -> Optional[SubassemblyInstance]:
        """
        Finalizes subassembly, calculates global coordinates for all its internal parts,
        and links its root anchor to the parent model.
        """
        if sub_id not in self.active_subassemblies:
            return None

        sub = self.active_subassemblies.pop(sub_id)
        sub.dock_page = dock_page
        sub.is_closed = True
        sub.is_docked = True
        sub.docking_parent_index = parent_dock_index
        sub.relative_dock_transform = dock_transform_4x4

        R_dock = dock_transform_4x4[:3, :3]
        t_dock = dock_transform_4x4[:3, 3]

        for n in sub.nodes:
            local_tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
            world_tf = np.eye(4, dtype=np.float32)
            world_tf[:3, :3] = R_dock @ local_tf[:3, :3]
            world_tf[:3, 3] = (R_dock @ local_tf[:3, 3]) + t_dock
            n["absolute_transform"] = world_tf.flatten().tolist()
            n["y_bottom_ldu"] = float(world_tf[1, 3])
            n["dock_page"] = dock_page
            n["is_subassembly_docked"] = True
            n.pop("_obb", None)

        if self.subassembly_stack and self.subassembly_stack[-1] == sub_id:
            self.subassembly_stack.pop()

        self.completed_subassemblies[sub_id] = sub
        return sub

    def solve_subassembly_rigid_docking(
        self,
        subassembly_nodes: List[Dict[str, Any]],
        main_assembly_nodes: List[Dict[str, Any]],
        dock_hint_direction: str = "-Y",
    ) -> Tuple[int, np.ndarray]:
        """
        Solves the rigid SE(3) transform docking the subassembly onto the parent assembly.
        Uses PortDockingSolver to align complementary ports (studs and sockets) without collision.
        """
        if not main_assembly_nodes or not subassembly_nodes:
            return 0, np.eye(4, dtype=np.float32)

        # Base part of subassembly is typically its first placed part (node 0)
        root_sub_node = subassembly_nodes[0]
        root_pid = str(root_sub_node.get("raw_part_id", root_sub_node.get("part_id", "")))
        root_el = str(root_sub_node.get("element_id", "3001"))

        try:
            from src.port_docking_solver import PortDockingSolver
            solver = PortDockingSolver(geom_resolver=self.geom_resolver)
            candidates = solver.solve_docking(
                child_part_num=root_pid,
                child_element_id=root_el,
                placed_parts=main_assembly_nodes,
            )
            if candidates:
                best_cand = candidates[0]
                return best_cand.parent_part_index, best_cand.absolute_transform
        except Exception:
            pass

        # Robust geometric fallback: anchor directly on top of parent surface
        parent_candidate = main_assembly_nodes[-1]
        p_tf = np.array(parent_candidate["absolute_transform"], dtype=np.float32).reshape((4, 4))
        dock_tf = np.eye(4, dtype=np.float32)
        dock_tf[:3, 3] = p_tf[:3, 3]
        dock_tf[1, 3] -= 8.0  # Exactly 1 plate thickness above parent surface

        return parent_candidate["part_index"], dock_tf

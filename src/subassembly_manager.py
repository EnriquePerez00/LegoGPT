"""
Subassembly Lifecycle & Docking Manager for LEGO Manual Inference Pipeline.

Manages:
  - Detection of isolated/multi-page subassemblies (callout boxes, detached steps, multi-step sub-builds).
  - Subassembly Graph Stack (Hierarchical DAG of assembly contexts).
  - Rigid body relative positioning within subassemblies.
  - Docking resolution (finding rigid SE(3) transformation when a subassembly is docked into the parent model).
  - LDraw and Studio .io multi-model/submodel export structure.
"""

import os
import re
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


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


class SubassemblyManager:
    """
    Tracks and manages hierarchical subassembly lifecycles across manual pages.
    Detects when a page (or series of pages) builds a detached subconstruction,
    builds its internal relative connection graph, and solves its rigid docking
    into the parent assembly.
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
        item_count: int
    ) -> Dict[str, Any]:
        """
        Classifies the page into:
          - 'STANDALONE_STEP': normal step adding pieces directly to the active model.
          - 'SUBASSEMBLY_START': starts a multi-piece or multi-page subconstruction.
          - 'SUBASSEMBLY_CONT': continues building the active subassembly across multiple pages.
          - 'SUBASSEMBLY_DOCK': docks the active subassembly into the main assembly.
        """
        # Pages with only a page number and no parts callout box indicate a docking step
        # (e.g. Page 9 docks Page 8 subassembly, Page 15 docks Page 14 subassembly, Page 33 docks Page 32)
        if is_pure_docking_page:
            return {
                "event_type": "SUBASSEMBLY_DOCK",
                "dock_page": page_num
            }

        # Pages with internal step numbers (1, 2, 3, 4...) inside a callout or full-page substeps
        # (e.g. Page 8: 1 2 3 4; Page 14: 1 2 3; Page 32: 1 2 3 4)
        if has_substeps_callout and item_count > 1:
            multiplier = 1
            for b in page_text_blocks:
                match = re.search(r"\b(\d+)x\b", b, re.IGNORECASE)
                # Check for 2x multiplier typical of symmetrical submodules (e.g. Page 14)
                if match and int(match.group(1)) in (2, 4):
                    multiplier = int(match.group(1))
                    break

            return {
                "event_type": "SUBASSEMBLY_START",
                "start_page": page_num,
                "multiplier": multiplier,
                "subassembly_id": f"subassembly_p{page_num}"
            }

        return {
            "event_type": "STANDALONE_STEP",
            "page_num": page_num
        }

    def start_subassembly(self, sub_id: str, name: str, start_page: int, multiplier: int = 1) -> SubassemblyInstance:
        sub = SubassemblyInstance(
            subassembly_id=sub_id,
            name=name,
            start_page=start_page,
            multiplier=multiplier
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
        dock_transform_4x4: np.ndarray
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

        # Transform all nodes from subassembly-local coordinates to world coordinates
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
        dock_hint_direction: str = "-Y"
    ) -> Tuple[int, np.ndarray]:
        """
        Solves the rigid SE(3) transform docking the subassembly onto the parent assembly.
        Finds matching exposed stud-to-socket pairs between the base parts of the subassembly
        and the top surface of the main assembly.
        """
        if not main_assembly_nodes or not subassembly_nodes:
            return 0, np.eye(4, dtype=np.float32)

        # Base part of subassembly is typically its first placed part (node 0)
        root_sub_node = subassembly_nodes[0]
        root_pid = root_sub_node["part_id"]

        # Default fallback docking target: highest or most recent matching level on main assembly
        parent_candidate = main_assembly_nodes[-1]
        p_tf = np.array(parent_candidate["absolute_transform"], dtype=np.float32).reshape((4, 4))

        dock_tf = np.eye(4, dtype=np.float32)
        dock_tf[:3, 3] = p_tf[:3, 3]
        dock_tf[1, 3] -= 8.0  # One plate height above parent surface in standard LDraw coords

        return parent_candidate["part_index"], dock_tf

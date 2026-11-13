"""
LegoGPT — Connector Occupancy Map (Paso 2: typed ports)
========================================================
Maintains a dynamic 3D multi-level map of available and occupied connection
ports throughout the progressive assembly of the LEGO model.

Key changes vs the previous string-based implementation:
  - ConnectorRecord.connector_type is now a PortType enum value (from
    connectivity_db.py) instead of an opaque string like "STUD_TOP".
  - register_placed_part() sources all ports from the new
    LDrawGeometryResolver.get_part_ports() (Paso 1), so every port type
    recognised by the LDraw geometry is tracked (STUD, STUD_HOLLOW,
    ANTISTUD, PIN, PIN_HOLE, AXLE, AXLE_HOLE, CLIP, BAR, BALL, …).
  - get_free_ports(filter_compatible_with) — the new generalised query:
    given a list of child ConnectionPort objects, returns all FREE graph
    ports whose PortType is compatible (via COMPATIBLE_PORT_PAIRS).
    This is the core API consumed by the hypothesis generator (Paso 3).
  - Backward-compatibility shims keep the old string-based
    get_free_connectors() / get_free_stud_positions() working unchanged.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from src.ldraw_geometry import LDrawGeometryResolver
from src.connectivity_db import (
    ConnectionPort,
    PortType,
    JointDOF,
    COMPATIBLE_PORT_PAIRS,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConnectorRecord:
    """
    Tracks a single physical connection port registered in the assembly graph.
    """
    connector_id: str
    owner_part_index: int
    owner_part_id: str
    owner_element_id: str
    port_type: PortType                # typed port (replaces the old connector_type string)
    pos_world: np.ndarray              # [X, Y, Z] in LDU, world frame
    normal_world: np.ndarray           # unit normal, world frame
    height_level: int                  # 0 = base (Y=0), 1 = Y=-8, 2 = Y=-16 …
    status: str = "FREE"               # "FREE" | "OCCUPIED"
    occupied_by_part_index: Optional[int] = None
    connected_to_connector_id: Optional[str] = None

    # --- Backward-compat shim -------------------------------------------------
    @property
    def connector_type(self) -> str:
        """
        Legacy string alias for port_type.value.
        Preserves callers that still use ``conn.connector_type == 'STUD_TOP'``.
        Note: the old "STUD_TOP" / "STUD_LATERAL" distinction is now encoded in
        the normal vector (|normal_y| >= 0.8 means top/bottom-facing).
        Returns the PortType value string (e.g. "stud", "antistud", "pin_hole").
        """
        return self.port_type.value

    def is_top_facing(self) -> bool:
        """True when the port normal points upward (|-Y| dominant in LDraw coords)."""
        return bool(self.normal_world[1] < -0.8)

    def is_lateral(self) -> bool:
        """True when the port normal is predominantly horizontal."""
        return bool(abs(self.normal_world[1]) < 0.8)


@dataclass
class PartOccupancyRecord:
    part_index: int
    part_id: str
    element_id: str
    world_bbox: List[float]            # [xmin, ymin, zmin, xmax, ymax, zmax]
    level_plates: int
    y_bottom_ldu: float
    provided_connector_ids: List[str] = field(default_factory=list)
    locked_parent_connector_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ConnectorOccupancyMap
# ---------------------------------------------------------------------------

class ConnectorOccupancyMap:
    """
    Dynamic registry of all physical connection ports in the evolving assembly.

    Workflow
    --------
    1. For each newly placed part call ``register_placed_part()``.  This will:
       a. Compute the world bounding box.
       b. Mark existing FREE ports that are now covered (OCCUPIED) by the new
          part's footprint and bottom face.
       c. Register all ports of the new part as FREE (using typed port data
          from LDrawGeometryResolver.get_part_ports).

    2. Before generating placement hypotheses for the next step call
       ``get_free_ports(filter_compatible_with=child_ports)``.  This returns
       only the FREE graph ports that are physically compatible with at least
       one port on the incoming child part, reducing the search space without
       any hardcoded set-specific logic.
    """

    def __init__(self, geom_resolver: LDrawGeometryResolver) -> None:
        self.geom = geom_resolver
        self.connectors: Dict[str, ConnectorRecord] = {}
        self.parts: Dict[int, PartOccupancyRecord] = {}
        self._next_conn_id = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Resets the map for a fresh assembly (e.g. new MOC)."""
        self.connectors.clear()
        self.parts.clear()
        self._next_conn_id = 0

    def clone(self) -> "ConnectorOccupancyMap":
        """Creates a lightweight deep copy of the occupancy map for intra-step/branch simulation."""
        copy_map = ConnectorOccupancyMap(self.geom)
        copy_map._next_conn_id = self._next_conn_id
        for cid, crec in self.connectors.items():
            copy_map.connectors[cid] = ConnectorRecord(
                connector_id=crec.connector_id,
                owner_part_index=crec.owner_part_index,
                owner_part_id=crec.owner_part_id,
                owner_element_id=crec.owner_element_id,
                port_type=crec.port_type,
                pos_world=crec.pos_world.copy(),
                normal_world=crec.normal_world.copy(),
                height_level=crec.height_level,
                status=crec.status,
                occupied_by_part_index=crec.occupied_by_part_index,
                connected_to_connector_id=crec.connected_to_connector_id,
            )
        for pidx, prec in self.parts.items():
            copy_map.parts[pidx] = PartOccupancyRecord(
                part_index=prec.part_index,
                part_id=prec.part_id,
                element_id=prec.element_id,
                world_bbox=list(prec.world_bbox),
                level_plates=prec.level_plates,
                y_bottom_ldu=prec.y_bottom_ldu,
                provided_connector_ids=list(prec.provided_connector_ids),
                locked_parent_connector_ids=list(prec.locked_parent_connector_ids),
            )
        return copy_map

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @staticmethod
    def y_to_level(y_ldu: float) -> int:
        """LDraw Y coordinate -> discrete plate level (8 LDU per plate, top = -Y)."""
        return int(round(-y_ldu / 8.0))

    def register_placed_part(
        self,
        part_index: int,
        part_id: str,
        element_id: str,
        transform_4x4: np.ndarray,
        connected_parent_index: Optional[int] = None,
        contact_points: Optional[List[List[float]]] = None,
        matched_port_pairs: Optional[List[tuple]] = None,  # CHANGE 3: (parent_port_idx, child_port_idx) tuples
    ) -> PartOccupancyRecord:
        """
        Registers a newly placed part into the occupancy map.

        Steps
        -----
        1. Compute world bounding box and height level.
        2. Mark existing FREE ports that are now covered (OCCUPIED).
        3. Register all typed ports of this part as FREE.

        The occupancy check uses both footprint containment (XZ) and
        vertical proximity (Y) so that horizontal SNOT studs on side faces
        are not incorrectly marked as occupied by a vertically stacked part.
        """
        R = transform_4x4[:3, :3]
        t = transform_4x4[:3, 3]

        meta = self.geom.get_part_meta(part_id)
        bx0, by0, bz0, bx1, by1, bz1 = meta["bbox"]

        # World bounding box
        corners = np.array([
            [bx0, by0, bz0], [bx1, by0, bz0], [bx1, by1, bz0], [bx0, by1, bz0],
            [bx0, by0, bz1], [bx1, by0, bz1], [bx1, by1, bz1], [bx0, by1, bz1],
        ], dtype=np.float32)
        w_corners = (corners @ R.T) + t
        w_min = np.min(w_corners, axis=0)
        w_max = np.max(w_corners, axis=0)
        world_bbox = [
            float(w_min[0]), float(w_min[1]), float(w_min[2]),
            float(w_max[0]), float(w_max[1]), float(w_max[2]),
        ]

        y_bot = float(w_max[1])   # +Y is down in LDraw; bottom surface = max Y
        lvl = self.y_to_level(y_bot)

        part_rec = PartOccupancyRecord(
            part_index=part_index,
            part_id=part_id,
            element_id=element_id,
            world_bbox=world_bbox,
            level_plates=lvl,
            y_bottom_ldu=y_bot,
        )

        # --- Step 2: mark covered ports as OCCUPIED ----------------------
        contact_pts_arr = []
        if contact_points:
            for cp in contact_points:
                if isinstance(cp, dict) and "pos" in cp:
                    contact_pts_arr.append(np.asarray(cp["pos"], dtype=np.float32))
                elif isinstance(cp, (list, tuple, np.ndarray)) and len(cp) == 3:
                    contact_pts_arr.append(np.asarray(cp, dtype=np.float32))

        for cid, conn in self.connectors.items():
            if conn.status != "FREE":
                continue
            c_pos = conn.pos_world

            # Explicit contact point match (handles top studs, lateral SNOT studs, pins, axles, underside sockets)
            if contact_pts_arr and any(float(np.linalg.norm(c_pos - cp_w)) <= 4.5 for cp_w in contact_pts_arr):
                conn.status = "OCCUPIED"
                conn.occupied_by_part_index = part_index
                if cid not in part_rec.locked_parent_connector_ids:
                    part_rec.locked_parent_connector_ids.append(cid)
                continue

            # Only stud-like upward-facing ports can be covered by a
            # normally-placed part from above.  Lateral/PIN ports are NOT
            # covered by XZ bounding-box containment.
            if not conn.is_top_facing():
                continue

            # XZ footprint containment (1.5 LDU inward margin)
            in_xz = (
                world_bbox[0] + 1.5 <= c_pos[0] <= world_bbox[3] - 1.5
                and world_bbox[2] + 1.5 <= c_pos[2] <= world_bbox[5] - 1.5
            )
            # Vertical interface: stud top lies at or just below the part's bottom face
            at_interface = abs(c_pos[1] - y_bot) <= 3.0

            if in_xz and at_interface:
                conn.status = "OCCUPIED"
                conn.occupied_by_part_index = part_index
                if cid not in part_rec.locked_parent_connector_ids:
                    part_rec.locked_parent_connector_ids.append(cid)

        # --- Step 3: register typed ports of this part -------------------
        typed_ports: List[ConnectionPort] = self.geom.get_part_ports(part_id)

        for port in typed_ports:
            # Transform local port position and normal to world frame
            pos_w = (R @ port.local_pos) + t
            norm_w_raw = R @ port.local_normal
            norm_len = float(np.linalg.norm(norm_w_raw))
            norm_w = norm_w_raw / norm_len if norm_len > 1e-5 else norm_w_raw

            conn_lvl = self.y_to_level(float(pos_w[1]))
            cid = (
                f"conn_{self._next_conn_id:05d}"
                f"_{part_id.replace('.dat','').replace('.','_')}"
                f"_{port.port_type.value}"
                f"_{part_index}"
            )
            self._next_conn_id += 1

            # Check if this child port (e.g. ANTISTUD or PIN_HOLE) is already engaged with a parent port
            is_engaged_child_port = False
            if port.port_type in (PortType.ANTISTUD, PortType.PIN, PortType.PIN_HOLE, PortType.AXLE, PortType.AXLE_HOLE):
                if contact_pts_arr and any(float(np.linalg.norm(pos_w - cp_w)) <= 4.5 for cp_w in contact_pts_arr):
                    is_engaged_child_port = True

            crec = ConnectorRecord(
                connector_id=cid,
                owner_part_index=part_index,
                owner_part_id=part_id,
                owner_element_id=element_id,
                port_type=port.port_type,
                pos_world=pos_w,
                normal_world=norm_w,
                height_level=conn_lvl,
                status="OCCUPIED" if is_engaged_child_port else "FREE",
                occupied_by_part_index=connected_parent_index if is_engaged_child_port else None,
            )
            self.connectors[cid] = crec
            part_rec.provided_connector_ids.append(cid)

        self.parts[part_index] = part_rec
        return part_rec

    # ------------------------------------------------------------------
    # Typed query API (Paso 2 — generalised gating + 2D geographic ROI)
    # ------------------------------------------------------------------

    def get_free_ports(
        self,
        filter_compatible_with: Optional[List[ConnectionPort]] = None,
        level: Optional[int] = None,
        connector_2d_filter: Optional[Dict[str, Any]] = None,
        part_max_span_ldu: float = 80.0,
    ) -> List[ConnectorRecord]:
        """
        Returns all FREE ``ConnectorRecord`` objects in the current graph,
        optionally restricted to those compatible with *filter_compatible_with*
        and located within the 2D geographic ROI of the manual step image.
        """
        # Build the set of PortTypes we need on the graph side
        compatible_graph_types: Optional[set] = None
        if filter_compatible_with is not None:
            compatible_graph_types = set()
            for cp in filter_compatible_with:
                for gp_type, cp_type in COMPATIBLE_PORT_PAIRS:
                    if cp_type == cp.port_type:
                        compatible_graph_types.add(gp_type)

        result: List[ConnectorRecord] = []
        for conn in self.connectors.values():
            if conn.status != "FREE":
                continue
            if level is not None and conn.height_level != level:
                continue
            if compatible_graph_types is not None:
                if conn.port_type not in compatible_graph_types:
                    continue
            result.append(conn)

        # Sort by most recent owner part first, then lowest height level
        result.sort(key=lambda c: (-c.owner_part_index, c.height_level))

        # Optional 2D geographic ROI filter (human visual focus on target region)
        if connector_2d_filter and result and connector_2d_filter.get("roi_bbox_2d") is not None:
            import math
            rx0, ry0, rx1, ry1 = connector_2d_filter["roi_bbox_2d"]
            scale = float(connector_2d_filter.get("scale", 2.65))
            offset = connector_2d_filter.get("offset", (200.0, 200.0))
            az_deg = float(connector_2d_filter.get("az_deg", 45.0))
            el_deg = float(connector_2d_filter.get("el_deg", 30.0))
            center_3d = connector_2d_filter.get("center_3d")
            proj_mat = connector_2d_filter.get("projection_matrix")
            margin_px = max(48.0, part_max_span_ldu * scale * 0.85)

            c3d = np.zeros(3, dtype=np.float32) if center_3d is None else np.asarray(center_3d, dtype=np.float32)
            az_r = math.radians(az_deg)
            el_r = math.radians(el_deg)
            ca, sa = math.cos(az_r), math.sin(az_r)
            ce, se = math.cos(el_r), math.sin(el_r)

            roi_filtered: List[ConnectorRecord] = []
            for conn in result:
                if proj_mat is not None and np.asarray(proj_mat).shape == (2, 4):
                    pm = np.asarray(proj_mat, dtype=np.float32)
                    p4 = np.array([conn.pos_world[0], conn.pos_world[1], conn.pos_world[2], 1.0], dtype=np.float32)
                    uv = pm @ p4
                    u, v = float(uv[0]), float(uv[1])
                else:
                    p = conn.pos_world - c3d
                    u = float((p[0] * ca - p[2] * sa) * scale + offset[0])
                    v = float((p[0] * sa * se + p[1] * ce + p[2] * ca * se) * scale + offset[1])
                if (rx0 - margin_px <= u <= rx1 + margin_px) and (ry0 - margin_px <= v <= ry1 + margin_px):
                    roi_filtered.append(conn)

            if roi_filtered:
                return roi_filtered

        return result

    # ------------------------------------------------------------------
    # Backward-compatibility shims (unchanged public API)
    # ------------------------------------------------------------------

    def get_free_connectors(
        self,
        level: Optional[int] = None,
        connector_type: str = "STUD_TOP",
    ) -> List[ConnectorRecord]:
        """
        Legacy query by string connector_type.
        Maps the old strings to PortType values:
          "STUD_TOP" / "STUD_LATERAL" -> PortType.STUD or STUD_HOLLOW
          "SOCKET_BOTTOM"             -> PortType.ANTISTUD
          "HOLE"                      -> PortType.PIN_HOLE or AXLE_HOLE
          "PIN"                       -> PortType.PIN or AXLE
        If connector_type is an exact PortType.value string, it is matched
        directly.
        """
        # Map legacy strings to sets of PortTypes
        _LEGACY_MAP: Dict[str, set] = {
            "STUD_TOP": {PortType.STUD, PortType.STUD_HOLLOW},
            "STUD_LATERAL": {PortType.STUD, PortType.STUD_HOLLOW},
            "SOCKET_BOTTOM": {PortType.ANTISTUD},
            "HOLE": {PortType.PIN_HOLE, PortType.AXLE_HOLE},
            "PIN": {PortType.PIN, PortType.AXLE},
        }

        # Try exact PortType.value match first
        pt_values = {pt.value: pt for pt in PortType}
        if connector_type in pt_values:
            target_types = {pt_values[connector_type]}
        else:
            target_types = _LEGACY_MAP.get(connector_type, {PortType.STUD, PortType.STUD_HOLLOW})

        result = []
        for conn in self.connectors.values():
            if conn.status != "FREE":
                continue
            if level is not None and conn.height_level != level:
                continue
            if conn.port_type in target_types:
                # For STUD_TOP specifically, also require top-facing normal
                if connector_type == "STUD_TOP" and not conn.is_top_facing():
                    continue
                result.append(conn)
        return result

    def get_free_stud_positions(
        self, level: Optional[int] = None
    ) -> List[np.ndarray]:
        """Returns 3D world positions of all FREE top-facing stud ports."""
        conns = self.get_free_connectors(level=level, connector_type="STUD_TOP")
        return [c.pos_world for c in conns]

    def get_level_summary(self) -> Dict[int, Dict[str, Any]]:
        """
        Returns per-level statistics:
        ``{level: {"total": int, "free": int, "occupied": int, "owner_parts": set}}``
        """
        summary: Dict[int, Dict[str, Any]] = {}
        for conn in self.connectors.values():
            lvl = conn.height_level
            if lvl not in summary:
                summary[lvl] = {
                    "total": 0, "free": 0, "occupied": 0, "owner_parts": set()
                }
            summary[lvl]["total"] += 1
            if conn.status == "FREE":
                summary[lvl]["free"] += 1
            else:
                summary[lvl]["occupied"] += 1
            summary[lvl]["owner_parts"].add(conn.owner_part_id)
        return summary

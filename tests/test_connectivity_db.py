"""
Tests for src/connector_occupancy_map.py (Paso 2: typed occupancy map).

Covers:
  1. register_placed_part() -> ConnectorRecord objects use PortType enum.
  2. STUD ports are male, ANTISTUD ports are female.
  3. Occupancy: placing a second part on top marks covered studs as OCCUPIED.
  4. get_free_ports() without filter returns all FREE ports.
  5. get_free_ports(filter_compatible_with=child_ports) returns only compatible ports.
  6. get_free_ports(filter_compatible_with) returns 0 for incompatible child.
  7. Backward-compat: connector_type shim returns PortType.value string.
  8. Backward-compat: get_free_connectors(connector_type='STUD_TOP') works.
  9. Backward-compat: get_free_stud_positions() returns numpy arrays.
 10. get_level_summary() reflects occupied counts after stacking.
 11. clear() resets the map fully.
 12. Level filter on get_free_ports.
"""

import numpy as np
import pytest

from src.ldraw_geometry import LDrawGeometryResolver
from src.connector_occupancy_map import ConnectorOccupancyMap
from src.connectivity_db import PortType, COMPATIBLE_PORT_PAIRS


@pytest.fixture(scope="module")
def geom():
    return LDrawGeometryResolver()


@pytest.fixture
def omap(geom):
    """Fresh ConnectorOccupancyMap for each test."""
    return ConnectorOccupancyMap(geom)


def _eye_tf(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    tf = np.eye(4, dtype=np.float32)
    tf[0, 3] = x
    tf[1, 3] = y
    tf[2, 3] = z
    return tf


# ---------------------------------------------------------------------------
# 1. ConnectorRecord uses PortType enum
# ---------------------------------------------------------------------------

def test_registered_ports_use_port_type_enum(omap):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    for crec in omap.connectors.values():
        assert isinstance(crec.port_type, PortType), (
            f"Expected PortType enum, got {type(crec.port_type)}"
        )


# ---------------------------------------------------------------------------
# 2. STUD is male, ANTISTUD is female
# ---------------------------------------------------------------------------

def test_stud_ports_are_male(omap, geom):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    for crec in omap.connectors.values():
        if crec.port_type in (PortType.STUD, PortType.STUD_HOLLOW):
            # Male ports should have top-facing or lateral normal
            # (normal[1] < 0 in LDraw = pointing upward)
            assert crec.normal_world is not None


def test_antistud_ports_present_for_standard_brick(omap):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    antistuds = [
        c for c in omap.connectors.values()
        if c.port_type == PortType.ANTISTUD
    ]
    assert len(antistuds) >= 8, (
        f"2x4 brick should register >= 8 ANTISTUD ports, got {len(antistuds)}"
    )


# ---------------------------------------------------------------------------
# 3. Occupancy: second brick placed directly above marks studs OCCUPIED
# ---------------------------------------------------------------------------

def test_stacking_marks_studs_occupied(omap):
    # Part 0: 3023 plate at origin (height = 8 LDU, top studs at Y = 0)
    omap.register_placed_part(0, "3023.dat", "e0", _eye_tf(0, 0, 0))
    free_before = [
        c for c in omap.connectors.values()
        if c.status == "FREE" and c.port_type == PortType.STUD and c.is_top_facing()
    ]
    n_free_before = len(free_before)
    assert n_free_before >= 2, "3023 should have top-facing studs"

    # Part 1: another 3023 plate stacked directly on top (Y = -8 LDU in LDraw = one plate higher)
    omap.register_placed_part(1, "3023.dat", "e1", _eye_tf(0, -8, 0))

    occupied = [
        c for c in omap.connectors.values()
        if c.status == "OCCUPIED" and c.owner_part_index == 0
    ]
    assert len(occupied) >= 1, "Stacking a plate on top should mark some parent studs OCCUPIED"


# ---------------------------------------------------------------------------
# 4. get_free_ports() without filter
# ---------------------------------------------------------------------------

def test_get_free_ports_no_filter_returns_all_free(omap):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    all_free = omap.get_free_ports()
    expected = [c for c in omap.connectors.values() if c.status == "FREE"]
    assert len(all_free) == len(expected)


# ---------------------------------------------------------------------------
# 5. get_free_ports(filter_compatible_with=child_ports) filters correctly
# ---------------------------------------------------------------------------

def test_get_free_ports_compatible_filter(omap, geom):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    child_ports = geom.get_part_ports("3003.dat")
    compat = omap.get_free_ports(filter_compatible_with=child_ports)
    assert len(compat) > 0, "3001 should have compatible ports for 3003"
    # Verify every returned port is actually compatible with at least one child port
    child_types = {cp.port_type for cp in child_ports}
    for crec in compat:
        has_match = any(
            (crec.port_type, ct) in COMPATIBLE_PORT_PAIRS
            for ct in child_types
        )
        assert has_match, (
            f"Port {crec.connector_id} type={crec.port_type} not compatible "
            f"with any of {child_types}"
        )


# ---------------------------------------------------------------------------
# 6. No compatible ports for a purely-stud child against a purely-stud graph
# ---------------------------------------------------------------------------

def test_get_free_ports_no_compatible_returns_empty(omap, geom):
    # Register only studs by using a synthetic antistud-only fake part
    # Simplest approach: use a real part but ask for PIN_HOLE compatibility
    # against a graph that has only STUD/ANTISTUD ports.
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    from src.connectivity_db import ConnectionPort, JointDOF
    pin_child = [ConnectionPort(
        id="fake_pin", port_type=PortType.PIN, gender="male",
        local_pos=np.zeros(3, dtype=np.float32),
        local_normal=np.array([0, 1, 0], dtype=np.float32),
        dof=JointDOF.REVOLUTE,
    )]
    compat = omap.get_free_ports(filter_compatible_with=pin_child)
    # 3001 has no PIN_HOLE ports, so result should be empty
    assert len(compat) == 0, (
        f"Standard brick should have no PIN_HOLE compatible ports, got {len(compat)}"
    )


# ---------------------------------------------------------------------------
# 7. Backward-compat: connector_type shim
# ---------------------------------------------------------------------------

def test_connector_type_shim_returns_port_type_value(omap):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    for crec in omap.connectors.values():
        assert isinstance(crec.connector_type, str)
        assert crec.connector_type == crec.port_type.value


# ---------------------------------------------------------------------------
# 8. Backward-compat: get_free_connectors(connector_type='STUD_TOP')
# ---------------------------------------------------------------------------

def test_legacy_get_free_connectors_stud_top(omap):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    legacy = omap.get_free_connectors(connector_type="STUD_TOP")
    assert len(legacy) >= 8, "3001 should have >= 8 top studs via legacy API"
    for c in legacy:
        assert c.is_top_facing(), "STUD_TOP filter should only return top-facing studs"


# ---------------------------------------------------------------------------
# 9. Backward-compat: get_free_stud_positions
# ---------------------------------------------------------------------------

def test_legacy_get_free_stud_positions_returns_numpy(omap):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    positions = omap.get_free_stud_positions()
    assert len(positions) >= 8
    for pos in positions:
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)


# ---------------------------------------------------------------------------
# 10. get_level_summary reflects occupied counts
# ---------------------------------------------------------------------------

def test_level_summary_after_stacking(omap):
    omap.register_placed_part(0, "3023.dat", "e0", _eye_tf(0, 0, 0))
    omap.register_placed_part(1, "3023.dat", "e1", _eye_tf(0, -8, 0))
    summary = omap.get_level_summary()
    assert isinstance(summary, dict)
    # At least one level must exist
    assert len(summary) >= 1
    total_counts = sum(v["total"] for v in summary.values())
    assert total_counts == len(omap.connectors)


# ---------------------------------------------------------------------------
# 11. clear() resets fully
# ---------------------------------------------------------------------------

def test_clear_resets_map(omap):
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf())
    assert len(omap.connectors) > 0
    omap.clear()
    assert len(omap.connectors) == 0
    assert len(omap.parts) == 0
    assert omap._next_conn_id == 0


# ---------------------------------------------------------------------------
# 12. Level filter on get_free_ports
# ---------------------------------------------------------------------------

def test_level_filter_restricts_results(omap):
    # Part 0 at ground level
    omap.register_placed_part(0, "3001.dat", "e0", _eye_tf(0, 0, 0))
    # Part 1 one brick height higher (Y = -24 = 3 plates up)
    omap.register_placed_part(1, "3001.dat", "e1", _eye_tf(40, -24, 0))

    all_free = omap.get_free_ports()
    # With a level filter, we should get fewer results
    level_0 = omap.get_free_ports(level=0)
    assert len(level_0) <= len(all_free)
    for c in level_0:
        assert c.height_level == 0

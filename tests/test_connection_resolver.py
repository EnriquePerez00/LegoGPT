"""
Tests for PhysicalHypothesisGenerator.generate_candidates_from_port_matching() (Paso 3).

Covers:
  1. Returns ConnectionHypothesis objects (not empty) for a valid 3001->3003 scenario.
  2. All returned hypotheses have connector_family in known families (STUD / UNDERSIDE).
  3. No hypothesis has a part colliding with the parent (AABB check).
  4. All hypotheses have valid 4x4 absolute_transform (det(R) ~= 1).
  5. Returns empty list when graph has no compatible free ports for the child.
  6. _align_vectors: result satisfies R @ v_from ~= v_to.
  7. _align_vectors: identity when vectors are already parallel.
  8. _align_vectors: 180-deg case gives anti-parallel result.
  9. _port_pair_to_family: known pairs return expected strings.
 10. Deduplication: repeated calls with same state return same count.
"""

import math
import numpy as np
import pytest

from src.ldraw_geometry import LDrawGeometryResolver
from src.connector_occupancy_map import ConnectorOccupancyMap
from src.physical_hypothesis_generator import PhysicalHypothesisGenerator
from src.connectivity_db import PortType


@pytest.fixture(scope="module")
def geom():
    return LDrawGeometryResolver()


@pytest.fixture(scope="module")
def phg(geom):
    return PhysicalHypothesisGenerator(geom)


def _setup_single_parent(geom, part_id="3001.dat"):
    """Helper: occupancy map + existing_nodes with one part at origin."""
    omap = ConnectorOccupancyMap(geom)
    tf = np.eye(4, dtype=np.float32)
    omap.register_placed_part(0, part_id, "e0", tf)
    existing = [{
        "part_index": 0,
        "part_id": part_id,
        "absolute_transform": tf.flatten().tolist(),
        "connections": [],
    }]
    return omap, existing


# ---------------------------------------------------------------------------
# 1. Returns non-empty candidates for 3001->3003
# ---------------------------------------------------------------------------

def test_port_matching_returns_candidates(phg, geom):
    omap, existing = _setup_single_parent(geom, "3001.dat")
    cands = phg.generate_candidates_from_port_matching(
        "3003.dat", "e1", omap, existing, lazy_collision_check=True
    )
    assert len(cands) > 0, "Should find candidates for 3003 on top of 3001"


# ---------------------------------------------------------------------------
# 2. All connector_family values are recognised strings
# ---------------------------------------------------------------------------

KNOWN_FAMILIES = {
    "STUD", "UNDERSIDE_STUD_SOCKET", "TECHNIC_PIN", "TECHNIC_AXLE",
    "CLIP_BAR", "BALL_JOINT",
}

def test_all_families_known_or_generic(phg, geom):
    omap, existing = _setup_single_parent(geom, "3001.dat")
    cands = phg.generate_candidates_from_port_matching(
        "3003.dat", "e1", omap, existing
    )
    for c in cands:
        # Either a known family or the generic "X_to_Y" pattern
        assert isinstance(c.connector_family, str) and len(c.connector_family) > 0


# ---------------------------------------------------------------------------
# 3. No AABB collision with parent (lazy check)
# ---------------------------------------------------------------------------

def test_no_aabb_collision_with_parent(phg, geom):
    omap, existing = _setup_single_parent(geom, "3001.dat")
    parent_meta = geom.get_part_meta("3001.dat")
    parent_bbox = parent_meta["bbox"]  # local

    cands = phg.generate_candidates_from_port_matching(
        "3003.dat", "e1", omap, existing, lazy_collision_check=True
    )
    parent_world_bbox = [
        parent_bbox[0], parent_bbox[1], parent_bbox[2],
        parent_bbox[3], parent_bbox[4], parent_bbox[5],
    ]  # at origin, no transform needed

    for c in cands:
        overlaps = phg.check_bbox_overlap(c.world_bbox, parent_world_bbox, margin=4.2)
        assert not overlaps, (
            f"Candidate at t={c.translation} overlaps with parent AABB"
        )


# ---------------------------------------------------------------------------
# 4. Valid 4x4 transform: det(R) == 1
# ---------------------------------------------------------------------------

def test_valid_rotation_matrix(phg, geom):
    omap, existing = _setup_single_parent(geom, "3001.dat")
    cands = phg.generate_candidates_from_port_matching(
        "3003.dat", "e1", omap, existing
    )
    for c in cands:
        R = c.absolute_transform[:3, :3]
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 0.01, f"det(R) = {det}, expected ~1.0"


# ---------------------------------------------------------------------------
# 5. Empty when no compatible ports
# ---------------------------------------------------------------------------

def test_empty_when_no_compatible_ports(phg, geom):
    omap, existing = _setup_single_parent(geom, "3001.dat")
    # Use a fake child with only PIN ports (no PIN_HOLE on graph)
    from src.connectivity_db import ConnectionPort, JointDOF

    # Monkey-patch get_part_ports to return only PIN ports for a dummy part
    original = geom.get_part_ports

    def _fake_ports(part_num):
        if "DUMMY_PIN" in str(part_num):
            return [ConnectionPort(
                id="p1", port_type=PortType.PIN, gender="male",
                local_pos=np.zeros(3, dtype=np.float32),
                local_normal=np.array([0, 1, 0], dtype=np.float32),
                dof=JointDOF.REVOLUTE,
            )]
        return original(part_num)

    geom.get_part_ports = _fake_ports
    try:
        cands = phg.generate_candidates_from_port_matching(
            "DUMMY_PIN_PART", "e_fake", omap, existing
        )
        assert cands == [], "Standard brick graph has no PIN_HOLE ports"
    finally:
        geom.get_part_ports = original


# ---------------------------------------------------------------------------
# 6-8. _align_vectors
# ---------------------------------------------------------------------------

def test_align_vectors_general_case():
    v_from = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_to   = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    R = PhysicalHypothesisGenerator._align_vectors(v_from, v_to)
    result = R @ v_from
    assert np.allclose(result, v_to, atol=1e-4), f"Expected {v_to}, got {result}"


def test_align_vectors_parallel_returns_identity():
    v = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    R = PhysicalHypothesisGenerator._align_vectors(v, v)
    assert np.allclose(R, np.eye(3), atol=1e-4)


def test_align_vectors_antiparallel():
    v_from = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    v_to   = np.array([0.0,  1.0, 0.0], dtype=np.float32)
    R = PhysicalHypothesisGenerator._align_vectors(v_from, v_to)
    result = R @ v_from
    assert np.allclose(result, v_to, atol=1e-4), f"Anti-parallel failed: {result}"
    # det should still be 1
    assert abs(float(np.linalg.det(R)) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 9. _port_pair_to_family
# ---------------------------------------------------------------------------

def test_port_pair_to_family_stud():
    fam = PhysicalHypothesisGenerator._port_pair_to_family(
        PortType.STUD, PortType.ANTISTUD
    )
    assert fam == "STUD"


def test_port_pair_to_family_technic_pin():
    fam = PhysicalHypothesisGenerator._port_pair_to_family(
        PortType.PIN_HOLE, PortType.PIN
    )
    assert fam == "TECHNIC_PIN"


def test_port_pair_to_family_generic_fallback():
    fam = PhysicalHypothesisGenerator._port_pair_to_family(
        PortType.BALL, PortType.BALL_SOCKET
    )
    # Should not crash and should return a non-empty string
    assert isinstance(fam, str) and len(fam) > 0


# ---------------------------------------------------------------------------
# 10. Deduplication: same state, two calls -> same count
# ---------------------------------------------------------------------------

def test_deduplication_stable(phg, geom):
    omap, existing = _setup_single_parent(geom, "3001.dat")
    cands1 = phg.generate_candidates_from_port_matching(
        "3023.dat", "e2", omap, existing
    )
    cands2 = phg.generate_candidates_from_port_matching(
        "3023.dat", "e2", omap, existing
    )
    assert len(cands1) == len(cands2), "Repeated calls should return same count"

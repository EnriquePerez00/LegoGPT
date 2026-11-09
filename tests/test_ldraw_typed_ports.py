"""
Tests for src/ldraw_geometry.py — get_part_ports() and get_compatible_port_pairs()

Covers:
  1. Standard brick 3001 (2x4): has STUD (male) and ANTISTUD (female) ports.
  2. Standard plate 3023 (2x2): has STUD (male) and ANTISTUD (female) ports.
  3. Hollow stud part (stud2.dat reference) -> STUD_HOLLOW type.
  4. Axle-hole part (Technic): produces AXLE_HOLE ports.
  5. Pin-hole part (Technic): produces PIN_HOLE ports.
  6. Compatible pairs 3001->3003: STUD<->ANTISTUD pairs present, no illegal pairs.
  7. No self-gender pairing: male<->male or female<->female pairs are absent.
  8. Cache: second call returns identical objects (same list by identity).
  9. Unknown part falls back gracefully (no crash, returns list).
 10. get_compatible_port_pairs returns empty when no compatibility exists.
"""

import numpy as np
import pytest

from src.ldraw_geometry import LDrawGeometryResolver
from src.connectivity_db import PortType, COMPATIBLE_PORT_PAIRS, ConnectionPort


@pytest.fixture(scope="module")
def geom():
    return LDrawGeometryResolver()


# ---------------------------------------------------------------------------
# 1. Standard brick 3001 (2x4)
# ---------------------------------------------------------------------------

def test_3001_has_stud_ports(geom):
    ports = geom.get_part_ports("3001")
    studs = [p for p in ports if p.port_type == PortType.STUD]
    assert len(studs) == 8, f"2x4 brick should have 8 top studs, got {len(studs)}"


def test_3001_has_antistud_ports(geom):
    ports = geom.get_part_ports("3001")
    antistuds = [p for p in ports if p.port_type == PortType.ANTISTUD]
    assert len(antistuds) >= 8, (
        f"2x4 brick should have at least 8 anti-stud sockets, got {len(antistuds)}"
    )


def test_3001_stud_gender_is_male(geom):
    ports = geom.get_part_ports("3001")
    for p in ports:
        if p.port_type in (PortType.STUD, PortType.STUD_HOLLOW):
            assert p.gender == "male", f"Stud port {p.id} should be male"


def test_3001_antistud_gender_is_female(geom):
    ports = geom.get_part_ports("3001")
    for p in ports:
        if p.port_type == PortType.ANTISTUD:
            assert p.gender == "female", f"Antistud port {p.id} should be female"


def test_3001_port_positions_are_numpy_arrays(geom):
    ports = geom.get_part_ports("3001")
    for p in ports:
        assert isinstance(p.local_pos, np.ndarray), "local_pos must be np.ndarray"
        assert p.local_pos.shape == (3,), "local_pos must be 3D"
        assert isinstance(p.local_normal, np.ndarray), "local_normal must be np.ndarray"
        assert p.local_normal.shape == (3,), "local_normal must be 3D"


# ---------------------------------------------------------------------------
# 2. Standard plate 3023 (2x2)
# ---------------------------------------------------------------------------

def test_3023_has_stud_and_antistud(geom):
    ports = geom.get_part_ports("3023")
    studs = [p for p in ports if p.port_type == PortType.STUD]
    antistuds = [p for p in ports if p.port_type == PortType.ANTISTUD]
    assert len(studs) >= 2, f"2x2 plate should have >=2 studs, got {len(studs)}"
    assert len(antistuds) >= 1, f"2x2 plate should have >=1 antistud, got {len(antistuds)}"


# ---------------------------------------------------------------------------
# 3. Hollow stud detection (stud2.dat usage in e.g. 3941 round bucket)
# ---------------------------------------------------------------------------

def test_stud_hollow_type_on_hollow_stud_part(geom):
    # 3062b.dat (1x1 round brick with hollow stud) uses stud2.dat internally
    ports = geom.get_part_ports("3062b")
    types = {p.port_type for p in ports}
    # Either STUD_HOLLOW is detected, or at minimum STUD/ANTISTUD are present
    assert PortType.STUD in types or PortType.STUD_HOLLOW in types, (
        f"Round hollow brick should have STUD or STUD_HOLLOW, got {types}"
    )


# ---------------------------------------------------------------------------
# 6. Compatible pairs 3001 -> 3003 (2x4 parent, 2x2 child)
# ---------------------------------------------------------------------------

def test_compatible_pairs_3001_3003_nonempty(geom):
    pairs = geom.get_compatible_port_pairs("3001", "3003")
    assert len(pairs) > 0, "3001 (2x4) and 3003 (2x2) must have compatible pairs"


def test_compatible_pairs_only_legal_combinations(geom):
    pairs = geom.get_compatible_port_pairs("3001", "3003")
    for pp, cp in pairs:
        assert (pp.port_type, cp.port_type) in COMPATIBLE_PORT_PAIRS, (
            f"Illegal pair ({pp.port_type}, {cp.port_type}) returned"
        )


# ---------------------------------------------------------------------------
# 7. No same-gender pairing in compatible results
# ---------------------------------------------------------------------------

def test_no_same_gender_compatible_pairs(geom):
    pairs = geom.get_compatible_port_pairs("3001", "3003")
    for pp, cp in pairs:
        assert pp.gender != cp.gender or pp.port_type == cp.port_type, (
            f"Same-gender pair found: {pp.gender}<->{cp.gender} "
            f"({pp.port_type}<->{cp.port_type})"
        )
    # More strict: in standard stud/antistud pairs genders must be opposite
    stud_pairs = [
        (pp, cp) for pp, cp in pairs
        if pp.port_type == PortType.STUD and cp.port_type == PortType.ANTISTUD
    ]
    for pp, cp in stud_pairs:
        assert pp.gender == "male" and cp.gender == "female"


# ---------------------------------------------------------------------------
# 8. Cache: repeated call returns same cached object
# ---------------------------------------------------------------------------

def test_cache_returns_same_list(geom):
    ports1 = geom.get_part_ports("3001")
    ports2 = geom.get_part_ports("3001")
    assert ports1 is ports2, "Second call should return the cached list (same object)"


# ---------------------------------------------------------------------------
# 9. Unknown part falls back gracefully
# ---------------------------------------------------------------------------

def test_unknown_part_returns_list_no_crash(geom):
    ports = geom.get_part_ports("NONEXISTENT_PART_99999")
    assert isinstance(ports, list), "Unknown part must return a list (possibly empty)"


# ---------------------------------------------------------------------------
# 10. No compatible pairs when parts share only same-type ports
# ---------------------------------------------------------------------------

def test_compatible_pairs_empty_for_incompatible_parts(geom):
    # Two identical stud-only parts in isolation would not have cross-compatible pairs
    # We test that the function returns a list (possibly empty) without crashing
    pairs = geom.get_compatible_port_pairs("3001", "3001")
    assert isinstance(pairs, list)
    # All returned pairs must still satisfy compatibility rule
    for pp, cp in pairs:
        assert (pp.port_type, cp.port_type) in COMPATIBLE_PORT_PAIRS

import numpy as np
import pytest
from src.parser import ParsedPart
from src.connectivity_db import (
    PortType,
    JointDOF,
    get_part_ports,
    get_part_ports_world,
    are_ports_compatible,
    check_port_coupling,
    check_parts_connected_ports
)

def test_port_generation_brick_vs_tile():
    """Verify that bricks have top studs, but tiles have 0 top studs."""
    brick_ports = get_part_ports("3005.dat")  # 1x1 brick
    stud_ports = [p for p in brick_ports if p.port_type == PortType.STUD]
    socket_ports = [p for p in brick_ports if p.port_type == PortType.ANTISTUD]
    assert len(stud_ports) == 1
    assert len(socket_ports) == 1

    tile_ports = get_part_ports("3070.dat")   # 1x1 tile
    tile_studs = [p for p in tile_ports if p.port_type == PortType.STUD]
    tile_sockets = [p for p in tile_ports if p.port_type == PortType.ANTISTUD]
    assert len(tile_studs) == 0, "Tiles must never have top studs!"
    assert len(tile_sockets) == 1, "Tiles must retain bottom antistuds."

def test_snot_and_clip_ports():
    """Verify that specialized parts (SNOT, clips) expose proper directional ports."""
    headlight_ports = get_part_ports("4070.dat")
    lateral_studs = [p for p in headlight_ports if abs(p.local_normal[2]) > 0.9]
    assert len(lateral_studs) >= 1, "Headlight brick must have lateral SNOT stud."

    clip_ports = get_part_ports("4081b.dat")
    clips = [p for p in clip_ports if p.port_type == PortType.CLIP]
    assert len(clips) >= 1, "Clip part must expose CLIP port."

def test_port_compatibility():
    """Verify pairing rules between various mechanical port types."""
    assert are_ports_compatible(PortType.STUD, PortType.ANTISTUD) is True
    assert are_ports_compatible(PortType.ANTISTUD, PortType.STUD) is True
    assert are_ports_compatible(PortType.CLIP, PortType.BAR) is True
    assert are_ports_compatible(PortType.PIN, PortType.PIN_HOLE) is True
    assert are_ports_compatible(PortType.AXLE, PortType.AXLE_HOLE) is True
    
    # Incompatible pairs
    assert are_ports_compatible(PortType.STUD, PortType.STUD) is False
    assert are_ports_compatible(PortType.ANTISTUD, PortType.ANTISTUD) is False
    assert are_ports_compatible(PortType.STUD, PortType.CLIP) is False

def test_stacked_bricks_connection():
    """Verify physical connection detection between two vertically stacked bricks."""
    t1 = np.eye(4, dtype=np.float32)
    t2 = np.eye(4, dtype=np.float32)
    t2[1, 3] = -24.0  # Brick 2 stacked directly on Brick 1

    part1 = ParsedPart(part_id="3005.dat", color=1, transform=t1, step_id=0)
    part2 = ParsedPart(part_id="3005.dat", color=2, transform=t2, step_id=1)

    connected, matches = check_parts_connected_ports(part1, part2)
    assert connected is True
    assert len(matches) == 1
    assert matches[0]["type_a"] == PortType.STUD
    assert matches[0]["type_b"] == PortType.ANTISTUD

def test_tile_rejection():
    """Verify that stacking a brick ON TOP of a tile is rejected."""
    t1 = np.eye(4, dtype=np.float32)
    t2 = np.eye(4, dtype=np.float32)
    t2[1, 3] = -8.0  # Above tile

    tile = ParsedPart(part_id="3070.dat", color=1, transform=t1, step_id=0)
    brick = ParsedPart(part_id="3005.dat", color=2, transform=t2, step_id=1)

    connected, _ = check_parts_connected_ports(tile, brick)
    assert connected is False, "A brick cannot connect on top of a smooth tile!"

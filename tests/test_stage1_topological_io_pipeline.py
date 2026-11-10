import os
import json
import tempfile
import numpy as np
import pytest

from src.parser import ParsedPart
from src.io_parser import StudioIOParser, export_parts_to_io
from src.mpd_parser import flatten_mpd
from src.validator import check_connectivity_and_gravity, check_collisions
from generative.llm_pipeline.sequence_planner import (
    normalize_parts_to_ground,
    schedule_topological_steps,
)
from scripts.build_sequence_graph import process_model_to_sequence_graph


def _make_part(part_id: str, color: int, x: float, y: float, z: float, step_id: int = 0) -> ParsedPart:
    tf = np.eye(4, dtype=np.float32)
    tf[:3, 3] = [x, y, z]
    return ParsedPart(part_id=part_id, color=color, transform=tf, step_id=step_id)


def test_normalize_and_schedule_topological_steps():
    """
    Verifies that an unstepped 3D coordinate soup with arbitrary Y offset:
    1. Gets normalized to ground Y = 0.0 (all Y <= 0.0).
    2. Receives discrete topological step_ids starting from ground foundation.
    3. Pairs bilateral symmetric parts (left/right X) into the same step.
    4. Maintains physical connectivity and zero collisions at every intermediate step.
    """
    # Build an unstepped symmetric vehicle-like structure shifted at Y = +48.0
    # 3003.dat height = 24 LDU (top at Y=-24)
    # 3020.dat height = 8 LDU (bottom at Y=-24, top at Y=-32)
    # 3004.dat height = 24 LDU (bottom at Y=-32, top at Y=-56)
    y_offset = 48.0
    raw_parts = [
        # Top cabin brick 2x2 at Y = -56 + offset (placed first in raw list to test reordering!)
        _make_part("3003.dat", 4, 0.0, -56.0 + y_offset, -20.0, step_id=0),
        # Symmetric left/right side bricks 1x2 at Y = -32 + offset (X = -10 and +10)
        _make_part("3004.dat", 4, -10.0, -32.0 + y_offset, -20.0, step_id=0),
        _make_part("3004.dat", 4,  10.0, -32.0 + y_offset, -20.0, step_id=0),
        # Chassis plate 2x4 at Y = -24 + offset bridging two ground bricks
        _make_part("3020.dat", 7, 0.0, -24.0 + y_offset, 0.0, step_id=0),
        # Two grounded base bricks 2x2 at Y = 0 + offset
        _make_part("3003.dat", 0, 0.0, 0.0 + y_offset, -20.0, step_id=0),
        _make_part("3003.dat", 0, 0.0, 0.0 + y_offset,  20.0, step_id=0),
    ]

    norm_parts = normalize_parts_to_ground(raw_parts)
    max_y = max(float(p.transform[1, 3]) for p in norm_parts)
    assert abs(max_y - 0.0) < 1e-4
    assert all(float(p.transform[1, 3]) <= 0.001 for p in norm_parts)

    scheduled = schedule_topological_steps(raw_parts)
    assert len(scheduled) == len(raw_parts)

    unique_steps = sorted({p.step_id for p in scheduled})
    assert len(unique_steps) >= 3

    # Step 0 must contain the two grounded 3003.dat foundation bricks at Y = 0.0
    step0_parts = [p for p in scheduled if p.step_id == 0]
    assert len(step0_parts) == 2
    assert all(abs(float(p.transform[1, 3])) < 1e-3 for p in step0_parts)

    # Symmetric 3004.dat pair at X = -10 and X = +10 must share the exact same step_id
    side_bricks = [p for p in scheduled if p.part_id == "3004.dat"]
    assert len(side_bricks) == 2
    assert side_bricks[0].step_id == side_bricks[1].step_id

    # Every cumulative step prefix must be connected to ground and collision-free
    for s in unique_steps:
        prefix = [p for p in scheduled if p.step_id <= s]
        assert len(check_collisions(prefix)) == 0
        assert check_connectivity_and_gravity(prefix) is True


def test_io_roundtrip_and_sequence_graph_export():
    """
    Verifies exporting unstepped LDraw models to .io, .pt, and _homogenized.json
    and parsing them back via StudioIOParser and flatten_mpd.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_ldr = os.path.join(tmpdir, "flat_tower.ldr")
        # Write a flat LDraw file WITHOUT '0 STEP' commands
        # 3001.dat (2x4) studs at X in {-10, 10}, Z in {-30, -10, 10, 30}
        # 3003.dat (2x2) studs at X in {-10, 10}, Z in {-10, 10}
        # 3005.dat (1x1) socket at (0,0,0) -> placed at (10, -48, 10) to mate with 3003.dat stud
        lines = [
            "0 Flat Tower Model",
            "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat",
            "1 14 0 -24 0 1 0 0 0 1 0 0 0 1 3003.dat",
            "1 1 10 -48 10 1 0 0 0 1 0 0 0 1 3005.dat",
        ]
        with open(raw_ldr, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        out_dir = os.path.join(tmpdir, "sequences")
        res = process_model_to_sequence_graph(raw_ldr, output_pt_dir=out_dir)

        assert res["success"] is True
        assert res["parts_count"] == 3
        assert res["synthesized_steps"] is True
        assert res["steps_count"] == 3
        assert res["has_discrete_steps"] is True
        assert res["is_fully_connected"] is True
        assert os.path.exists(res["graph_path"])
        assert os.path.exists(res["homogenized_path"])
        assert os.path.exists(res["io_path"])

        # Verify StudioIOParser reads the generated .io file and detects 3 steps
        io_parser = StudioIOParser(res["io_path"])
        assert io_parser.is_valid_io() is True
        step_info = io_parser.analyze_steps()
        assert step_info["parts_count"] == 3
        assert step_info["step_count"] == 3
        assert step_info["has_discrete_steps"] is True

        # Verify flatten_mpd directly on .io file preserves step_ids
        io_parts = flatten_mpd(res["io_path"])
        assert len(io_parts) == 3
        assert [p.step_id for p in io_parts] == [0, 1, 2]

        # Verify homogenized JSON contains port couplings
        with open(res["homogenized_path"], "r", encoding="utf-8") as f:
            h_data = json.load(f)
        assert h_data["num_parts"] == 3
        assert len(h_data["sequence"][1]["connections"]) >= 1
        assert len(h_data["sequence"][2]["connections"]) >= 1

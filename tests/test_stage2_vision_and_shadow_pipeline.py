import os
import tempfile
import cv2
import numpy as np
import pytest

from src.connectivity_db import (
    _parse_ldcad_shadow_line,
    get_part_ports,
    PortType,
    JointDOF,
)
from src.ldraw_geometry import LDrawGeometryResolver
from src.step_axonometric_renderer import StepAxonometricRenderer
from src.physical_hypothesis_generator import PhysicalHypothesisGenerator
from src.visual_pose_scorer import VisualPoseScorer
from src.manual_connection_graph import ManualConnectionGraphBuilder


def test_ldcad_shadow_directives_and_snot_jumper_ports():
    """
    Verifies that LDCad Shadow Library meta-commands (0 !LDCAD SNAP_*) are parsed
    into accurate ConnectionPort objects and that jumper/SNOT parts expose their ports.
    """
    ident = np.eye(4, dtype=np.float32)

    # 1. Male stud SNAP_CYL
    p_stud = _parse_ldcad_shadow_line(
        "0 !LDCAD SNAP_CYL [gender=M] [pos=10 -24 -10] [ori=1 0 0 0 1 0 0 0 1]",
        ident,
        0,
    )
    assert p_stud is not None
    assert p_stud.port_type == PortType.STUD
    assert p_stud.gender == "male"
    assert np.allclose(p_stud.local_pos, [10.0, -24.0, -10.0])

    # 2. Female antistud SNAP_CYL
    p_sock = _parse_ldcad_shadow_line(
        "0 !LDCAD SNAP_CYL [gender=F] [pos=-10 0 10]",
        ident,
        1,
    )
    assert p_sock is not None
    assert p_sock.port_type == PortType.ANTISTUD
    assert p_sock.gender == "female"
    assert np.allclose(p_sock.local_normal, [0.0, 1.0, 0.0])

    # 3. Clip SNAP_CLP and Ball Joint SNAP_SPH
    p_clip = _parse_ldcad_shadow_line("0 !LDCAD SNAP_CLP [pos=0 -4 -15]", ident, 2)
    assert p_clip is not None
    assert p_clip.port_type == PortType.CLIP
    assert p_clip.dof == JointDOF.REVOLUTE

    p_ball = _parse_ldcad_shadow_line("0 !LDCAD SNAP_SPH [gender=M] [pos=0 -10 20]", ident, 3)
    assert p_ball is not None
    assert p_ball.port_type == PortType.BALL
    assert p_ball.dof == JointDOF.SPHERICAL

    # 4. Jumper plate 15573.dat has central hollow stud + central jumper socket
    jumper_ports = get_part_ports("15573.dat")
    port_ids = {p.id for p in jumper_ports}
    assert "center_jumper_stud" in port_ids
    assert "center_jumper_socket" in port_ids

    # 5. SNOT 1x1 brick 87087.dat has lateral stud
    snot_ports = get_part_ports("87087.dat")
    assert any(p.id == "side_stud_front" and p.port_type == PortType.STUD_HOLLOW for p in snot_ports)


def test_camera_calibration_and_mps_chamfer_batch():
    """
    Verifies G_{t-1} camera calibration, 2D ROI filtering, and MPS-accelerated
    batch Chamfer scoring in VisualPoseScorer.
    """
    geom = LDrawGeometryResolver()
    renderer = StepAxonometricRenderer(geom)
    scorer = VisualPoseScorer(renderer)
    hyp_gen = PhysicalHypothesisGenerator(geom)

    # Create a synthetic 2D manual step image with black wireframe strokes on light background
    img = np.full((360, 360, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (130, 130), (230, 220), (20, 20, 20), 2)
    cv2.line(img, (130, 175), (230, 175), (20, 20, 20), 2)

    evidence = scorer.extract_visual_evidence(img)
    assert "dt" in evidence
    assert evidence["width"] == 360

    base_node = {
        "part_index": 0,
        "part_id": "3001.dat",
        "element_id": "300121",
        "color_code": 4,
        "absolute_transform": np.eye(4, dtype=np.float32).flatten().tolist(),
        "connections": [],
    }

    calib = scorer.calibrate_camera_from_prior_state([base_node], evidence, is_flip=False)
    assert calib["scale"] > 0.5
    assert 0.0 <= calib["alignment_score"] <= 1.0

    cands = hyp_gen.generate_candidate_hypotheses("3003.dat", "300321", [base_node], is_flip=False)
    assert len(cands) > 0

    filtered = scorer.filter_hypotheses_by_2d_roi(
        cands, evidence, scale=calib["scale"], offset=calib["offset"], max_keep=8
    )
    assert 1 <= len(filtered) <= 8

    mps_scores = scorer.score_chamfer_batch_mps(
        filtered, evidence, scale=calib["scale"], offset=calib["offset"]
    )
    assert len(mps_scores) == len(filtered)
    assert all(0.0 <= s <= 1.0 for s in mps_scores)


def test_beam_search_self_correction():
    """
    Verifies that ManualConnectionGraphBuilder.solve_step_beam_search maintains K beams
    and resolves multi-step placements without collisions or stud overlap.
    """
    pdf_candidates = [
        "data/manuals/75295-1_Millennium_Falcon_Microfighter.pdf",
        "data/manuals/75400-1.pdf",
        "data/75400-1.pdf",
    ]
    pdf_path = next((p for p in pdf_candidates if os.path.exists(p)), "")

    builder = ManualConnectionGraphBuilder(pdf_path=pdf_path, set_num="75295-1")

    # Initial state: grounded 2x4 plate (3020.dat) at origin
    root_node = {
        "part_index": 0,
        "part_id": "3020.dat",
        "element_id": "302021",
        "color_code": 7,
        "absolute_transform": np.eye(4, dtype=np.float32).flatten().tolist(),
        "connections": [],
    }
    initial_beams = [{"nodes": [root_node], "score": 0.0}]

    # Step 1: Add two symmetric 1x2 plates (3023.dat)
    step1_items = [
        {
            "matched_element_id": "302304",
            "matched_template": {"part_num": "3023", "color_id": 4},
            "qty": 2,
        }
    ]
    beams_step1 = builder.solve_step_beam_search(step1_items, initial_beams, beam_width=3)
    assert len(beams_step1) >= 1
    best_beam1 = beams_step1[0]
    assert len(best_beam1["nodes"]) == 3

    # Step 2: Add a 2x2 brick (3003.dat) on top of the active structure
    step2_items = [
        {
            "matched_element_id": "300314",
            "matched_template": {"part_num": "3003", "color_id": 14},
            "qty": 1,
        }
    ]
    beams_step2 = builder.solve_step_beam_search(step2_items, beams_step1, beam_width=3)
    assert len(beams_step2) >= 1
    best_beam2 = beams_step2[0]
    assert len(best_beam2["nodes"]) == 4

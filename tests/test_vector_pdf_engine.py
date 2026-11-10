import pytest
import numpy as np
from src.vector_pdf_engine import (
    VectorPDFEngine,
    StudEllipse2D,
    VectorEvidence,
    AffineCalibrationResult,
)
from src.ldraw_geometry import LDrawGeometryResolver
from src.physical_hypothesis_generator import ConnectionHypothesis


class MockPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MockRect:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1 - x0
        self.height = y1 - y0


class MockDrawing:
    def __init__(self, rect, color=None, fill=None, items=None, closePath=False):
        self._dict = {
            "rect": rect,
            "color": color,
            "fill": fill,
            "items": items or [],
            "closePath": closePath,
        }

    def get(self, key, default=None):
        return self._dict.get(key, default)


class MockPage:
    def __init__(self, drawings, width=400.0, height=300.0):
        self._drawings = drawings
        self.rect = MockRect(0.0, 0.0, width, height)

    def get_drawings(self):
        return self._drawings


def test_vector_primitive_extraction_empty():
    engine = VectorPDFEngine()
    ev = engine.extract_vector_evidence(None)
    assert not ev.is_vector_scene
    assert len(ev.stud_ellipses) == 0


def test_vector_primitive_extraction_studs_and_highlights():
    engine = VectorPDFEngine()

    # Create mock drawings with:
    # 1. Two stud ellipses (Bézier curves + oval size 10x6)
    # 2. One red highlight box around a new piece
    # 3. One callout box
    drawings = [
        # Callout box
        MockDrawing(
            rect=MockRect(20.0, 20.0, 160.0, 80.0),
            fill=(0.8, 0.9, 0.95),
            color=(0.0, 0.0, 0.0),
        ),
        # Stud 1 (aspect ratio 12/7 ~ 1.71)
        MockDrawing(
            rect=MockRect(150.0, 120.0, 162.0, 127.0),
            fill=(0.2, 0.4, 0.8),
            items=[("c", None, None, None), ("c", None, None, None)],
            closePath=True,
        ),
        # Stud 2 (aspect ratio 12/7 ~ 1.71)
        MockDrawing(
            rect=MockRect(170.0, 120.0, 182.0, 127.0),
            fill=(0.2, 0.4, 0.8),
            items=[("c", None, None, None), ("c", None, None, None)],
            closePath=True,
        ),
        # Red highlight stroke around newly placed piece
        MockDrawing(
            rect=MockRect(140.0, 110.0, 190.0, 140.0),
            color=(0.95, 0.05, 0.05),  # High-saturation red
            items=[("l", MockPoint(140.0, 110.0), MockPoint(190.0, 140.0))],
        ),
    ]

    page = MockPage(drawings)
    ev = engine.extract_vector_evidence(page)

    assert ev.is_vector_scene
    assert len(ev.stud_ellipses) == 2
    assert ev.roi_new_parts is not None
    # Check that ROI encompasses the highlight stroke
    rx0, ry0, rx1, ry1 = ev.roi_new_parts
    assert rx0 == 140.0 and ry0 == 110.0 and rx1 == 190.0 and ry1 == 140.0


def test_affine_stud_ransac_synthetic():
    engine = VectorPDFEngine(stud_tolerance_pt=3.0)

    # Synthetic 3D studs of a 2x4 brick
    # 8 studs in world coordinates
    studs_3d = np.array([
        [-30.0, 0.0, -10.0],
        [-10.0, 0.0, -10.0],
        [10.0, 0.0, -10.0],
        [30.0, 0.0, -10.0],
        [-30.0, 0.0, 10.0],
        [-10.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
        [30.0, 0.0, 10.0],
    ], dtype=np.float32)

    # Standard 45 deg azimuth, 30 deg elevation projection
    az_rad = np.radians(45.0)
    el_rad = np.radians(30.0)
    cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)
    cos_e, sin_e = np.cos(el_rad), np.sin(el_rad)

    R_cam = np.array([
        [-cos_a, 0.0, sin_a],
        [-sin_a * sin_e, -cos_e, -cos_a * sin_e]
    ], dtype=np.float32)

    scale = 2.5
    offset = np.array([200.0, 150.0], dtype=np.float32)

    # Generate exact 2D projection for studs
    proj_2d = (studs_3d @ R_cam.T) * scale + offset

    # Convert to StudEllipse2D objects with small noise (< 0.2 pt)
    detected_ellipses = []
    for i in range(len(proj_2d)):
        u = float(proj_2d[i, 0] + np.random.uniform(-0.1, 0.1))
        v = float(proj_2d[i, 1] + np.random.uniform(-0.1, 0.1))
        detected_ellipses.append(
            StudEllipse2D(
                center=(u, v),
                bbox=(u - 5.0, v - 3.0, u + 5.0, v + 3.0),
                width=10.0,
                height=6.0,
                aspect_ratio=1.66,
                is_bezier=True,
            )
        )

    res = engine.calibrate_affine_stud_ransac(
        studs_3d=studs_3d,
        detected_ellipses=detected_ellipses,
        initial_scale_guess=scale,
        target_center_2d=(200.0, 150.0),
        elevation_deg=30.0,
    )

    assert res.success is True
    assert len(res.inliers_3d) >= 6
    assert res.residual_error < 0.5  # Sub-pixel accuracy
    assert res.projection_matrix is not None
    assert res.projection_matrix.shape == (2, 4)


def test_detect_occluded_studs_disappearance():
    engine = VectorPDFEngine()
    geom_resolver = LDrawGeometryResolver()

    # Step t-1 has one 2x4 brick (id 3001) at the origin
    part_meta = geom_resolver.get_part_meta("3001")
    assert len(part_meta.get("studs", [])) == 8

    # Identity transform
    node = {
        "part_index": 0,
        "part_id": "3001",
        "absolute_transform": np.eye(4, dtype=np.float32).flatten().tolist(),
    }

    # Affine projection matrix mapping 3D to 2D
    # Scale = 2.0, Offset = (100, 100)
    P = np.array([
        [2.0, 0.0, 0.0, 100.0],
        [0.0, 0.0, 2.0, 100.0],
    ], dtype=np.float32)

    # In step t, a 2x2 brick is placed on the first 4 studs (negative X side).
    # Those 4 studs disappear from the PDF!
    # Only the remaining 4 studs (positive X side) still have visible 2D ellipses:
    studs = part_meta["studs"]
    visible_ellipses = []
    for s in studs:
        s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
        if s_pos[0] > 0.0:  # Only positive X studs remain visible
            s_homo = np.array([s_pos[0], s_pos[1], s_pos[2], 1.0], dtype=np.float32)
            u, v = float(P[0] @ s_homo), float(P[1] @ s_homo)
            visible_ellipses.append(
                StudEllipse2D(
                    center=(u, v),
                    bbox=(u - 4, v - 2, u + 4, v + 2),
                    width=8.0,
                    height=4.0,
                    aspect_ratio=2.0,
                )
            )

    # ROI covering the newly placed part (the negative X area)
    # Studs with s_pos[0] < 0 have u ~ 100 + 2*(-10) = 80, or 2*(-30) = 40
    roi_new = (30.0, 80.0, 90.0, 120.0)

    occluded = engine.detect_occluded_studs(
        existing_nodes=[node],
        geom_resolver=geom_resolver,
        projection_matrix=P,
        detected_ellipses_step_t=visible_ellipses,
        roi_new=roi_new,
        stud_search_radius_pt=5.0,
    )

    # Exactly 4 studs must be detected as occluded!
    assert len(occluded) == 4
    for item in occluded:
        assert item["parent_part_index"] == 0
        # Pos X should be negative for all occluded studs
        assert item["stud_world_pos"][0] < 0.0


def test_score_hypotheses_with_occluded_studs():
    engine = VectorPDFEngine()

    occluded_studs = [
        {"parent_part_index": 0, "stud_world_pos": [-10.0, 0.0, -10.0]},
        {"parent_part_index": 0, "stud_world_pos": [-10.0, 0.0, 10.0]},
    ]

    # Hypothesis 1 matches both occluded studs
    hyp_correct = ConnectionHypothesis(
        child_part_id="3003",
        child_element_id="123",
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        absolute_transform=np.eye(4, dtype=np.float32),
        parent_part_index=0,
        matched_studs=[(-10.0, 0.0, -10.0), (-10.0, 0.0, 10.0)],
        total_score=1.0,
    )

    # Hypothesis 2 connects elsewhere (wrong studs)
    hyp_wrong = ConnectionHypothesis(
        child_part_id="3003",
        child_element_id="123",
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.array([20.0, 0.0, 0.0], dtype=np.float32),
        absolute_transform=np.eye(4, dtype=np.float32),
        parent_part_index=0,
        matched_studs=[(30.0, 0.0, -10.0), (30.0, 0.0, 10.0)],
        total_score=2.0,  # Starts with higher prior score
    )

    scored = engine.score_hypotheses_with_occluded_studs(
        [hyp_wrong, hyp_correct],
        occluded_studs,
        stud_distance_threshold_ldu=4.0,
    )

    # The correct hypothesis should now have occlusion score 1.0 and be sorted first!
    assert scored[0] == hyp_correct
    assert hyp_correct.stud_occlusion_score == 1.0
    assert hyp_correct.total_score >= 6.0


def test_check_occluded_studs_bidirectional():
    engine = VectorPDFEngine()
    geom_resolver = LDrawGeometryResolver()

    detected_occluded = [
        {"parent_part_index": 0, "stud_world_pos": [-10.0, 0.0, -10.0]},
        {"parent_part_index": 0, "stud_world_pos": [-10.0, 0.0, 10.0]},
    ]

    # Perfect match: covers exactly the 2 occluded studs
    hyp_perfect = ConnectionHypothesis(
        child_part_id="3003",
        child_element_id="1",
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        absolute_transform=np.eye(4, dtype=np.float32),
        parent_part_index=0,
        matched_studs=[(-10.0, 0.0, -10.0), (-10.0, 0.0, 10.0)],
    )

    pass_perf, f1_perf = engine.check_occluded_studs_bidirectional(
        hyp_perfect, [], geom_resolver, detected_occluded
    )
    assert pass_perf is True
    assert f1_perf >= 0.99

    # Partial / False positive match: covers 1 occluded stud and 1 visible stud
    hyp_bad = ConnectionHypothesis(
        child_part_id="3003",
        child_element_id="1",
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        absolute_transform=np.eye(4, dtype=np.float32),
        parent_part_index=0,
        matched_studs=[(-10.0, 0.0, -10.0), (30.0, 0.0, 10.0)],
    )

    pass_bad, f1_bad = engine.check_occluded_studs_bidirectional(
        hyp_bad, [], geom_resolver, detected_occluded
    )
    assert pass_bad is False
    assert f1_bad < 0.6


def test_check_new_studs_exposed():
    engine = VectorPDFEngine()
    geom_resolver = LDrawGeometryResolver()

    # 2x2 brick 3003 with 4 top studs
    P = np.array([
        [2.0, 0.0, 0.0, 100.0],
        [0.0, 0.0, 2.0, 100.0],
    ], dtype=np.float32)

    hyp = ConnectionHypothesis(
        child_part_id="3003",
        child_element_id="1",
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.array([0.0, -24.0, 0.0], dtype=np.float32),
        absolute_transform=np.eye(4, dtype=np.float32),
        parent_part_index=0,
    )

    # Calculate where the 4 top studs will land in 2D
    meta = geom_resolver.get_part_meta("3003")
    studs = meta["studs"]
    detected_ellipses = []
    for s in studs:
        s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
        s_w = s_pos + np.array([0.0, -24.0, 0.0], dtype=np.float32)
        h = np.array([s_w[0], s_w[1], s_w[2], 1.0], dtype=np.float32)
        u, v = float(P[0] @ h), float(P[1] @ h)
        detected_ellipses.append(
            StudEllipse2D(
                center=(u + 0.1, v - 0.1),  # tiny jitter < 0.2 pt
                bbox=(u-4, v-2, u+4, v+2),
                width=8.0,
                height=4.0,
                aspect_ratio=2.0,
            )
        )

    passed, ratio, mean_res = engine.check_new_studs_exposed(
        hyp, P, detected_ellipses, geom_resolver, max_dist_pt=1.0
    )
    assert passed is True
    assert ratio == 1.0
    assert mean_res < 0.5


def test_check_vector_edge_alignment():
    engine = VectorPDFEngine()
    geom_resolver = LDrawGeometryResolver()

    P = np.array([
        [1.5, 0.0, 0.0, 50.0],
        [0.0, 0.0, 1.5, 50.0],
    ], dtype=np.float32)

    hyp = ConnectionHypothesis(
        child_part_id="3003",
        child_element_id="1",
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        absolute_transform=np.eye(4, dtype=np.float32),
        parent_part_index=0,
    )

    edges_3d = geom_resolver.get_part_edges_type2("3003")
    if not edges_3d:
        edges_3d = [
            (np.array([-10.0, 0.0, -10.0], dtype=np.float32), np.array([10.0, 0.0, -10.0], dtype=np.float32)),
            (np.array([10.0, 0.0, -10.0], dtype=np.float32), np.array([10.0, 0.0, 10.0], dtype=np.float32)),
        ]

    # Generate matching PDF stroke segments
    stroke_segments = []
    for p1, p2 in edges_3d:
        h1 = np.array([p1[0], p1[1], p1[2], 1.0], dtype=np.float32)
        h2 = np.array([p2[0], p2[1], p2[2], 1.0], dtype=np.float32)
        u1, v1 = float(P[0] @ h1), float(P[1] @ h1)
        u2, v2 = float(P[0] @ h2), float(P[1] @ h2)
        stroke_segments.append(((u1, v1), (u2, v2)))

    passed, ratio, mean_res = engine.check_vector_edge_alignment(
        hyp, P, stroke_segments, geom_resolver, wireframe_edges_3d=edges_3d, max_dist_pt=1.5
    )
    assert passed is True
    assert ratio >= 0.70
    assert mean_res < 0.1


def test_validate_infallible_vector_fit_composite():
    engine = VectorPDFEngine()
    geom_resolver = LDrawGeometryResolver()

    # Realistic axonometric projection (az=45, el=30)
    az_rad = np.radians(45.0)
    el_rad = np.radians(30.0)
    cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)
    cos_e, sin_e = np.cos(el_rad), np.sin(el_rad)

    R_cam = np.array([
        [-cos_a, 0.0, sin_a],
        [-sin_a * sin_e, -cos_e, -cos_a * sin_e]
    ], dtype=np.float32)

    scale = 2.0
    P = np.zeros((2, 4), dtype=np.float32)
    P[:2, :3] = R_cam * scale
    P[:2, 3] = np.array([200.0, 200.0], dtype=np.float32)

    parent_node = {
        "part_index": 0,
        "part_id": "3001",
        "absolute_transform": np.eye(4, dtype=np.float32).flatten().tolist(),
    }

    # Cand attaches to 4 studs of 3001 (Y=-24)
    matched_s = [(-10.0, 0.0, -10.0), (-10.0, 0.0, 10.0), (10.0, 0.0, -10.0), (10.0, 0.0, 10.0)]
    hyp = ConnectionHypothesis(
        child_part_id="3003",
        child_element_id="1",
        rotation_matrix=np.eye(3, dtype=np.float32),
        translation=np.array([0.0, -24.0, 0.0], dtype=np.float32),
        absolute_transform=np.eye(4, dtype=np.float32),
        parent_part_index=0,
        matched_studs=matched_s,
    )

    # 1. Detected ellipses in PDF:
    # Top studs of 3003
    meta_child = geom_resolver.get_part_meta("3003")
    ellipses = []
    roi_pts = []
    for s in meta_child["studs"]:
        s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
        s_w = s_pos + np.array([0.0, -24.0, 0.0], dtype=np.float32)
        h = np.array([s_w[0], s_w[1], s_w[2], 1.0], dtype=np.float32)
        u, v = float(P[0] @ h), float(P[1] @ h)
        ellipses.append(StudEllipse2D(center=(u, v), bbox=(u-4, v-2, u+4, v+2), width=8, height=4, aspect_ratio=2))
        roi_pts.append((u, v))

    # Also include the remaining exposed studs of 3001 (the other 4 studs)
    meta_parent = geom_resolver.get_part_meta("3001")
    for s in meta_parent["studs"]:
        s_pos = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
        if tuple(np.round(s_pos, 1)) not in {tuple(np.round(ms, 1)) for ms in matched_s}:
            h = np.array([s_pos[0], s_pos[1], s_pos[2], 1.0], dtype=np.float32)
            u, v = float(P[0] @ h), float(P[1] @ h)
            ellipses.append(StudEllipse2D(center=(u, v), bbox=(u-4, v-2, u+4, v+2), width=8, height=4, aspect_ratio=2))

    # 2. Stroke segments matching 3003 wireframe
    edges_3d = geom_resolver.get_part_edges_type2("3003")
    stroke_segs = []
    for p1, p2 in edges_3d:
        p1_w = p1 + np.array([0.0, -24.0, 0.0], dtype=np.float32)
        p2_w = p2 + np.array([0.0, -24.0, 0.0], dtype=np.float32)
        h1 = np.array([p1_w[0], p1_w[1], p1_w[2], 1.0], dtype=np.float32)
        h2 = np.array([p2_w[0], p2_w[1], p2_w[2], 1.0], dtype=np.float32)
        u1, v1 = float(P[0] @ h1), float(P[1] @ h1)
        u2, v2 = float(P[0] @ h2), float(P[1] @ h2)
        stroke_segs.append(((u1, v1), (u2, v2)))
        roi_pts.extend([(u1, v1), (u2, v2)])

    xs = [p[0] for p in roi_pts]
    ys = [p[1] for p in roi_pts]
    roi_new = (min(xs) - 5.0, min(ys) - 5.0, max(xs) + 5.0, max(ys) + 5.0)

    vec_ev = VectorEvidence(
        is_vector_scene=True,
        stud_ellipses=ellipses,
        stroke_segments=stroke_segs,
        roi_new_parts=roi_new,
    )

    is_inf, conf, metrics = engine.validate_infallible_vector_fit(
        hypothesis=hyp,
        existing_nodes=[parent_node],
        projection_matrix=P,
        vector_evidence=vec_ev,
        geom_resolver=geom_resolver,
        wireframe_edges_3d=edges_3d,
    )

    assert is_inf is True
    assert conf == 1.0
    assert hyp.is_infallible_match is True
    assert hyp.total_score >= 100.0


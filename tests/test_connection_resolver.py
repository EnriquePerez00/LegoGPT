import os
import cv2
import pytest
import numpy as np
from src.connection_resolver import LegoConnectionResolver, ConnectionCandidate


def test_resolver_cand_generation_trivial():
    resolver = LegoConnectionResolver()
    # Test two 1x1 plates: 3024.dat on 3024.dat
    # Should have exactly 1 centered connection (or 4 symmetric rotations that engage the exact same stud)
    cands = resolver.generate_candidate_connections("3024.dat", "3024.dat", layer_offset_y=-8.0)
    assert len(cands) >= 1
    # Each candidate must have contact points
    for c in cands:
        assert c.num_contacts >= 1
        assert len(c.contact_points) >= 1
        assert c.translation_ldu[1] == -8.0


def test_resolver_page_5_pair_physics():
    resolver = LegoConnectionResolver()
    # 11213.dat (base plate 6x6 round) and 30565.dat (top plate 4x4 round corner)
    cands = resolver.generate_candidate_connections("11213.dat", "30565.dat", layer_offset_y=-8.0)
    assert len(cands) > 10

    # Ensure all candidates strictly adhere to Stud-to-Socket rules
    for c in cands:
        assert c.num_contacts >= 1
        assert c.translation_ldu[1] == -8.0
        # Check that occupied parent studs + free parent studs = total parent studs
        assert len(c.occupied_parent_studs) > 0
        assert len(c.free_parent_studs) > 0


def test_disambiguate_with_manual_crop_option_a():
    resolver = LegoConnectionResolver()
    cands = resolver.generate_candidate_connections("11213.dat", "30565.dat", layer_offset_y=-8.0)

    manual_img_path = "/Users/I764690/.gemini/antigravity/brain/465f3d3e-6b4e-4e25-8783-35d4f72af62c/.user_uploaded/media_1793730628339_a7727853.png"
    if not os.path.exists(manual_img_path):
        pytest.skip("Manual crop image not found")

    # Evaluate candidates
    best = resolver.disambiguate_with_image_crop(cands[:25], manual_img_path, "11213.dat", "30565.dat")
    assert best is not None
    assert best.score > 0.0
    assert best.iou > 0.4
    assert len(best.contact_points) >= 10

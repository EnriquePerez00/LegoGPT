import numpy as np
import pytest
from src.parser import ParsedPart
from src.graph_validator import LegoGraphValidator

# Identity rotation matrix for non-rotated parts
IDENTITY_ROTATION = np.eye(3, dtype=np.float32)

def create_brick(part_id: str, color: int, x: float, y: float, z: float) -> dict:
    """Helper to create a brick dictionary representation."""
    return {
        "part_id": part_id,
        "color": color,
        "position": [x, y, z],
        "rotation": IDENTITY_ROTATION.tolist()
    }

def test_valid_three_brick_assembly():
    """
    Test a valid assembly of three 2x4 bricks (3001.dat).
    Brick 1: at ground level [0, 0, 0]
    Brick 2: stacked on Brick 1, offset by 1 stud (20 LDU) along Z [0, -24, 20]
    Brick 3: stacked on Brick 2, offset by 1 stud (20 LDU) along Z [0, -48, 40]
    """
    brick1 = create_brick("3001.dat", 1, 0.0, 0.0, 0.0)
    brick2 = create_brick("3001.dat", 2, 0.0, -24.0, 20.0)
    brick3 = create_brick("3001.dat", 3, 0.0, -48.0, 40.0)
    
    # Start with empty structure, first brick should be allowed
    validator = LegoGraphValidator()
    assert validator.can_place_brick(brick1, current_state=[]) is True
    
    # Place brick 1
    state = [brick1]
    
    # Place brick 2
    assert validator.can_place_brick(brick2, current_state=state) is True
    state.append(brick2)
    
    # Place brick 3
    assert validator.can_place_brick(brick3, current_state=state) is True

def test_colliding_brick():
    """
    Test that a brick that physically overlaps with the existing state is rejected.
    """
    brick1 = create_brick("3001.dat", 1, 0.0, 0.0, 0.0)
    # This brick overlaps with brick1 vertically and horizontally
    colliding_brick = create_brick("3001.dat", 2, 10.0, -10.0, 10.0)
    
    validator = LegoGraphValidator(current_state=[brick1])
    assert validator.can_place_brick(colliding_brick) is False

def test_floating_brick():
    """
    Test that a floating brick with no stud/antistud connections is rejected.
    """
    brick1 = create_brick("3001.dat", 1, 0.0, 0.0, 0.0)
    # Placed in the air far away along Z direction
    floating_brick = create_brick("3001.dat", 2, 0.0, -24.0, 100.0)
    
    validator = LegoGraphValidator(current_state=[brick1])
    assert validator.can_place_brick(floating_brick) is False

def test_connectivity_tolerance():
    """
    Test that the 5 LDU tolerance for stud-to-antistud connection is correctly enforced.
    """
    brick1 = create_brick("3001.dat", 1, 0.0, 0.0, 0.0)
    
    # 1. Perfect stacking (0 LDU shift) -> Valid
    perfect_brick = create_brick("3001.dat", 2, 0.0, -24.0, 0.0)
    validator = LegoGraphValidator(current_state=[brick1])
    assert validator.can_place_brick(perfect_brick) is True
    
    # 2. Shifted by 3.0 LDU horizontally (< 5.0 LDU tolerance) -> Valid
    shifted_valid = create_brick("3001.dat", 2, 3.0, -24.0, 0.0)
    assert validator.can_place_brick(shifted_valid) is True
    
    # 3. Shifted by 6.0 LDU horizontally (> 5.0 LDU tolerance) -> Invalid
    shifted_invalid = create_brick("3001.dat", 2, 6.0, -24.0, 0.0)
    assert validator.can_place_brick(shifted_invalid) is False


def test_manual_connection_graph_ldraw_physics():
    from src.inventory_matcher import LegoInventoryMatcher
    from src.manual_connection_graph import ManualConnectionGraphBuilder
    import os

    pdf_path = "data/manuals/75295-1_Millennium_Falcon_Microfighter.pdf"
    if not os.path.exists(pdf_path):
        import pytest
        pytest.skip("PDF not found")

    matcher = LegoInventoryMatcher(pdf_path, temp_dir="scratch/test_graph_templates")
    res_steps = matcher.extract_and_match_all_steps(db_path="data/catalog/models_catalog.db", set_num="75295-1")

    builder = ManualConnectionGraphBuilder(pdf_path)
    graph = builder.build_connection_graph(res_steps["step_pages"])

    assert graph["total_nodes"] > 50
    assert 28 in graph["flip_steps_detected"]
    assert 29 in graph["flip_steps_detected"]

    # Verify Page 4 Minifig subassembly
    minifig_nodes = [n for n in graph["nodes"] if n["page_num"] == 4]
    assert len(minifig_nodes) == 5
    blaster_node = next(n for n in minifig_nodes if "92738" in n["part_id"])
    assert len(blaster_node["connections"]) == 1
    assert blaster_node["connections"][0]["child_anchor"] == "handle_bar"

    # Verify Page 5 base & corner connection
    p5_nodes = [n for n in graph["nodes"] if n["page_num"] == 5]
    assert len(p5_nodes) == 2
    assert p5_nodes[0]["part_id"] == "11213.dat"
    assert p5_nodes[0]["y_bottom_ldu"] == 0
    assert p5_nodes[1]["part_id"] == "30565.dat"
    assert p5_nodes[1]["y_bottom_ldu"] == -8

    # Verify Page 28 flipped state
    p28_nodes = [n for n in graph["nodes"] if n["page_num"] == 28]
    assert len(p28_nodes) == 4
    for n in p28_nodes:
        assert n["view_flipped"] is True
        assert n["part_id"] == "2654.dat"


def test_generalized_part_resolution():
    """Validates that non-native/printed/variant LEGO parts resolve automatically to valid LDraw files."""
    from src.ldraw_geometry import LDrawGeometryResolver

    resolver = LDrawGeometryResolver()
    
    # Printed pieces -> base mold resolution
    assert resolver.resolve_compatible_part_id("3626cpr1368") == "3626c.dat"
    assert resolver.resolve_compatible_part_id("3001pr0001") == "3001.dat"
    assert resolver.resolve_compatible_part_id("3069bpr0002") == "3069b.dat"
    
    # Minifigure assemblies (must resolve to assembled torso with arms 973c01.dat)
    assert resolver.resolve_compatible_part_id("970c05pr0625") == "970c00.dat"
    assert resolver.resolve_compatible_part_id("973c26h02pr2301") == "973c01.dat"
    
    # Standard official parts
    assert resolver.resolve_compatible_part_id("26139") == "26139.dat"
    assert resolver.resolve_compatible_part_id("92738") == "92738.dat"
    assert resolver.resolve_compatible_part_id("3024") == "3024.dat"



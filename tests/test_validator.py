import numpy as np
import pytest
from src.parser import ParsedPart
from src.validator import check_collisions, check_connectivity_and_gravity, validate_rules

# Identity matrix for non-rotated parts
IDENTITY_ROTATION = np.eye(3)

def create_part(part_id: str, color: int, x: float, y: float, z: float, step: int = 0) -> ParsedPart:
    """Helper to create a ParsedPart with identity rotation."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = IDENTITY_ROTATION
    transform[:3, 3] = [x, y, z]
    return ParsedPart(part_id=part_id, color=color, transform=transform, step_id=step)

def test_valid_stacked_bricks():
    """Test two bricks stacked vertically with correct alignment (no collision, connected, anchored)."""
    # Brick 1x1 (3005.dat) at bottom (origin y=0)
    part_bottom = create_part("3005.dat", 14, 0.0, 0.0, 0.0)
    # Brick 1x1 (3005.dat) on top (y=-24, since Y is inverted)
    part_top = create_part("3005.dat", 14, 0.0, -24.0, 0.0)
    
    parts = [part_bottom, part_top]
    
    # 1. No collisions should be found
    collisions = check_collisions(parts)
    assert len(collisions) == 0, f"Expected no collisions, found: {collisions}"
    
    # 2. Both should be connected to the base (gravity/anchoring check)
    is_stable = check_connectivity_and_gravity(parts)
    assert is_stable is True, "Expected the stack to be stable and connected to the base"
    
    # 3. Both parts should respect the rules (valid IDs and colors)
    valid_rules = validate_rules(parts)
    assert valid_rules is True, "Expected parts to be valid under system rules"

def test_collision_detection():
    """Test that two overlapping bricks are flagged as a collision."""
    # Two identical 2x2 bricks (3003.dat) placed at the exact same location
    part_a = create_part("3003.dat", 4, 0.0, 0.0, 0.0)
    part_b = create_part("3003.dat", 4, 10.0, 0.0, 10.0)  # Overlapping by 30 LDU (dimensions are 40x24x40)
    
    parts = [part_a, part_b]
    
    collisions = check_collisions(parts)
    assert len(collisions) > 0, "Expected a collision to be detected between overlapping parts"
    # The collision pair should contain indices (0, 1) or (1, 0)
    assert (0, 1) in collisions or (1, 0) in collisions

def test_floating_pieces():
    """Test that a floating brick with no physical connection is flagged as unstable."""
    # Brick 1x1 (3005.dat) at base
    part_base = create_part("3005.dat", 1, 0.0, 0.0, 0.0)
    # Floating Brick 1x1 (3005.dat) far away and high up
    part_floating = create_part("3005.dat", 1, 100.0, -100.0, 100.0)
    
    parts = [part_base, part_floating]
    
    is_stable = check_connectivity_and_gravity(parts)
    assert is_stable is False, "Expected floating piece to cause instability (unanchored graph)"

def test_invalid_rules():
    """Test that invalid part IDs or colors are caught by rule checking."""
    # Invalid part ID
    part_invalid_id = create_part("9999.dat", 1, 0.0, 0.0, 0.0)
    assert validate_rules([part_invalid_id]) is False, "Expected verification to fail for invalid part ID"
    
    # Invalid color code (negative or out of allowed range)
    part_invalid_color = create_part("3005.dat", 999, 0.0, 0.0, 0.0)
    assert validate_rules([part_invalid_color]) is False, "Expected verification to fail for invalid color code"

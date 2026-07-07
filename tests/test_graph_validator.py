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

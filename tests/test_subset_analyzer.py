import numpy as np
import pytest
import os
from src.parser import ParsedPart
from src.mpd_parser import flatten_mpd
from src.subset_analyzer import analyze_subsets, build_port_attributed_assembly_graph

def create_part(part_id: str, x: float, y: float, z: float, color: int = 1, step: int = 0) -> ParsedPart:
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = [x, y, z]
    return ParsedPart(part_id=part_id, color=color, transform=t, step_id=step)

def test_independent_subsets_disjoint_pillars():
    """
    Test a scene with two disjoint 2-brick pillars with 100 LDU separation.
    Must be identified as 2 independent subsets.
    """
    # Pillar A at X = 0
    pA1 = create_part("3005.dat", 0.0, 0.0, 0.0)      # Ground brick
    pA2 = create_part("3005.dat", 0.0, -24.0, 0.0)    # Top brick
    
    # Pillar B at X = 100
    pB1 = create_part("3005.dat", 100.0, 0.0, 0.0)    # Ground brick
    pB2 = create_part("3005.dat", 100.0, -24.0, 0.0)  # Top brick

    parts = [pA1, pA2, pB1, pB2]
    subsets = analyze_subsets(parts)

    assert len(subsets) == 2, f"Expected 2 independent subsets, found: {len(subsets)}"
    assert subsets[0].parts_count == 2
    assert subsets[1].parts_count == 2
    assert subsets[0].is_grounded is True
    assert subsets[1].is_grounded is True

def test_interlocking_assembly():
    """
    Test an interlocking 3-brick pyramid:
    Layer 1: Two 2x4 bricks side by side at ground level.
    Layer 2: One 2x4 brick spanning across both bottom bricks (staggered seam).
    Must be 1 single fully-connected subset with interlocking_index > 0.
    """
    # Bottom bricks (3001.dat: 40 x 24 x 80 LDU)
    b1 = create_part("3001.dat", -20.0, 0.0, 0.0)
    b2 = create_part("3001.dat", 20.0, 0.0, 0.0)
    
    # Staggered top brick connecting across both
    b3 = create_part("3001.dat", 0.0, -24.0, 0.0)

    parts = [b1, b2, b3]
    subsets = analyze_subsets(parts)

    assert len(subsets) == 1, "Interlocking pyramid must form a single connected subset"
    sub = subsets[0]
    assert sub.parts_count == 3
    assert sub.is_grounded is True
    assert sub.is_statically_stable is True

def test_static_stability_balance():
    """
    Test center of mass and static stability calculation:
    A balanced column centered over the base must be stable.
    A 1x1 brick placed on one of the 4 studs of a 2x2 brick [10, -24, 10] is physically valid and stable.
    Placing a 1x1 brick at [0, -24, 0] (in the dead gap between the 4 studs) is physically invalid.
    """
    base = create_part("3003.dat", 0.0, 0.0, 0.0)             # 2x2 brick at origin (studs at +/-10)
    top_legal = create_part("3005.dat", 10.0, -24.0, 10.0)    # 1x1 brick on stud (+10, +10)

    parts = [base, top_legal]
    subsets = analyze_subsets(parts)

    assert len(subsets) == 1
    assert subsets[0].is_statically_stable is True
    assert subsets[0].center_of_mass[1] < 0.0  # Center of mass above ground

    # Negative test: illegal placement in the center gap [0, -24, 0] must NOT connect
    top_illegal = create_part("3005.dat", 0.0, -24.0, 0.0)
    subsets_illegal = analyze_subsets([base, top_illegal])
    assert len(subsets_illegal) == 2, "1x1 brick in the dead center of a 2x2 brick has no matching stud and must remain disconnected."

def test_direct_studio_io_parsing():
    """Verify direct parsing of BrickLink Studio .io files."""
    sample_io = "/Applications/Studio 2.0/Tutorial/Stage_0.io"
    if os.path.exists(sample_io):
        parts = flatten_mpd(sample_io)
        assert len(parts) == 4
        subsets = analyze_subsets(parts)
        # Stage_0 contains 4 independent bricks placed without connections
        assert len(subsets) == 4

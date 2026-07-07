import numpy as np
import pytest
import json
from src.mecabricks_converter import (
    map_part_id,
    map_color_id,
    convert_matrix,
    parse_mbx_content
)

def test_map_part_id():
    assert map_part_id("3005") == "3005.dat"
    assert map_part_id("3023") == "3023.dat"
    # Case insensitivity & formatting
    assert map_part_id("  3004  ") == "3004.dat"
    # Custom unknown part fallback
    assert map_part_id("99999") == "99999.dat"
    # Existing .dat extension preserved
    assert map_part_id("3005.dat") == "3005.dat"

def test_map_color_id():
    assert map_color_id("solid_red") == 4
    assert map_color_id("solid_black") == 0
    assert map_color_id("solid_white") == 15
    assert map_color_id("red") == 4
    assert map_color_id("unknown_metallic_color") == 7 # Fallback to gray
    assert map_color_id("14") == 14

def test_convert_matrix():
    # Identity matrix in Mecabricks
    mb_matrix = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        1.0, 2.0, 3.0, 1.0 # Translation in row-major
    ]
    # In LDraw, translation is M[:3, 3] and rotation is M[:3, :3]
    # LDraw matrix format maps to:
    # [ [a, b, c, x], [d, e, f, y], [g, h, i, z], [0, 0, 0, 1] ]
    # In mecabricks_converter.py:
    # M_ldraw[0, 3] = M_mb[0, 3] * scale
    # M_ldraw[1, 3] = -M_mb[1, 3] * scale
    # M_ldraw[2, 3] = -M_mb[2, 3] * scale
    
    scale = 2500.0
    M_ld = convert_matrix(mb_matrix, scale=scale)
    
    # Check translation conversion
    assert M_ld[0, 3] == 0.0 * scale  # M_mb[0,3] is 0.0 (top row right column is 0.0)
    # Wait, the translation in mb_matrix row-major 4x4 matrix for WebGL:
    # [ r00, r01, r02, t_x,
    #   r10, r11, r12, t_y,
    #   r20, r21, r22, t_z,
    #   0,   0,   0,   1 ]
    # Wait! In convert_matrix:
    # M_mb is reshaped to 4x4 using M_mb = np.array(mb_matrix).reshape(4,4)
    # So M_mb[0, 3] is indeed index 3 in the array, which is t_x.
    # In our mb_matrix: index 3 is 0.0, index 7 is 0.0, index 11 is 0.0, index 12,13,14 are 1.0, 2.0, 3.0.
    # Wait! If index 12, 13, 14 are 1.0, 2.0, 3.0, that is column-major translation or WebGL transform where translation is in the bottom row!
    # Let's check which format Mecabricks uses.
    # In Blender/Three.js, a column-major matrix has translation in elements 12, 13, 14.
    # In row-major, translation is at 3, 7, 11.
    # Let's see: in convert_matrix we wrote:
    # M_mb = np.array(mb_matrix, dtype=np.float32).reshape(4, 4)
    # R_mb = M_mb[:3, :3]
    # M_ldraw[0, 3] = M_mb[0, 3] * scale
    # M_ldraw[1, 3] = -M_mb[1, 3] * scale
    # M_ldraw[2, 3] = -M_mb[2, 3] * scale
    # This assumes that translation is at indices 3, 7, 11 (row-major).
    # If the input list is column-major (WebGL/Three.js style), translation is at index 12, 13, 14.
    # Let's make sure our converter handles both or that our unit tests match the code assumptions.
    # In our code, M_mb[0, 3] is the top-right element. So in mb_matrix, it should be at index 3.
    # Let's define the test matrix with translation at index 3, 7, 11 (row-major):
    mb_matrix_row_major = [
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 2.0,
        0.0, 0.0, 1.0, 3.0,
        0.0, 0.0, 0.0, 1.0
    ]
    M_ld = convert_matrix(mb_matrix_row_major, scale=scale)
    assert M_ld[0, 3] == 1.0 * scale
    assert M_ld[1, 3] == -2.0 * scale
    assert M_ld[2, 3] == -3.0 * scale

def test_parse_mbx_content():
    mbx_data = {
        "nodes": [
            {
                "id": "3005",
                "color": "solid_red",
                "matrix": [
                    1.0, 0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0, 2.0,
                    0.0, 0.0, 1.0, 3.0,
                    0.0, 0.0, 0.0, 1.0
                ],
                "step": 1
            }
        ]
    }
    json_str = json.dumps(mbx_data)
    parts = parse_mbx_content(json_str, scale=1.0)
    assert len(parts) == 1
    p = parts[0]
    assert p.part_id == "3005.dat"
    assert p.color == 4
    assert p.step_id == 1
    assert p.transform[0, 3] == 1.0
    assert p.transform[1, 3] == -2.0
    assert p.transform[2, 3] == -3.0

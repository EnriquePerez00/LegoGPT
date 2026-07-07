import os
import tempfile
import numpy as np
import pytest
import torch
from src.parser import parse_ldraw_file, build_pyg_graph

SAMPLE_LDRAW_CONTENT = """0 LegoGPT Sample Model
0 Name: sample.ldr
0 Author: Antigravity
1 14 0 0 0 1 0 0 0 1 0 0 0 1 3005.dat
0 STEP
1 14 0 -24 0 1 0 0 0 1 0 0 0 1 3005.dat
1 15 20 -24 0 1 0 0 0 1 0 0 0 1 3024.dat
0 STEP
1 4 0 -32 0 1 0 0 0 1 0 0 0 1 3003.dat
"""

def test_ldraw_parser():
    """Test parsing of LDraw content and correct step detection."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ldr', delete=False) as tmp:
        tmp.write(SAMPLE_LDRAW_CONTENT)
        tmp_path = tmp.name
        
    try:
        parts = parse_ldraw_file(tmp_path)
        
        # Check part count (should be 4 parts)
        assert len(parts) == 4
        
        # Check parts details
        # Part 0: 3005.dat, color 14, step 0
        assert parts[0].part_id == "3005.dat"
        assert parts[0].color == 14
        assert parts[0].step_id == 0
        assert np.allclose(parts[0].transform[:3, 3], [0.0, 0.0, 0.0])
        
        # Part 1: 3005.dat, color 14, step 1
        assert parts[1].part_id == "3005.dat"
        assert parts[1].color == 14
        assert parts[1].step_id == 1
        assert np.allclose(parts[1].transform[:3, 3], [0.0, -24.0, 0.0])
        
        # Part 2: 3024.dat, color 15, step 1
        assert parts[2].part_id == "3024.dat"
        assert parts[2].color == 15
        assert parts[2].step_id == 1
        assert np.allclose(parts[2].transform[:3, 3], [20.0, -24.0, 0.0])
        
        # Part 3: 3003.dat, color 4, step 2
        assert parts[3].part_id == "3003.dat"
        assert parts[3].color == 4
        assert parts[3].step_id == 2
        assert np.allclose(parts[3].transform[:3, 3], [0.0, -32.0, 0.0])
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_build_pyg_graph():
    """Test that a list of parsed parts is correctly converted to a PyTorch Geometric Graph."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ldr', delete=False) as tmp:
        tmp.write(SAMPLE_LDRAW_CONTENT)
        tmp_path = tmp.name
        
    try:
        parts = parse_ldraw_file(tmp_path)
        data = build_pyg_graph(parts)
        
        # Check data object properties
        assert data.num_nodes == 4
        
        # Features x should be of shape [4, 60]
        # (10 part classes + 16 color classes + 3 translation + 9 rotation = 38)
        assert data.x.shape == (4, 60)

        
        # Check edge_index and edge_attr
        # At least some connections should form
        assert data.edge_index.shape[0] == 2
        
        if data.edge_index.shape[1] > 0:
            assert data.edge_attr.shape[1] == 12  # 3 rel trans + 9 rel rot
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

import os
import tempfile
import numpy as np
import pytest
import torch
from src.mpd_parser import parse_mpd_to_submodels, flatten_mpd
from src.dataset_generator import generate_set_dataset

MOCK_MPD_CONTENT = """0 FILE main.ldr
0 Name: main.ldr
0 Author: Test
1 16 0 0 0 1 0 0 0 1 0 0 0 1 submodel.ldr
0 STEP
1 14 0 -24 0 1 0 0 0 1 0 0 0 1 3005.dat
0 FILE submodel.ldr
0 Name: submodel.ldr
1 15 10 20 30 1 0 0 0 1 0 0 0 1 3001.dat
"""

def test_parse_mpd_to_submodels():
    """Verify that an MPD file is split into correct submodel blocks."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mpd', delete=False) as tmp:
        tmp.write(MOCK_MPD_CONTENT)
        tmp_path = tmp.name
        
    try:
        submodels = parse_mpd_to_submodels(tmp_path)
        assert len(submodels) == 2
        assert "main.ldr" in submodels
        assert "submodel.ldr" in submodels
        
        # Check submodel contents
        assert any("3005.dat" in line for line in submodels["main.ldr"])
        assert any("3001.dat" in line for line in submodels["submodel.ldr"])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_flatten_mpd():
    """Verify that hierarchical submodels are correctly resolved to physical parts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mpd', delete=False) as tmp:
        tmp.write(MOCK_MPD_CONTENT)
        tmp_path = tmp.name
        
    try:
        parts = flatten_mpd(tmp_path)
        # Should return exactly 2 physical parts (leaves): 3001.dat and 3005.dat
        assert len(parts) == 2
        
        # First physical part (from submodel.ldr, resolved in main)
        assert parts[0].part_id == "3001.dat"
        assert parts[0].color == 15
        assert np.allclose(parts[0].transform[:3, 3], [10.0, 20.0, 30.0])
        assert parts[0].step_id == 0
        
        # Second physical part (direct in main, step 1)
        assert parts[1].part_id == "3005.dat"
        assert parts[1].color == 14
        assert np.allclose(parts[1].transform[:3, 3], [0.0, -24.0, 0.0])
        assert parts[1].step_id == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_generate_set_dataset():
    """Verify that dataset generator creates output files for sets < 100 parts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mpd', delete=False) as tmp:
        tmp.write(MOCK_MPD_CONTENT)
        tmp_path = tmp.name
        
    # We rename the temp file to look like a set number for filename parsing
    set_tmp_path = os.path.join(os.path.dirname(tmp_path), "9999-1_testset.mpd")
    with open(set_tmp_path, "w") as f:
        f.write(MOCK_MPD_CONTENT)
        
    output_dir = tempfile.mkdtemp()
    
    try:
        success = generate_set_dataset(set_tmp_path, output_dir)
        assert success is True
        
        # Check files created
        assembly_file = os.path.join(output_dir, "9999-1_assembly.json")
        graph_file = os.path.join(output_dir, "9999-1_graph.pt")
        
        assert os.path.exists(assembly_file)
        assert os.path.exists(graph_file)
        
        # Read assembly JSON
        with open(assembly_file, "r") as f:
            data = json.load(f)
            assert data["set_number"] == "9999-1"
            assert data["num_parts"] == 2
            assert len(data["parts"]) == 2

            
        # Read PyG graph
        graph = torch.load(graph_file, weights_only=False)
        assert graph.num_nodes == 2
        assert graph.x.shape == (2, 60)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(set_tmp_path):
            os.remove(set_tmp_path)
        if os.path.exists(assembly_file):
            os.remove(assembly_file)
        if os.path.exists(graph_file):
            os.remove(graph_file)
        if os.path.exists(output_dir):
            os.rmdir(output_dir)
            
import json

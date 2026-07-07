import os
import pytest
from generative.llm_pipeline.standardizer import standardize_and_order_ldr
from src.parser import parse_ldraw_file

def test_standardizer_flattening():
    # Create a mock MPD file containing a submodel
    mock_mpd = """
    0 FILE main_car.ldr
    1 14 0 0 0 1 0 0 0 1 0 0 0 1 wheel_sub.ldr
    1 14 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat
    0 FILE wheel_sub.ldr
    1 4 10 0 0 1 0 0 0 1 0 0 0 1 3005.dat
    1 4 -10 0 0 1 0 0 0 1 0 0 0 1 3005.dat
    """
    
    os.makedirs("scratch", exist_ok=True)
    temp_mpd_path = "scratch/temp_mock.mpd"
    temp_out_path = "scratch/temp_flat.ldr"
    
    with open(temp_mpd_path, "w") as f:
        f.write(mock_mpd.strip())
        
    try:
        # Run standardizer
        ok = standardize_and_order_ldr(temp_mpd_path, temp_out_path)
        assert ok
        
        # Verify output LDraw file exists
        assert os.path.exists(temp_out_path)
        
        # Parse output to verify flattening
        parts = parse_ldraw_file(temp_out_path)
        # Main has: 1 submodel call (resolves to 2 parts) + 1 direct part = 3 parts total
        assert len(parts) == 3
        
        # Check coordinates of flattened wheel_sub parts
        # wheel_sub has pos [10, 0, 0] and [-10, 0, 0], and Y-coordinate of parent is 0
        x_coords = [p.transform[0, 3] for p in parts if p.part_id == "3005.dat"]
        assert 10.0 in x_coords
        assert -10.0 in x_coords
        
        # Check step formatting (0 STEP should be present)
        with open(temp_out_path, "r") as f:
            content = f.read()
            assert "0 STEP" in content
            
    finally:
        # Clean up files
        if os.path.exists(temp_mpd_path):
            os.remove(temp_mpd_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

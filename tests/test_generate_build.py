import os
import subprocess
import pytest
from src.parser import parse_ldraw_file

def test_generate_build_end_to_end():
    output_dir = "scratch/test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run generate_build.py CLI
    cmd = [
        "./legogpt_env/bin/python",
        "generate_build.py",
        "--prompt", "Silla roja",
        "--output_dir", output_dir
    ]
    
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(res.stdout)
        
        # Verify LDraw file creation
        ldr_path = os.path.join(output_dir, "silla_roja.ldr")
        assert os.path.exists(ldr_path)
        
        # 2. Check that the LDraw file is parseable and has correct layout
        parts = parse_ldraw_file(ldr_path)
        assert len(parts) == 6
        for p in parts:
            assert p.color == 4  # Red LDraw code
            
        # Verify that 0 STEP is present
        with open(ldr_path, "r") as f:
            content = f.read()
            assert "0 STEP" in content
            
        # 3. Verify that the PNG render is generated
        png_path = os.path.join(output_dir, "silla_roja.png")
        assert os.path.exists(png_path)
        
    finally:
        # Clean up files
        for name in ["silla_roja.ldr", "silla_roja.png"]:
            p = os.path.join(output_dir, name)
            if os.path.exists(p):
                os.remove(p)

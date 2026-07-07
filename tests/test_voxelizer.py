import os
import trimesh
import pytest
from generative.voxelizer import voxelize_mesh_to_lego
from src.parser import parse_ldraw_file
from src.writer import write_ldraw_file

def test_voxelizer_with_simple_mesh():
    # 1. Create a simple box mesh (100 x 50 x 100 LDU equivalent, scaled)
    # 1 stud = 20 LDU, H = 24 LDU.
    # A box of size 60 x 48 x 60 mm (where 1mm = 2.5 LDU) -> 150 x 120 x 150 LDU.
    box = trimesh.creation.box(extents=(150.0, 120.0, 150.0))
    
    # Save the mesh to a temporary OBJ file in the scratch/ folder
    os.makedirs("scratch", exist_ok=True)
    temp_mesh_path = "scratch/temp_test_mesh.obj"
    box.export(temp_mesh_path)
    
    try:
        # 2. Voxelize the mesh
        parts = voxelize_mesh_to_lego(temp_mesh_path, stud_pitch=20.0, brick_height=24.0)
        
        # Verify we got bricks
        assert len(parts) > 0
        
        # Check that we only generated valid part IDs
        valid_ids = {"3001.dat", "3004.dat", "3005.dat"}
        for p in parts:
            assert p.part_id in valid_ids
            assert p.transform.shape == (4, 4)
            
        # 3. Export to a temporary LDraw file
        temp_ldr_path = "scratch/temp_voxelized.ldr"
        write_ldraw_file(parts, temp_ldr_path)
        
        # 4. Verify that the LDraw file is parseable by the existing parser
        parsed_parts = parse_ldraw_file(temp_ldr_path)
        assert len(parsed_parts) == len(parts)
        for p_orig, p_parsed in zip(parts, parsed_parts):
            assert p_orig.part_id == p_parsed.part_id
            
    finally:
        # Clean up temporary files
        if os.path.exists(temp_mesh_path):
            os.remove(temp_mesh_path)
        if os.path.exists("scratch/temp_voxelized.ldr"):
            os.remove("scratch/temp_voxelized.ldr")

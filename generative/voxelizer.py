import os
import numpy as np
import trimesh
from src.parser import ALLOWED_PARTS, ALLOWED_COLORS
from src.writer import write_ldraw_file, ParsedPart

def voxelize_mesh_to_lego(mesh_path: str, stud_pitch: float = 20.0, brick_height: float = 24.0) -> list[ParsedPart]:
    """
    Loads a standard 3D mesh and voxelizes it, merging adjacent voxels
    into standard LEGO bricks (2x4, 1x2, 1x1) with interlocking offsets.
    """
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
        
    # Get bounding box
    bounds = mesh.bounds
    min_bounds, max_bounds = bounds[0], bounds[1]
    
    # Calculate grid sizes
    dx = max_bounds[0] - min_bounds[0]
    dy = max_bounds[1] - min_bounds[1]
    dz = max_bounds[2] - min_bounds[2]
    
    nx = max(1, int(np.ceil(dx / stud_pitch)))
    ny = max(1, int(np.ceil(dy / brick_height)))
    nz = max(1, int(np.ceil(dz / stud_pitch)))
    
    # Create grid centers
    grid_x = min_bounds[0] + np.arange(nx) * stud_pitch + stud_pitch / 2.0
    grid_y = min_bounds[1] + np.arange(ny) * brick_height + brick_height / 2.0
    grid_z = min_bounds[2] + np.arange(nz) * stud_pitch + stud_pitch / 2.0
    
    # Check which grid cells are inside the mesh
    points = []
    indices = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points.append([grid_x[i], grid_y[j], grid_z[k]])
                indices.append((i, j, k))
                
    points = np.array(points)
    inside = mesh.contains(points)
    
    # Map active voxels
    active_voxels = set()
    for idx, is_inside in enumerate(inside):
        if is_inside:
            active_voxels.add(indices[idx])
            
    parts = []
    used = set()
    color_idx = 14  # Default yellow color code
    step_id = 1
    
    # Process bottom-up (Y increases downwards in LDraw, so we process layers in order)
    # To simulate interlocking brick offsets, we alternate grid matching by layer
    for j in range(ny):
        layer_voxels = [v for v in active_voxels if v[1] == j]
        if not layer_voxels:
            continue
            
        # Get coordinates for the current layer
        layer_set = {(v[0], v[2]) for v in layer_voxels}
        layer_used = set()
        
        # 1. Try to merge into 2x4 bricks (dim in grid: 2x4 or 4x2)
        # We alternate the scan search to create an interlocking pattern
        offset = j % 2
        for x in range(offset, nx - 1, 2):
            for z in range(offset, nz - 3, 4):
                # Check 2x4 orientation
                coords_2x4 = [(x + dx, z + dz) for dx in range(2) for dz in range(4)]
                if all(c in layer_set and c not in layer_used for c in coords_2x4):
                    for c in coords_2x4:
                        layer_used.add(c)
                        used.add((c[0], j, c[1]))
                    # Center of the 2x4 brick
                    cx = min_bounds[0] + (x + 0.5) * stud_pitch + stud_pitch
                    cy = min_bounds[1] + j * brick_height + brick_height  # LDraw top surface is -H, bottom is 0
                    cz = min_bounds[2] + (z + 1.5) * stud_pitch + stud_pitch
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, 3] = [cx, -cy, cz]
                    parts.append(ParsedPart(part_id="3001.dat", color=color_idx, transform=transform, step_id=step_id))
                    
        # Try 4x2 bricks (rotated 2x4)
        for x in range(offset, nx - 3, 4):
            for z in range(offset, nz - 1, 2):
                coords_4x2 = [(x + dx, z + dz) for dx in range(4) for dz in range(2)]
                if all(c in layer_set and c not in layer_used for c in coords_4x2):
                    for c in coords_4x2:
                        layer_used.add(c)
                        used.add((c[0], j, c[1]))
                    cx = min_bounds[0] + (x + 1.5) * stud_pitch + stud_pitch
                    cy = min_bounds[1] + j * brick_height + brick_height
                    cz = min_bounds[2] + (z + 0.5) * stud_pitch + stud_pitch
                    transform = np.eye(4, dtype=np.float32)
                    # Rotate 90 degrees around Y-axis
                    transform[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
                    transform[:3, 3] = [cx, -cy, cz]
                    parts.append(ParsedPart(part_id="3001.dat", color=color_idx, transform=transform, step_id=step_id))

        # 2. Try to merge into 1x2 bricks (dim: 1x2 or 2x1)
        for x in range(nx):
            for z in range(nz - 1):
                coords_1x2 = [(x, z), (x, z + 1)]
                if all(c in layer_set and c not in layer_used for c in coords_1x2):
                    for c in coords_1x2:
                        layer_used.add(c)
                        used.add((c[0], j, c[1]))
                    cx = min_bounds[0] + x * stud_pitch + stud_pitch / 2.0
                    cy = min_bounds[1] + j * brick_height + brick_height
                    cz = min_bounds[2] + (z + 0.5) * stud_pitch + stud_pitch
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, 3] = [cx, -cy, cz]
                    parts.append(ParsedPart(part_id="3004.dat", color=color_idx, transform=transform, step_id=step_id))
                    
        for x in range(nx - 1):
            for z in range(nz):
                coords_2x1 = [(x, z), (x + 1, z)]
                if all(c in layer_set and c not in layer_used for c in coords_2x1):
                    for c in coords_2x1:
                        layer_used.add(c)
                        used.add((c[0], j, c[1]))
                    cx = min_bounds[0] + (x + 0.5) * stud_pitch + stud_pitch
                    cy = min_bounds[1] + j * brick_height + brick_height
                    cz = min_bounds[2] + z * stud_pitch + stud_pitch / 2.0
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
                    transform[:3, 3] = [cx, -cy, cz]
                    parts.append(ParsedPart(part_id="3004.dat", color=color_idx, transform=transform, step_id=step_id))

        # 3. Remaining voxels become 1x1 bricks
        for x in range(nx):
            for z in range(nz):
                c = (x, z)
                if c in layer_set and c not in layer_used:
                    layer_used.add(c)
                    used.add((x, j, z))
                    cx = min_bounds[0] + x * stud_pitch + stud_pitch / 2.0
                    cy = min_bounds[1] + j * brick_height + brick_height
                    cz = min_bounds[2] + z * stud_pitch + stud_pitch / 2.0
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, 3] = [cx, -cy, cz]
                    parts.append(ParsedPart(part_id="3005.dat", color=color_idx, transform=transform, step_id=step_id))
                    
        step_id += 1
        
    return parts

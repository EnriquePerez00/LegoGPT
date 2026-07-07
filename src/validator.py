import numpy as np
import networkx as nx
from src.parser import ParsedPart

PART_DIMENSIONS = {
    "3005.dat": (20.0, 24.0, 20.0),
    "3004.dat": (20.0, 24.0, 40.0),
    "3003.dat": (40.0, 24.0, 40.0),
    "3002.dat": (40.0, 24.0, 60.0),
    "3001.dat": (40.0, 24.0, 80.0),
    "3010.dat": (20.0, 24.0, 80.0),
    "3009.dat": (20.0, 24.0, 120.0),
    "3008.dat": (20.0, 24.0, 160.0),
    "3007.dat": (40.0, 24.0, 160.0),
    "3006.dat": (40.0, 24.0, 200.0),
    "3024.dat": (20.0, 8.0, 20.0),
    "3023.dat": (20.0, 8.0, 40.0),
    "3022.dat": (40.0, 8.0, 40.0),
    "3021.dat": (40.0, 8.0, 60.0),
    "3020.dat": (40.0, 8.0, 80.0),
    "3710.dat": (20.0, 8.0, 80.0),
    "3666.dat": (20.0, 8.0, 120.0),
    "3460.dat": (20.0, 8.0, 160.0),
    "3032.dat": (80.0, 8.0, 120.0),
    "3031.dat": (80.0, 8.0, 80.0),
    "3030.dat": (80.0, 8.0, 200.0),
    "3035.dat": (80.0, 8.0, 160.0),
    "3034.dat": (40.0, 8.0, 160.0),
    "3832.dat": (40.0, 8.0, 200.0),
    "3062b.dat": (20.0, 24.0, 20.0),
    "3941.dat": (40.0, 24.0, 40.0),
    "3036.dat": (120.0, 8.0, 160.0),
    "3958.dat": (120.0, 8.0, 120.0),
    "3005p01.dat": (20.0, 24.0, 20.0),
    "3004p01.dat": (20.0, 24.0, 40.0),
    "3003p01.dat": (40.0, 24.0, 40.0),
    "3010p01.dat": (20.0, 24.0, 80.0),
    
    # 75400-1 specific part dimensions
    "3070.dat": (20.0, 8.0, 20.0),
    "3069.dat": (20.0, 8.0, 40.0),
    "63864.dat": (20.0, 8.0, 60.0),
    "15573.dat": (20.0, 8.0, 40.0),
    "26601.dat": (40.0, 8.0, 40.0),
    "11477.dat": (20.0, 16.0, 40.0),
    "15556.dat": (10.0, 24.0, 10.0),
    "28697.dat": (10.0, 64.0, 10.0),
    "28701.dat": (20.0, 24.0, 20.0),
    "35338.dat": (20.0, 16.0, 20.0),
    "35380.dat": (20.0, 8.0, 20.0),
    "35480.dat": (20.0, 8.0, 40.0),
    "35787.dat": (40.0, 8.0, 40.0),
    "3665.dat": (20.0, 24.0, 40.0),
    "41769.dat": (40.0, 8.0, 80.0),
    "41770.dat": (40.0, 8.0, 80.0),
    "41822.dat": (80.0, 8.0, 80.0),
    "42610.dat": (20.0, 16.0, 20.0),
    "4274.dat": (10.0, 20.0, 10.0),
    "42923.dat": (20.0, 8.0, 40.0),
    "48205.dat": (80.0, 8.0, 120.0),
    "48208.dat": (80.0, 8.0, 120.0),
    "50340.dat": (20.0, 8.0, 40.0),
    "5091.dat": (20.0, 8.0, 40.0),
    "5092.dat": (20.0, 8.0, 40.0),
    "51483.dat": (20.0, 8.0, 80.0),
    "5414.dat": (20.0, 24.0, 80.0),
    "5415.dat": (20.0, 24.0, 80.0),
    "65426.dat": (40.0, 8.0, 80.0),
    "65429.dat": (40.0, 8.0, 80.0),
    "69754.dat": (20.0, 8.0, 40.0),
    "69755.dat": (20.0, 8.0, 20.0),
    "76382.dat": (20.0, 24.0, 10.0),
    "111870.dat": (20.0, 24.0, 20.0),
    "112754.dat": (20.0, 24.0, 20.0),
    "112755.dat": (40.0, 24.0, 40.0),
    "32803.dat": (40.0, 16.0, 40.0),
    "79491.dat": (40.0, 8.0, 40.0),
}


def get_part_dimensions(part_id: str) -> tuple[float, float, float]:
    """Helper to get part dimensions (width, height, depth), with fallback for unknown parts."""
    dim = PART_DIMENSIONS.get(part_id)
    if not dim:
        # Try to infer from name / description in PART_NAMES
        name = ""
        try:
            from src.sequence_visualizer import PART_NAMES
            name = PART_NAMES.get(part_id, "").lower()
        except Exception:
            pass
            
        part_lower = part_id.lower()
        # Default height
        h = 24.0 # default brick height
        if "plate" in part_lower or "tile" in part_lower or "plate" in name or "tile" in name:
            h = 8.0
        elif "brick" in part_lower or "brick" in name:
            h = 24.0
            
        # Try to find dimensions like "2x8", "1x2" in description or ID name
        import re
        m = re.search(r"(\d+)\s*x\s*(\d+)", name)
        if not m:
            m = re.search(r"(\d+)\s*x\s*(\d+)", part_lower)
            
        if m:
            dim_x = float(m.group(1)) * 20.0
            dim_z = float(m.group(2)) * 20.0
            return (dim_x, h, dim_z)
            
        # prefix based fallbacks
        if part_lower.startswith("302") or part_lower.startswith("307"):
            return (20.0, 8.0, 20.0)
        return (20.0, 24.0, 20.0)  # Standard 1x1 brick fallback
    return dim

def get_part_corners_world(part: ParsedPart) -> np.ndarray:
    """Computes the 8 corners of the part's bounding box in world space."""
    dim = get_part_dimensions(part.part_id)
    w, h, d = dim[0], dim[1], dim[2]
    # Local corners: centered in X and Z, Y goes from -h (top) to 0 (bottom)
    local_corners = np.array([
        [-w/2, 0.0, -d/2, 1.0],
        [ w/2, 0.0, -d/2, 1.0],
        [-w/2, 0.0,  d/2, 1.0],
        [ w/2, 0.0,  d/2, 1.0],
        [-w/2,  -h, -d/2, 1.0],
        [ w/2,  -h, -d/2, 1.0],
        [-w/2,  -h,  d/2, 1.0],
        [ w/2,  -h,  d/2, 1.0]
    ], dtype=np.float32)
    # Transform to world space
    world_corners = (part.transform @ local_corners.T).T[:, :3]
    return world_corners

def get_studs_and_sockets_world(part: ParsedPart) -> tuple[np.ndarray, np.ndarray]:
    """Generates the coordinates of the studs (top) and sockets (bottom) in world space."""
    dim = get_part_dimensions(part.part_id)
    W, H, D = dim[0], dim[1], dim[2]
    N = max(1, int(round(W / 20.0)))
    M = max(1, int(round(D / 20.0)))
    
    studs_local = []
    sockets_local = []
    for i in range(N):
        x = -10.0 * (N - 1) + 20.0 * i
        for j in range(M):
            z = -10.0 * (M - 1) + 20.0 * j
            studs_local.append([x, -H, z, 1.0])
            sockets_local.append([x, 0.0, z, 1.0])
            
    studs_world = (part.transform @ np.array(studs_local).T).T[:, :3]
    sockets_world = (part.transform @ np.array(sockets_local).T).T[:, :3]
    return studs_world, sockets_world

def check_connection(part_a: ParsedPart, part_b: ParsedPart) -> bool:
    """Determines if there is a stud-to-socket connection between part_a and part_b."""
    studs_a, sockets_a = get_studs_and_sockets_world(part_a)
    studs_b, sockets_b = get_studs_and_sockets_world(part_b)
    
    # 1.0 LDU connection tolerance threshold
    threshold = 1.0
    
    # Check if a stud of A connects to a socket of B
    for sa in studs_a:
        for sb in sockets_b:
            if np.linalg.norm(sa - sb) < threshold:
                return True
                
    # Check if a stud of B connects to a socket of A
    for sb in studs_b:
        for sa in sockets_a:
            if np.linalg.norm(sb - sa) < threshold:
                return True
                
    return False

def check_connection_optimized(part_a: ParsedPart, part_b: ParsedPart) -> bool:
    """Optimized connection check with vertical adjacency filter first."""
    corners_a = get_part_corners_world(part_a)
    corners_b = get_part_corners_world(part_b)
    
    y_min_a, y_max_a = np.min(corners_a[:, 1]), np.max(corners_a[:, 1])
    y_min_b, y_max_b = np.min(corners_b[:, 1]), np.max(corners_b[:, 1])
    
    # Vertical range connection check (tolerance 2.0 LDU)
    vertical_touch = (abs(y_max_a - y_min_b) < 2.0) or (abs(y_max_b - y_min_a) < 2.0)
    if not vertical_touch:
        return False
        
    return check_connection(part_a, part_b)

def obb_overlap_sat(c_a: np.ndarray, axes_a: np.ndarray, h_a: np.ndarray,
                    c_b: np.ndarray, axes_b: np.ndarray, h_b: np.ndarray) -> bool:
    """
    Separating Axis Theorem (SAT) for 3D Oriented Bounding Boxes (OBB).
    Returns True if OBB A and OBB B overlap, False otherwise.
    """
    T = c_b - c_a
    
    # Projecting onto the 15 candidate axes
    # 1. Face normals of A
    for i in range(3):
        L = axes_a[:, i]
        r_a = h_a[i]
        r_b = sum(h_b[k] * np.abs(np.dot(axes_b[:, k], L)) for k in range(3))
        if np.abs(np.dot(T, L)) > r_a + r_b:
            return False
            
    # 2. Face normals of B
    for i in range(3):
        L = axes_b[:, i]
        r_a = sum(h_a[k] * np.abs(np.dot(axes_a[:, k], L)) for k in range(3))
        r_b = h_b[i]
        if np.abs(np.dot(T, L)) > r_a + r_b:
            return False
            
    # 3. Cross products of axes
    for i in range(3):
        for j in range(3):
            L = np.cross(axes_a[:, i], axes_b[:, j])
            L_norm = np.linalg.norm(L)
            if L_norm < 1e-5:
                continue
            L = L / L_norm
            
            r_a = sum(h_a[k] * np.abs(np.dot(axes_a[:, k], L)) for k in range(3))
            r_b = sum(h_b[k] * np.abs(np.dot(axes_b[:, k], L)) for k in range(3))
            if np.abs(np.dot(T, L)) > r_a + r_b:
                return False
                
    return True

def get_part_mesh(part_id: str, transform: np.ndarray):
    """Dynamically creates a Trimesh representation of the part if trimesh is installed."""
    import trimesh
    dim = get_part_dimensions(part_id)
    # Subtract 0.2 LDU clearance from width/depth, and 0.02 LDU from height to prevent false collision triggers
    w, h, d = dim[0] - 0.2, dim[1] - 0.02, dim[2] - 0.2
    mesh = trimesh.creation.box(extents=(w, h, d))
    
    # Translate so the bottom center of the brick is the local (0, 0, 0)
    shift = np.eye(4, dtype=np.float32)
    shift[1, 3] = -dim[1] / 2.0
    mesh.apply_transform(shift)
    
    # Apply world transform
    mesh.apply_transform(transform)
    return mesh

def check_collisions(parts: list[ParsedPart]) -> list[tuple[int, int]]:
    """
    Checks all parts for intersections.
    Returns a list of colliding index pairs (i, j).
    """
    try:
        # Try importing and using trimesh CollisionManager
        import trimesh
        manager = trimesh.collision.CollisionManager()
        for idx, part in enumerate(parts):
            mesh = get_part_mesh(part.part_id, part.transform)
            manager.add_object(f"part_{idx}", mesh)
            
        in_collision, names = manager.in_collision_internal(return_names=True)
        collisions = []
        if in_collision:
            for name_a, name_b in names:
                idx_a = int(name_a.split("_")[1])
                idx_b = int(name_b.split("_")[1])
                collisions.append((idx_a, idx_b))
        return collisions
    except Exception:
        # Fallback to Separating Axis Theorem (SAT) using Numpy
        collisions = []
        num_parts = len(parts)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                part_a = parts[i]
                part_b = parts[j]
                dim_a = get_part_dimensions(part_a.part_id)
                dim_b = get_part_dimensions(part_b.part_id)
                
                # OBB properties for A
                c_a = (part_a.transform @ np.array([0.0, -dim_a[1]/2.0, 0.0, 1.0]))[:3]
                axes_a = part_a.transform[:3, :3]
                h_a = np.array([dim_a[0]/2.0 - 0.1, dim_a[1]/2.0 - 0.01, dim_a[2]/2.0 - 0.1])
                
                # OBB properties for B
                c_b = (part_b.transform @ np.array([0.0, -dim_b[1]/2.0, 0.0, 1.0]))[:3]
                axes_b = part_b.transform[:3, :3]
                h_b = np.array([dim_b[0]/2.0 - 0.1, dim_b[1]/2.0 - 0.01, dim_b[2]/2.0 - 0.1])
                
                if obb_overlap_sat(c_a, axes_a, h_a, c_b, axes_b, h_b):
                    collisions.append((i, j))
        return collisions

def check_connectivity_and_gravity(parts: list[ParsedPart]) -> bool:
    """
    Builds a connectivity graph and verifies that there are no floating parts.
    All parts must connect directly or indirectly to the base.
    Also strictly enforces that no part goes below ground level (Y > 1.0)
    and at most 4 parts reside on the ground level (|Y| <= 1.0).
    """
    if not parts:
        return True

    # 1. Ground level check: no parts below Y=0.0 (in LDraw Y increases downwards)
    for part in parts:
        if part.transform[1, 3] > 1.0:
            return False

    # 2. Base layer count: at most 4 parts on the ground surface
    ground_count = sum(1 for part in parts if abs(part.transform[1, 3]) <= 1.0)
    if ground_count > 4:
        return False
        
    G = nx.Graph()
    for idx in range(len(parts)):
        G.add_node(idx)
        
    # Build connection edges
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if check_connection_optimized(parts[i], parts[j]):
                G.add_edge(i, j)
                
    # Detect anchored base parts
    # The base consists of the parts with the lowest elevation (maximum Y value in LDraw coords)
    y_max_values = []
    for part in parts:
        corners = get_part_corners_world(part)
        y_max_values.append(np.max(corners[:, 1]))
        
    global_max_y = max(y_max_values)
    
    # Any part within 1.0 LDU of the absolute lowest point is anchored to the base
    anchored_nodes = [idx for idx, y in enumerate(y_max_values) if abs(y - global_max_y) < 1.0]
    
    # Virtual ground node
    G.add_node("ground")
    for idx in anchored_nodes:
        G.add_edge("ground", idx)
        
    # Check if the entire assembly is in the same component as the ground
    connected_components = list(nx.connected_components(G))
    for component in connected_components:
        if "ground" in component:
            return len(component) == len(parts) + 1
            
    return False

def validate_rules(parts: list[ParsedPart]) -> bool:
    """Verifies that all parts adhere to standard vocabulary limits (design IDs and color bounds)."""
    from src.parser import ALLOWED_PARTS, ALLOWED_COLORS
    for part in parts:
        if part.part_id not in ALLOWED_PARTS:
            return False
        if part.color not in ALLOWED_COLORS:
            return False
    return True

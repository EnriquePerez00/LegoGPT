import numpy as np
import torch
from torch_geometric.data import Data
from dataclasses import dataclass

ALLOWED_PARTS = [
    "3005.dat",
    "3004.dat",
    "3003.dat",
    "3002.dat",
    "3001.dat",
    "3010.dat",
    "3009.dat",
    "3008.dat",
    "3007.dat",
    "3006.dat",
    "3024.dat",
    "3023.dat",
    "3022.dat",
    "3021.dat",
    "3020.dat",
    "3710.dat",
    "3666.dat",
    "3460.dat",
    "3032.dat",
    "3031.dat",
    "3030.dat",
    "3035.dat",
    "3034.dat",
    "3832.dat",
    "3062b.dat",
    "3941.dat",
    "3036.dat",
    "3958.dat",
    "3005p01.dat",
    "3004p01.dat",
    "3003p01.dat",
    "3010p01.dat"
]
ALLOWED_COLORS = list(range(16))


@dataclass
class ParsedPart:
    part_id: str
    color: int
    transform: np.ndarray  # 4x4 homogenous matrix
    step_id: int

def parse_ldraw_file(file_path: str) -> list[ParsedPart]:
    """
    Parses a standard LDraw file (.ldr or .mpd) and extracts parts, colors,
    3D transformation matrices, and construction step identifiers.
    
    Args:
        file_path: Absolute path to the LDraw file.
        
    Returns:
        A list of ParsedPart objects.
    """
    parts = []
    current_step = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            cmd_type = tokens[0]
            
            if cmd_type == "0":
                # Check for comment instruction signaling a step separator
                if len(tokens) >= 2 and tokens[1].upper() == "STEP":
                    current_step += 1
                    if len(tokens) >= 4 and tokens[2].upper() == "PAGE":
                        try:
                            current_page = int(tokens[3])
                        except ValueError:
                            current_page = current_step
                    else:
                        current_page = current_step
            elif cmd_type == "1":
                # Instruction type 1 is a part reference.
                # Format: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part_name>
                if len(tokens) >= 15:
                    color = int(tokens[1])
                    # Position
                    x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                    # Rotation matrix
                    a, b, c = float(tokens[5]), float(tokens[6]), float(tokens[7])
                    d, e, f = float(tokens[8]), float(tokens[9]), float(tokens[10])
                    g, h, i = float(tokens[11]), float(tokens[12]), float(tokens[13])
                    part_name = tokens[14].lower()
                    
                    # Assemble 4x4 matrix
                    transform = np.array([
                        [a, b, c, x],
                        [d, e, f, y],
                        [g, h, i, z],
                        [0.0, 0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    
                    # Track page number if parsed
                    if 'current_page' not in locals():
                        current_page = current_step
                        
                    part = ParsedPart(
                        part_id=part_name,
                        color=color,
                        transform=transform,
                        step_id=current_step
                    )
                    part.page_num = current_page
                    parts.append(part)
                    
    return parts

def build_pyg_graph(parts: list[ParsedPart], allowed_parts=None, allowed_colors=None) -> Data:
    """
    Converts a list of ParsedParts into a PyTorch Geometric Graph (Data).
    Nodes are the parts, and edges are physical contact connections.
    
    Args:
        parts: List of ParsedPart objects.
        allowed_parts: Optional list of allowed part IDs.
        allowed_colors: Optional list of allowed color codes.
        
    Returns:
        A torch_geometric.data.Data object.
    """
    # Import check_connection_optimized locally to avoid circular dependencies
    from src.validator import check_connection_optimized
    
    if allowed_parts is None:
        allowed_parts = ALLOWED_PARTS
    if allowed_colors is None:
        allowed_colors = ALLOWED_COLORS
        
    num_nodes = len(parts)
    node_features = []
    
    # 1. Build Node Features
    for part in parts:
        # Part ID One-hot encoding
        part_one_hot = [1.0 if part.part_id == p else 0.0 for p in allowed_parts]
        
        # Color One-hot encoding
        color_one_hot = [1.0 if part.color == c else 0.0 for c in allowed_colors]
        
        # Translation vector (X, Y, Z)
        translation = part.transform[:3, 3].tolist()
        
        # Rotation vector (9 components)
        rotation = part.transform[:3, :3].flatten().tolist()
        
        # Combined Node Feature Vector (6 + 16 + 3 + 9 = 34 dimensions)
        feature_vector = part_one_hot + color_one_hot + translation + rotation
        node_features.append(feature_vector)
        
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # 2. Build Edge Connectivity (edge_index and edge_attr)
    edge_list = []
    edge_features = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if check_connection_optimized(parts[i], parts[j]):
                edge_list.append([i, j])
                
                # Relative translation and rotation features
                t_rel = parts[j].transform[:3, 3] - parts[i].transform[:3, 3]
                R_rel = parts[i].transform[:3, :3].T @ parts[j].transform[:3, :3]
                
                edge_feat = t_rel.tolist() + R_rel.flatten().tolist()
                edge_features.append(edge_feat)
                
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 12), dtype=torch.float32)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

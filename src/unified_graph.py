"""
Unified Lego Graph Representation Model.
Unifies:
  1. Manual Connection Graph (JSON / PDF instruction manuals pipeline)
  2. 3D Sequence Graph (LDraw / OMR / Studio .io pipeline)
  3. PyTorch Geometric (PyG) Tensor Graph for Generative AI Models
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from src.parser import ParsedPart, ALLOWED_PARTS, ALLOWED_COLORS, Data
from src.validator import check_connection_optimized, get_studs_and_sockets_world

class UnifiedLegoGraph:
    """
    Canonical representation of a Lego construction graph.
    Bridges the semantic manual assembly pipeline (JSON) and the 3D / PyG neural tensor pipeline.
    """
    def __init__(self, model_name: str = "", num_parts: int = 0):
        self.model_name = model_name
        self.num_parts = num_parts
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

    @classmethod
    def from_parsed_parts(cls, parts: List[ParsedPart], model_name: str = "") -> "UnifiedLegoGraph":
        """Builds unified graph from parsed LDraw/Studio parts list."""
        ug = cls(model_name=model_name, num_parts=len(parts))
        
        # 1. Build Nodes
        for idx, p in enumerate(parts):
            # Extract 4x4 matrix and decomposition
            tf = p.transform
            R = tf[:3, :3]
            t = tf[:3, 3]
            
            node_entry = {
                "part_index": idx,
                "part_id": p.part_id,
                "color_code": p.color,
                "step_id": p.step_id,
                "translation": t.tolist(),
                "rotation": R.flatten().tolist(),
                "absolute_transform": tf.flatten().tolist(),
                "is_minifig": getattr(p, "is_minifig", False)
            }
            ug.nodes.append(node_entry)
            
        # 2. Build Physical Connections (Edges)
        for i in range(len(parts)):
            for j in range(len(parts)):
                if i == j:
                    continue
                if check_connection_optimized(parts[i], parts[j]):
                    t_rel = parts[j].transform[:3, 3] - parts[i].transform[:3, 3]
                    R_rel = parts[i].transform[:3, :3].T @ parts[j].transform[:3, :3]
                    
                    ug.edges.append({
                        "source": i,
                        "target": j,
                        "relative_translation": t_rel.tolist(),
                        "relative_rotation": R_rel.flatten().tolist(),
                        "connection_type": "stud_socket"
                    })
                    
        return ug

    @classmethod
    def from_manual_json(cls, data: Dict[str, Any]) -> "UnifiedLegoGraph":
        """Builds unified graph from PDF manual assembly JSON."""
        seq = data.get("sequence", [])
        name = data.get("set_name", "")
        ug = cls(model_name=name, num_parts=len(seq))
        
        for idx, item in enumerate(seq):
            tf = np.array(item.get("absolute_transform", np.eye(4))).reshape(4, 4)
            ug.nodes.append({
                "part_index": item.get("part_index", idx),
                "part_id": item.get("part_id", ""),
                "color_code": item.get("color_code", 0),
                "step_id": item.get("step_id", item.get("page_num", 1)),
                "translation": tf[:3, 3].tolist(),
                "rotation": tf[:3, :3].flatten().tolist(),
                "absolute_transform": tf.flatten().tolist(),
                "is_minifig": item.get("is_minifig", False)
            })
            
            for conn in item.get("connections", []):
                parent_idx = conn.get("parent_part_index")
                if parent_idx is not None:
                    rel_tf = np.array(conn.get("relative_transform", np.eye(4))).reshape(4, 4)
                    ug.edges.append({
                        "source": parent_idx,
                        "target": item.get("part_index", idx),
                        "relative_translation": rel_tf[:3, 3].tolist(),
                        "relative_rotation": rel_tf[:3, :3].flatten().tolist(),
                        "parent_anchor": conn.get("parent_anchor", "unknown"),
                        "child_anchor": conn.get("child_anchor", "unknown"),
                        "connection_type": "physical_anchor"
                    })
        return ug

    def to_pyg_data(self, allowed_parts: Optional[List[str]] = None, allowed_colors: Optional[List[int]] = None) -> Data:
        """Converts into PyTorch Geometric Data tensor for training / inference."""
        if allowed_parts is None:
            allowed_parts = ALLOWED_PARTS
        if allowed_colors is None:
            allowed_colors = ALLOWED_COLORS

        node_features = []
        for n in self.nodes:
            p_vec = [1.0 if n["part_id"] == p else 0.0 for p in allowed_parts]
            c_vec = [1.0 if n["color_code"] == c else 0.0 for c in allowed_colors]
            t_vec = n["translation"]
            r_vec = n["rotation"]
            node_features.append(p_vec + c_vec + t_vec + r_vec)

        x = torch.tensor(node_features, dtype=torch.float32)
        step_ids = torch.tensor([n["step_id"] for n in self.nodes], dtype=torch.long)

        edge_indices = []
        edge_attrs = []
        for e in self.edges:
            edge_indices.append([e["source"], e["target"]])
            edge_attrs.append(e["relative_translation"] + e["relative_rotation"])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 12), dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, step_id=step_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes canonical graph to JSON-friendly dictionary."""
        return {
            "model_name": self.model_name,
            "num_parts": self.num_parts,
            "nodes": self.nodes,
            "edges": self.edges
        }

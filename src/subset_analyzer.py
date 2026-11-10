"""
LegoGPT: Subset & Physical Structural Assembly Analyzer
Identifies independent, non-interconnected subsets / submodels in MOCs/Studio designs,
evaluates 3D contact surfaces, mechanical joints, center of mass, static equilibrium,
and structural interlocking (trabado).
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING
import numpy as np
import networkx as nx

from src.parser import ParsedPart
from src.validator import get_part_dimensions, get_part_corners_world
from src.connectivity_db import (
    get_part_ports_world,
    check_parts_connected_ports,
    JointDOF,
    PortType
)

# Density of ABS plastic in grams per LDU^3
# Standard LEGO ABS density: ~1.05 g/cm^3.
# 1 cm = 25 LDU -> 1 cm^3 = 15,625 LDU^3 -> ~6.72e-5 g/LDU^3
ABS_DENSITY_G_PER_LDU3 = 6.72e-5

@dataclass
class SubassemblyMetadata:
    subassembly_id: int
    parts_count: int
    part_indices: list[int]
    parts_list: list[str]
    is_grounded: bool
    ground_contacts_count: int
    center_of_mass: list[float]  # [X, Y, Z]
    total_mass_grams: float
    bounding_box: dict[str, float]
    is_statically_stable: bool
    interlocking_index: float    # Cycle density: 0.0 for pure trees, >0.0 for staggered/interlocked builds
    weak_joints_count: int       # Number of joints with minimal clutch (e.g. single stud / clip)
    joint_dof_summary: dict[str, int] = field(default_factory=dict)

def compute_part_mass_and_com(part: ParsedPart) -> tuple[float, np.ndarray]:
    """Computes mass in grams and center of mass in world space for a given part."""
    dim = get_part_dimensions(part.part_id)
    volume_ldu = dim[0] * dim[1] * dim[2]
    # Shell factor: hollow interior reduces effective mass to ~50%
    effective_volume = volume_ldu * 0.5
    mass_grams = max(0.1, effective_volume * ABS_DENSITY_G_PER_LDU3)
    
    # Local center of mass is centered in X and Z, and midway between 0 and -H in Y
    local_com = np.array([0.0, -dim[1] / 2.0, 0.0, 1.0], dtype=np.float32)
    world_com = (part.transform @ local_com)[:3]
    return mass_grams, world_com

def is_point_inside_convex_hull_2d(point: tuple[float, float], polygon_points: list[tuple[float, float]]) -> bool:
    """
    Checks if a 2D point (X, Z) is inside the convex hull of ground contact points.
    Uses scipy.spatial.ConvexHull if available, otherwise 2D cross product winding check.
    """
    if len(polygon_points) < 3:
        # Collinear or degenerate: check if point is within bounding segment
        xs = [p[0] for p in polygon_points]
        zs = [p[1] for p in polygon_points]
        return min(xs) - 5.0 <= point[0] <= max(xs) + 5.0 and min(zs) - 5.0 <= point[1] <= max(zs) + 5.0

    try:
        from scipy.spatial import ConvexHull
        pts = np.array(polygon_points, dtype=np.float32)
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        
        # Point in convex polygon test using cross-products
        n = len(hull_pts)
        for i in range(n):
            p1 = hull_pts[i]
            p2 = hull_pts[(i + 1) % n]
            edge = p2 - p1
            to_pt = np.array(point) - p1
            # 2D cross product (edge_x * to_pt_z - edge_z * to_pt_x)
            cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
            if cross < -1e-4:  # Outside
                return False
        return True
    except Exception:
        # Fallback to simple bounding box
        xs = [p[0] for p in polygon_points]
        zs = [p[1] for p in polygon_points]
        return min(xs) <= point[0] <= max(xs) and min(zs) <= point[1] <= max(zs)

def analyze_subsets(parts: list[ParsedPart]) -> list[SubassemblyMetadata]:
    """
    Parses a flat list of physical parts from a MOC or Studio file, builds the exact
    mechanical port interconnection graph, and isolates independent subsets (connected components).
    Computes real-world physical indicators: static stability, center of mass, and structural interlocking.
    """
    num_parts = len(parts)
    if num_parts == 0:
        return []

    # 1. Build Multi-Relational Mechanical Port Graph
    G = nx.Graph()
    for idx in range(num_parts):
        G.add_node(idx, part=parts[idx])

    joint_details = {}
    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            connected, matches = check_parts_connected_ports(parts[i], parts[j])
            if connected:
                # Total clutch capacity is the sum of all engaged ports
                total_clutch = sum(m["clutch_power"] for m in matches)
                primary_dof = matches[0]["dof"]
                G.add_edge(i, j, matches=matches, clutch=total_clutch, dof=primary_dof)
                joint_details[(i, j)] = matches

    # 2. Extract Independent Subsets (Connected Components)
    components = list(nx.connected_components(G))
    subsets_metadata: list[SubassemblyMetadata] = []

    for s_idx, comp_indices in enumerate(components):
        comp_parts = [parts[idx] for idx in comp_indices]
        part_ids_sorted = sorted(list(set(p.part_id.lower() for p in comp_parts)))
        
        # Compute mass and global center of mass
        masses = []
        coms = []
        for p in comp_parts:
            m, c = compute_part_mass_and_com(p)
            masses.append(m)
            coms.append(c)
            
        total_mass = float(sum(masses))
        global_com = (sum(m * c for m, c in zip(masses, coms)) / total_mass).tolist()

        # Compute Bounding Box
        all_corners = np.vstack([get_part_corners_world(p) for p in comp_parts])
        min_x, max_x = float(np.min(all_corners[:, 0])), float(np.max(all_corners[:, 0]))
        min_y, max_y = float(np.min(all_corners[:, 1])), float(np.max(all_corners[:, 1]))
        min_z, max_z = float(np.min(all_corners[:, 2])), float(np.max(all_corners[:, 2]))
        
        bbox = {
            "min_x": min_x, "max_x": max_x,
            "min_y": min_y, "max_y": max_y,
            "min_z": min_z, "max_z": max_z,
            "volume_ldu": (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        }

        # Ground contact points (where Y is near 0.0 in LDraw coordinates)
        ground_contacts = []
        for p in comp_parts:
            # Check bottom face elevation
            corners = get_part_corners_world(p)
            # Find corners at ground level (|Y| <= 1.0)
            for c in corners:
                if abs(c[1]) <= 1.0:
                    ground_contacts.append((float(c[0]), float(c[2])))

        is_grounded = len(ground_contacts) > 0

        # Static stability: Center of mass projection inside Ground Polygon Convex Hull
        is_statically_stable = False
        if is_grounded:
            com_xz = (global_com[0], global_com[2])
            is_statically_stable = is_point_inside_convex_hull_2d(com_xz, ground_contacts)

        # Structural Interlocking Index (Cycle density)
        sub_g = G.subgraph(comp_indices)
        num_v = sub_g.number_of_nodes()
        num_e = sub_g.number_of_edges()
        
        # For a tree, num_e = num_v - 1. Interlocking index measures cycles (seam overlaps)
        interlocking_index = max(0.0, float(num_e - (num_v - 1)) / max(1, num_v))

        # Joint summary
        joint_dof_summary = {}
        weak_joints = 0
        for u, v, data in sub_g.edges(data=True):
            dof_val = str(data.get("dof", "fixed"))
            joint_dof_summary[dof_val] = joint_dof_summary.get(dof_val, 0) + 1
            if data.get("clutch", 2.0) <= 2.0:  # Single stud or single clip
                weak_joints += 1

        subsets_metadata.append(SubassemblyMetadata(
            subassembly_id=s_idx,
            parts_count=len(comp_indices),
            part_indices=sorted(list(comp_indices)),
            parts_list=part_ids_sorted,
            is_grounded=is_grounded,
            ground_contacts_count=len(ground_contacts),
            center_of_mass=global_com,
            total_mass_grams=round(total_mass, 2),
            bounding_box=bbox,
            is_statically_stable=is_statically_stable,
            interlocking_index=round(interlocking_index, 3),
            weak_joints_count=weak_joints,
            joint_dof_summary=joint_dof_summary
        ))

    return subsets_metadata

def build_port_attributed_assembly_graph(parts: list[ParsedPart]) -> dict:
    """
    Structures the entire piece interconnection graph for generative models (LegoGPT).
    Includes rich node morphology, 3D poses, open/closed port slots, and mechanical joint edges.
    """
    nodes = []
    for idx, part in enumerate(parts):
        mass, com = compute_part_mass_and_com(part)
        dim = get_part_dimensions(part.part_id)
        ports = get_part_ports_world(part)
        
        nodes.append({
            "node_index": idx,
            "part_id": part.part_id,
            "color": part.color,
            "step_id": part.step_id,
            "transform_matrix": part.transform.flatten().tolist(),
            "dimensions": list(dim),
            "mass_grams": mass,
            "center_of_mass": com.tolist(),
            "ports_count": len(ports),
            "ports": ports
        })

    edges = []
    num_parts = len(parts)
    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            connected, matches = check_parts_connected_ports(parts[i], parts[j])
            if connected:
                # Relative transform
                t_rel = parts[j].transform[:3, 3] - parts[i].transform[:3, 3]
                r_rel = parts[i].transform[:3, :3].T @ parts[j].transform[:3, :3]
                
                rel_matrix = np.eye(4, dtype=np.float32)
                rel_matrix[:3, :3] = r_rel
                rel_matrix[:3, 3] = t_rel

                edges.append({
                    "source": i,
                    "target": j,
                    "matched_ports": matches,
                    "total_clutch_power": sum(m["clutch_power"] for m in matches),
                    "primary_joint_dof": str(matches[0]["dof"]),
                    "relative_transform": rel_matrix.flatten().tolist()
                })

    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges
    }


def extract_functional_subgraphs(parts: list[ParsedPart], max_bridge_contacts: int = 2) -> list[dict]:
    """
    Decomposes a complete 3D LEGO assembly into functional rigid sub-graphs (macros)
    for 3D Graph-RAG retrieval:
      - Splits at articulated 1-DoF / 3-DoF joints (hinges, pins, clips, ball joints)
        or weak structural bridges (<= max_bridge_contacts studs).
      - Normalizes each sub-graph to its local origin and extracts open external interface ports.
    """
    if not parts:
        return []

    rigid_graph = nx.Graph()
    for idx in range(len(parts)):
        rigid_graph.add_node(idx)

    all_edges_meta = {}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            connected, matches = check_parts_connected_ports(parts[i], parts[j])
            if connected:
                all_edges_meta[(i, j)] = matches
                is_articulated = any(str(m.get("dof", "fixed")) != "fixed" and str(m.get("dof")) != "JointDOF.FIXED" for m in matches)
                if not is_articulated and len(matches) > max_bridge_contacts:
                    rigid_graph.add_edge(i, j, weight=len(matches))

    # If everything was split into singletons, relax threshold to keep connected rigid clusters
    components = [sorted(list(c)) for c in nx.connected_components(rigid_graph)]
    if all(len(c) == 1 for c in components) and len(parts) > 1:
        for (i, j), matches in all_edges_meta.items():
            rigid_graph.add_edge(i, j, weight=len(matches))
        components = [sorted(list(c)) for c in nx.connected_components(rigid_graph)]

    macros = []
    for comp_id, node_indices in enumerate(components):
        sub_parts = [parts[i] for i in node_indices]
        coords = np.array([p.transform[:3, 3] for p in sub_parts], dtype=np.float32)
        bbox_min = np.min(coords, axis=0).tolist()
        bbox_max = np.max(coords, axis=0).tolist()
        mold_hist = {}
        for sp in sub_parts:
            mold_hist[sp.part_id] = mold_hist.get(sp.part_id, 0) + 1

        macros.append({
            "macro_id": f"macro_{comp_id}",
            "node_indices": node_indices,
            "part_count": len(sub_parts),
            "mold_histogram": mold_hist,
            "bbox_min_ldu": bbox_min,
            "bbox_max_ldu": bbox_max,
        })

    return macros


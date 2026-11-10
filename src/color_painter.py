"""
SemanticColorPainter: Post-Processing Color Assignment for Monochrome 3D LEGO Graphs.
Decouples 3D structural geometry generation (Monochrome Phase 1) from aesthetic color
assignment (Semantic Painting Phase 2) under strict closed-inventory constraints.

Enforces:
  1. Interior Occlusion Hiding: Assigns rare/odd inventory colors to occluded interior bricks.
  2. Bilateral X-Symmetry: Forces symmetric left/right part pairs across X=0 to share identical colors.
  3. Spatial Color-Blocking: Minimizes color changes across physically connected exterior parts.
"""

from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.parser import ParsedPart
from src.validator import check_connection_optimized, get_part_dimensions


class SemanticColorPainter:
    """
    Assigns inventory-valid LEGO color codes to a monochrome 3D LEGO assembly (`list[ParsedPart]`).
    """

    def __init__(self, symmetry_tol_ldu: float = 2.0):
        self.symmetry_tol_ldu = symmetry_tol_ldu

    @staticmethod
    def group_monochrome_inventory(colored_inventory: Dict[Tuple[str, int], int]) -> Dict[str, int]:
        """
        Aggregates a color-aware inventory {(part_id, color_id): qty} into a monochrome
        mold-only inventory {part_id: total_qty} for Phase 1 structural generation.
        """
        mono: Dict[str, int] = defaultdict(int)
        for (part_id, _), qty in colored_inventory.items():
            clean_id = part_id.lower()
            if not clean_id.endswith(".dat"):
                clean_id += ".dat"
            mono[clean_id] += int(qty)
        return dict(mono)

    def _compute_occlusion_scores(self, parts: List[ParsedPart]) -> List[float]:
        """
        Estimates how occluded/interior each part is based on surrounding parts above and around it.
        Higher score = more interior/hidden (ideal for consuming leftover/clash colors).
        """
        if not parts:
            return []
        positions = np.array([p.transform[:3, 3] for p in parts], dtype=np.float32)
        center_xz = np.mean(positions[:, [0, 2]], axis=0)
        max_r = max(20.0, float(np.max(np.linalg.norm(positions[:, [0, 2]] - center_xz, axis=1))))

        scores: List[float] = []
        for idx, part in enumerate(parts):
            pos = positions[idx]
            radial_dist = float(np.linalg.norm(pos[[0, 2]] - center_xz)) / max_r
            # Count parts directly above (smaller Y in LDraw) and horizontally close
            above_count = 0
            for j, other_pos in enumerate(positions):
                if j == idx:
                    continue
                if other_pos[1] < pos[1] - 4.0 and np.linalg.norm(other_pos[[0, 2]] - pos[[0, 2]]) < 30.0:
                    above_count += 1
            # Interiority combines central radial position + being covered from above
            interior_score = (1.0 - radial_dist) * 0.5 + min(1.0, above_count / 3.0) * 0.5
            scores.append(interior_score)
        return scores

    def _find_symmetry_pairs(self, parts: List[ParsedPart]) -> Tuple[Dict[int, int], List[int]]:
        """
        Identifies bilateral mirror pairs across the X=0 plane.
        Returns (pair_map {i: j}, unpaired_indices).
        """
        n = len(parts)
        paired: Dict[int, int] = {}
        used = set()

        for i in range(n):
            if i in used:
                continue
            pi = parts[i]
            xi, yi, zi = float(pi.transform[0, 3]), float(pi.transform[1, 3]), float(pi.transform[2, 3])
            if abs(xi) < 5.0:
                # Part lies on the centerline X=0
                continue
            for j in range(i + 1, n):
                if j in used:
                    continue
                pj = parts[j]
                if pi.part_id.lower() != pj.part_id.lower():
                    continue
                xj, yj, zj = float(pj.transform[0, 3]), float(pj.transform[1, 3]), float(pj.transform[2, 3])
                if (
                    abs(xi + xj) <= self.symmetry_tol_ldu
                    and abs(yi - yj) <= self.symmetry_tol_ldu
                    and abs(zi - zj) <= self.symmetry_tol_ldu
                ):
                    paired[i] = j
                    paired[j] = i
                    used.add(i)
                    used.add(j)
                    break

        unpaired = [i for i in range(n) if i not in used]
        return paired, unpaired

    def paint_parts(
        self,
        monochrome_parts: List[ParsedPart],
        colored_inventory: Optional[Dict[Tuple[str, int], int]] = None,
        default_primary_color: int = 71,
    ) -> List[ParsedPart]:
        """
        Paints a monochrome list of ParsedPart objects using available colors from `colored_inventory`
        while maximizing bilateral symmetry and spatial color-blocking continuity.
        """
        painted = deepcopy(monochrome_parts)
        if not painted:
            return painted

        # Organize available color pool per mold: {part_id: {color_id: remaining_qty}}
        mold_color_pool: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        if colored_inventory:
            for (pid, cid), qty in colored_inventory.items():
                clean_id = pid.lower()
                if not clean_id.endswith(".dat"):
                    clean_id += ".dat"
                if qty > 0:
                    mold_color_pool[clean_id][int(cid)] += int(qty)

        # Build adjacency list for spatial color-blocking
        n = len(painted)
        adj: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                if check_connection_optimized(painted[i], painted[j]):
                    adj[i].append(j)
                    adj[j].append(i)

        occlusion_scores = self._compute_occlusion_scores(painted)
        pair_map, _ = self._find_symmetry_pairs(painted)
        assigned: Dict[int, int] = {}

        # Step 1: Assign interior/occluded unpaired parts with rare/minority colors first
        interior_indices = sorted(
            [i for i in range(n) if i not in pair_map and occlusion_scores[i] >= 0.65],
            key=lambda i: occlusion_scores[i],
            reverse=True,
        )
        for idx in interior_indices:
            pid = painted[idx].part_id.lower()
            if not pid.endswith(".dat"):
                pid += ".dat"
            pool = mold_color_pool.get(pid, {})
            avail = [(cid, q) for cid, q in pool.items() if q > 0]
            if avail:
                # Pick rarest odd-count color to save abundant even-count colors for exterior & symmetry
                avail.sort(key=lambda x: (x[1] % 2 == 0, x[1]))
                chosen_cid = avail[0][0]
                assigned[idx] = chosen_cid
                pool[chosen_cid] -= 1

        # Step 2: Assign symmetric bilateral pairs (forcing identical color on left & right)
        visited_pairs = set()
        for i, j in pair_map.items():
            if i in visited_pairs or j in visited_pairs:
                continue
            visited_pairs.add(i)
            visited_pairs.add(j)
            pid = painted[i].part_id.lower()
            if not pid.endswith(".dat"):
                pid += ".dat"
            pool = mold_color_pool.get(pid, {})

            # Prefer neighbor color if already assigned and has >= 2 available
            neighbor_colors = [assigned[nb] for nb in (adj[i] + adj[j]) if nb in assigned]
            chosen_pair_cid = None
            for nc in neighbor_colors:
                if pool.get(nc, 0) >= 2:
                    chosen_pair_cid = nc
                    break

            if chosen_pair_cid is None:
                avail_pairs = [(cid, q) for cid, q in pool.items() if q >= 2]
                if avail_pairs:
                    avail_pairs.sort(key=lambda x: x[1], reverse=True)
                    chosen_pair_cid = avail_pairs[0][0]

            if chosen_pair_cid is not None:
                assigned[i] = chosen_pair_cid
                assigned[j] = chosen_pair_cid
                pool[chosen_pair_cid] -= 2
            else:
                # Fallback if no single color has qty >= 2
                for k_idx in (i, j):
                    avail_single = [(cid, q) for cid, q in pool.items() if q > 0]
                    if avail_single:
                        avail_single.sort(key=lambda x: x[1], reverse=True)
                        c_single = avail_single[0][0]
                        assigned[k_idx] = c_single
                        pool[c_single] -= 1
                    else:
                        assigned[k_idx] = default_primary_color

        # Step 3: Assign remaining exterior parts in topological order using spatial color-blocking
        for idx in range(n):
            if idx in assigned:
                continue
            pid = painted[idx].part_id.lower()
            if not pid.endswith(".dat"):
                pid += ".dat"
            pool = mold_color_pool.get(pid, {})

            # Color-blocking: match connected neighbors' dominant color if available in pool
            neighbor_counts: Dict[int, int] = defaultdict(int)
            for nb in adj[idx]:
                if nb in assigned:
                    neighbor_counts[assigned[nb]] += 1

            chosen_cid = None
            if neighbor_counts:
                for cand_cid, _ in sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True):
                    if pool.get(cand_cid, 0) > 0:
                        chosen_cid = cand_cid
                        break

            if chosen_cid is None:
                avail = [(cid, q) for cid, q in pool.items() if q > 0]
                if avail:
                    avail.sort(key=lambda x: x[1], reverse=True)
                    chosen_cid = avail[0][0]

            if chosen_cid is not None:
                assigned[idx] = chosen_cid
                pool[chosen_cid] -= 1
            else:
                # If no inventory constraint was provided, inherit neighbor color or default
                if neighbor_counts:
                    assigned[idx] = max(neighbor_counts.items(), key=lambda x: x[1])[0]
                else:
                    assigned[idx] = default_primary_color

        for idx, part in enumerate(painted):
            part.color = int(assigned.get(idx, default_primary_color))

        return painted

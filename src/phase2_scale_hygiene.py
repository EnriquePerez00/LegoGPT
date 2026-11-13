"""FASE 2: Scale Hygiene - Tier metrics, TTL, and experiment tracking."""

from enum import Enum
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class SetTier(Enum):
    """Classification by part count."""
    SMALL = (1, 100)
    MEDIUM = (101, 300)
    LARGE = (301, 10000)
    
    @classmethod
    def classify(cls, num_parts: int) -> str:
        for tier in cls:
            if tier.value[0] <= num_parts <= tier.value[1]:
                return tier.name
        return "UNKNOWN"


def compute_tier_metrics(moc_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """FASE 2: Segregates KPIs by set tier (small/medium/large)."""
    tier_data: Dict[str, List[Dict]] = {"SMALL": [], "MEDIUM": [], "LARGE": []}
    
    for res in moc_results:
        total_steps = res.get("total_steps", 1)
        tier = SetTier.classify(total_steps)
        if tier != "UNKNOWN":
            tier_data[tier].append(res)
    
    tier_metrics = {}
    for tier, results in tier_data.items():
        if results:
            completed = sum(1 for r in results if r.get("completed_all_steps"))
            tier_metrics[tier] = {
                "mean_survival_ratio": sum(r.get("survival_ratio", 0.0) for r in results) / len(results),
                "pass_rate_percent": 100.0 * completed / len(results),
                "num_sets": len(results),
            }
        else:
            tier_metrics[tier] = {"mean_survival_ratio": 0.0, "pass_rate_percent": 0.0, "num_sets": 0}
    return tier_metrics


def detect_cluster_regression(
    prev_clusters: List[Dict[str, Any]],
    curr_clusters: List[Dict[str, Any]],
) -> List[Tuple[str, str, float]]:
    """FASE 2: Detects clusters that regressed (got worse). Returns (root_cause, sub_cause, likelihood)."""
    prev_map = {(c["root_cause_category"], c["sub_cause"]): c for c in prev_clusters}
    regressions = []
    
    for curr in curr_clusters:
        key = (curr["root_cause_category"], curr["sub_cause"])
        if key in prev_map:
            prev_count = prev_map[key].get("affected_mocs_count", 0)
            curr_count = curr.get("affected_mocs_count", 0)
            if curr_count > prev_count * 1.5:  # >50% growth = regression
                likelihood = min(1.0, (curr_count - prev_count) / max(1, prev_count))
                regressions.append((key[0], key[1], likelihood))
    
    return sorted(regressions, key=lambda x: x[2], reverse=True)


def apply_bundle_ttl(bundle_dict: Dict[str, Any], current_iteration: int, creation_iteration: int, ttl=3) -> Dict[str, Any]:
    """FASE 2: Applies TTL to bundle rules. Expires old custom overrides."""
    if current_iteration - creation_iteration > ttl:
        bundle_dict["snot_and_hinge_parts"] = []
        bundle_dict["fractional_offset_parts"] = []
        bundle_dict["socket_coverage_parts"] = []
        bundle_dict["multi_box_hollow_parts"] = []
        bundle_dict["missing_port_parts"] = []
        bundle_dict["families_with_3d_rotations"] = []
        bundle_dict["roi_margin_overrides"] = {}
        bundle_dict["part_clearance_overrides"] = {}
    return bundle_dict

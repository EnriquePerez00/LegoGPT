"""
FASE 2 TEST: Scale Hygiene (Tiers, TTL, Regressions)
"""

import pytest
from src.phase2_scale_hygiene import SetTier, compute_tier_metrics, detect_cluster_regression, apply_bundle_ttl


class TestPhase2ScaleHygiene:
    """Test suite for FASE 2: scale hygiene."""

    def test_set_tier_classification(self):
        """Test classification of sets by part count."""
        assert SetTier.classify(50) == "SMALL"
        assert SetTier.classify(200) == "MEDIUM"
        assert SetTier.classify(500) == "LARGE"

    def test_compute_tier_metrics(self):
        """Test aggregation of KPIs by tier."""
        moc_results = [
            {
                "total_steps": 50,
                "completed_all_steps": True,
                "survival_ratio": 1.0,
            },
            {
                "total_steps": 150,
                "completed_all_steps": False,
                "survival_ratio": 0.5,
            },
            {
                "total_steps": 400,
                "completed_all_steps": False,
                "survival_ratio": 0.3,
            },
        ]
        
        tier_metrics = compute_tier_metrics(moc_results)
        
        assert "SMALL" in tier_metrics
        assert "MEDIUM" in tier_metrics
        assert "LARGE" in tier_metrics
        assert tier_metrics["SMALL"]["num_sets"] == 1
        assert tier_metrics["MEDIUM"]["num_sets"] == 1
        assert tier_metrics["LARGE"]["num_sets"] == 1

    def test_detect_cluster_regression(self):
        """Test detection of regressed clusters."""
        prev_clusters = [
            {"root_cause_category": "HYPOTHESIS_NOT_GENERATED", "sub_cause": "FRACTIONAL_OFFSET_GRID", "affected_mocs_count": 2},
        ]
        
        curr_clusters = [
            {"root_cause_category": "HYPOTHESIS_NOT_GENERATED", "sub_cause": "FRACTIONAL_OFFSET_GRID", "affected_mocs_count": 5},
        ]
        
        regressions = detect_cluster_regression(prev_clusters, curr_clusters)
        
        assert len(regressions) == 1
        assert regressions[0][0] == "HYPOTHESIS_NOT_GENERATED"
        assert regressions[0][2] > 0  # regression likelihood

    def test_apply_bundle_ttl(self):
        """Test TTL expiration of bundle rules."""
        bundle = {
            "snot_and_hinge_parts": ["part1", "part2"],
            "roi_margin_overrides": {"3001.dat": 0.7},
        }
        
        # Within TTL
        bundle_fresh = apply_bundle_ttl(bundle.copy(), 5, 3, ttl=3)
        assert len(bundle_fresh["snot_and_hinge_parts"]) == 2
        
        # Expired
        bundle_old = apply_bundle_ttl(bundle.copy(), 7, 3, ttl=3)
        assert len(bundle_old["snot_and_hinge_parts"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

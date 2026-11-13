"""
FASE 1 TEST: Agent Survival Attribution
Validates agent attribution system and summary integration.
"""

import pytest

from src.oracle_step_debugger import (
    StepFailureIncident,
    PatternClusterAnalyzer,
    map_root_cause_to_responsible_agent,
)


class TestPhase1AgentAttribution:
    """Test suite for FASE 1: agent survival attribution."""

    def test_map_root_cause_to_responsible_agent(self):
        """Test that root causes are correctly mapped to agents."""
        test_cases = [
            ("INVENTORY_OCR_MISMATCH", "CALLOUT_PART_OR_QTY_MISMATCH", "lego-context-agent"),
            ("MISSING_CONNECTIVITY_PORT", "UNRECOGNIZED_PORT_COUPLING", "lego-identifier-agent"),
            ("FALSE_SOLID_COLLISION_PRUNING", "OBB_OVERLAP_DETECTED", "lego-guardian-agent"),
            ("HYPOTHESIS_NOT_GENERATED", "SUBSTRATE_PARENT_FILTERED", "lego-identifier-agent"),
            ("VISUAL_SCORER_RANKING_ERROR", "ROI_FILTER_PRUNED", "lego-placer-agent"),
        ]
        
        for root_cause, sub_cause, expected_agent in test_cases:
            agent = map_root_cause_to_responsible_agent(root_cause, sub_cause)
            assert agent == expected_agent, f"Failed: {root_cause}/{sub_cause}"

    def test_step_failure_incident_responsible_agent(self):
        """Test that StepFailureIncident.responsible_agent is properly set."""
        inc = StepFailureIncident(
            moc_id="test_moc_1",
            step_index=3,
            total_gt_steps=10,
            steps_survived_before_error=2,
            survival_ratio=0.2,
            root_cause_category="MISSING_CONNECTIVITY_PORT",
            sub_cause="UNRECOGNIZED_PORT_COUPLING",
            child_part_id="3001.dat",
            child_color_id=16,
            responsible_agent="lego-identifier-agent",
        )
        assert inc.responsible_agent == "lego-identifier-agent"

    def test_compute_agent_survival_attribution(self):
        """Test aggregation of survival loss by agent."""
        incidents = [
            StepFailureIncident(
                moc_id=f"moc_{i}",
                step_index=1+i,
                total_gt_steps=10,
                steps_survived_before_error=1,
                survival_ratio=0.03,
                root_cause_category="HYPOTHESIS_NOT_GENERATED",
                sub_cause="FRACTIONAL_OFFSET_GRID",
                child_part_id="3001.dat",
                child_color_id=16,
                responsible_agent="lego-identifier-agent",
            )
            for i in range(5)
        ]
        
        analyzer = PatternClusterAnalyzer()
        attribution = analyzer._compute_agent_survival_attribution(incidents)
        
        assert "lego-identifier-agent" in attribution
        assert attribution["lego-identifier-agent"]["incidents"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

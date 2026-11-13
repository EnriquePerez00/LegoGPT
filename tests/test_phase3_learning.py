"""
FASE 3 TEST: Temporal Beam, Parameter Search, LLM Shadow Eval
"""

import pytest
import numpy as np
from src.phase3_learning import TemporalBeamSearchScheduler, ParameterSearchAgent, LLMShadowEvaluator


class TestPhase3Learning:
    """Test suite for FASE 3: advanced learning."""

    def test_temporal_beam_search_tracking(self):
        """Test beam search state tracking."""
        scheduler = TemporalBeamSearchScheduler(beam_width=3, lookback_window=2)
        
        nodes = [{"id": 0}]
        scheduler.track_step_state(nodes, 1.0)
        scheduler.track_step_state(nodes, 0.9)
        scheduler.track_step_state(nodes, 0.8)
        
        assert len(scheduler.step_states) == 3
        assert scheduler.can_backtrack(3)

    def test_temporal_beam_get_candidates(self):
        """Test retrieval of backtrack candidates."""
        scheduler = TemporalBeamSearchScheduler(beam_width=2, lookback_window=3)
        
        for i in range(5):
            scheduler.track_step_state([{"id": i}], 1.0 - i*0.1)
        
        candidates = scheduler.get_backtrack_candidates(4)
        assert len(candidates) <= 2
        # Should be sorted by probability (descending)
        for i in range(len(candidates)-1):
            assert candidates[i][1] >= candidates[i+1][1]

    def test_parameter_search_propose_perturbation(self):
        """Test parameter perturbation proposal."""
        agent = ParameterSearchAgent()
        current_params = {"w_edge_score": 0.6, "w_stud_score": 0.4}
        
        proposed = agent.propose_perturbation(current_params, iteration=0)
        
        assert "w_edge_score" in proposed
        assert "w_stud_score" in proposed
        # Should stay within bounds
        assert 0.4 <= proposed["w_edge_score"] <= 0.8
        assert 0.2 <= proposed["w_stud_score"] <= 0.6

    def test_parameter_search_best_params(self):
        """Test retrieval of best parameters."""
        agent = ParameterSearchAgent()
        
        agent.record_trial({"w_edge_score": 0.5}, reward=0.5)
        agent.record_trial({"w_edge_score": 0.6}, reward=0.8)
        agent.record_trial({"w_edge_score": 0.55}, reward=0.7)
        
        best = agent.get_best_params()
        assert best is not None
        assert best["w_edge_score"] == 0.6

    def test_llm_shadow_evaluator_pass(self):
        """Test LLM patch validation (passing case)."""
        proposed_patch = {"w_edge_score": 0.65}
        prior_kpis = {"mean_survival_ratio": 0.5}
        shadow_results = {"mean_survival_ratio": 0.55}
        
        is_valid, reason = LLMShadowEvaluator.shadow_evaluate_llm_patch(
            proposed_patch, prior_kpis, shadow_results
        )
        
        assert is_valid

    def test_llm_shadow_evaluator_fail(self):
        """Test LLM patch validation (failing case due to regression)."""
        proposed_patch = {"w_edge_score": 0.3}
        prior_kpis = {"mean_survival_ratio": 0.5}
        shadow_results = {"mean_survival_ratio": 0.4}  # 20% regression
        
        is_valid, reason = LLMShadowEvaluator.shadow_evaluate_llm_patch(
            proposed_patch, prior_kpis, shadow_results
        )
        
        assert not is_valid
        assert "regressed" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

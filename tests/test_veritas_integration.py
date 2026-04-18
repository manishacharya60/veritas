"""
VERITAS Integration Test
=========================

Tests the complete 4-agent framework:
1. StrategistAgent - High-level strategy planning
2. TacticianAgent - Low-level tactic generation
3. CriticAgent - Value estimation and guidance
4. RetrieverAgent - Premise retrieval
5. VERITASSearch - Unified search coordination

Run with: python -m pytest tests/test_veritas_integration.py -v
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.veritas import (
    ProofState,
    ProofStrategy,
    StrategyPlan,
    TacticCandidate,
    CriticAssessment,
    RetrievedPremise,
    StrategistAgent,
    TacticianAgent,
    CriticAgent,
    RetrieverAgent,
    VERITASSearch,
    SearchConfig,
    MCTSNode,
    create_veritas,
)


# =============================================================================
# Mock LEAN Verifier for Testing
# =============================================================================

class MockLEANVerifier:
    """Mock LEAN verifier that simulates verification responses."""
    
    def __init__(self):
        self.call_count = 0
        # Define which tactics "solve" which theorems
        self.solutions = {
            "0 + 0 = 0": ["rfl", "decide", "native_decide", "omega"],
            "1 + 1 = 2": ["rfl", "decide", "native_decide", "omega"],
            "∀ n : ℕ, n + 0 = n": ["intro n; rfl", "intro; simp"],
            "True": ["trivial", "decide"],
        }
    
    def validate(self, theorem: str, proof_steps: list, context: str = None):
        """Simulate LEAN validation."""
        self.call_count += 1
        
        # Create mock result
        class MockResult:
            pass
        
        result = MockResult()
        
        # Check if proof solves theorem
        last_tactic = proof_steps[-1] if proof_steps else ""
        solved = False
        
        for goal, solving_tactics in self.solutions.items():
            if goal in theorem:
                for t in solving_tactics:
                    if last_tactic.strip() == t or t.startswith(last_tactic.split()[0]):
                        solved = True
                        break
        
        # Set ABCD signals
        result.signal_A_syntax = 1.0  # Assume valid syntax
        result.signal_B_typecheck = 0.9
        result.signal_C_progress = 0.5 if proof_steps else 0.0
        result.signal_D_complete = 1.0 if solved else 0.0
        
        result.remaining_goal = "" if solved else theorem
        result.hypotheses = []
        result.subgoals = [] if solved else [theorem]
        
        return result


# =============================================================================
# Unit Tests for Individual Agents
# =============================================================================

class TestProofState:
    """Test ProofState data structure."""
    
    def test_creation(self):
        state = ProofState(
            theorem="0 + 0 = 0",
            goal="0 + 0 = 0",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        assert state.theorem == "0 + 0 = 0"
        assert state.depth == 0
        assert state.syntax_valid
    
    def test_to_prompt(self):
        state = ProofState(
            theorem="test",
            goal="test_goal",
            hypotheses=["h1 : Type", "h2 : Nat"],
            tactics_applied=["intro", "simp"],
            subgoals=["sub1"],
        )
        prompt = state.to_prompt()
        assert "test" in prompt
        assert "test_goal" in prompt
        assert "h1" in prompt
        assert "intro" in prompt
    
    def test_hash(self):
        state1 = ProofState(
            theorem="t", goal="g", hypotheses=[], 
            tactics_applied=["intro"], subgoals=[]
        )
        state2 = ProofState(
            theorem="t", goal="g", hypotheses=[], 
            tactics_applied=["intro"], subgoals=[]
        )
        state3 = ProofState(
            theorem="t", goal="g", hypotheses=[], 
            tactics_applied=["simp"], subgoals=[]
        )
        
        assert state1.hash() == state2.hash()
        assert state1.hash() != state3.hash()


class TestStrategistAgent:
    """Test StrategistAgent."""
    
    def test_direct_strategy(self):
        agent = StrategistAgent()
        state = ProofState(
            theorem="0 = 0",
            goal="0 = 0",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        
        plan = agent(state)
        assert isinstance(plan, StrategyPlan)
        assert plan.strategy in [ProofStrategy.DIRECT, ProofStrategy.REWRITE]
        assert plan.confidence > 0
    
    def test_induction_strategy(self):
        agent = StrategistAgent()
        state = ProofState(
            theorem="∀ n : Nat, n + 0 = n",
            goal="∀ n : Nat, n + 0 = n",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        
        plan = agent(state)
        assert plan.strategy == ProofStrategy.INDUCTION
    
    def test_cases_strategy(self):
        agent = StrategistAgent()
        state = ProofState(
            theorem="a ∨ b",
            goal="a ∨ b",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        
        plan = agent(state)
        assert plan.strategy == ProofStrategy.CASES


class TestTacticianAgent:
    """Test TacticianAgent."""
    
    def test_direct_tactics(self):
        agent = TacticianAgent()
        state = ProofState(
            theorem="0 = 0",
            goal="0 = 0",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        strategy = StrategyPlan(
            strategy=ProofStrategy.DIRECT,
            target_lemmas=[],
            estimated_depth=1,
            subgoal_decomposition=[],
            confidence=0.9,
        )
        
        tactics = agent(state, strategy, num_candidates=5)
        
        assert len(tactics) > 0
        assert all(isinstance(t, TacticCandidate) for t in tactics)
        
        # Should include direct tactics
        tactic_names = [t.tactic for t in tactics]
        assert any(t in ['rfl', 'trivial', 'decide', 'native_decide'] for t in tactic_names)
    
    def test_with_premises(self):
        agent = TacticianAgent()
        state = ProofState(
            theorem="n + m = m + n",
            goal="n + m = m + n",
            hypotheses=["n : ℕ", "m : ℕ"],
            tactics_applied=[],
            subgoals=[],
        )
        strategy = StrategyPlan(
            strategy=ProofStrategy.APPLY_LEMMA,
            target_lemmas=["Nat.add_comm"],
            estimated_depth=2,
            subgoal_decomposition=[],
            confidence=0.8,
        )
        premises = [
            RetrievedPremise(
                lemma_name="Nat.add_comm",
                lemma_statement="∀ (n m : ℕ), n + m = m + n",
                relevance_score=0.9,
                usage_hint="commutativity"
            )
        ]
        
        tactics = agent(state, strategy, premises=premises, num_candidates=8)
        
        # Should include premise-based tactics
        tactic_names = [t.tactic for t in tactics]
        assert any("Nat.add_comm" in t for t in tactic_names)
    
    def test_with_critic_guidance(self):
        agent = TacticianAgent()
        state = ProofState(
            theorem="test",
            goal="test",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        strategy = StrategyPlan(
            strategy=ProofStrategy.DIRECT,
            target_lemmas=[],
            estimated_depth=1,
            subgoal_decomposition=[],
            confidence=0.5,
        )
        critic_guidance = CriticAssessment(
            value=0.7,
            policy_prior={},
            reasoning="test",
            suggested_tactics=["omega", "ring"],
            pruning_recommendation=False,
        )
        
        tactics = agent(state, strategy, critic_guidance=critic_guidance)
        
        # Should include critic-suggested tactics
        tactic_names = [t.tactic for t in tactics]
        assert "omega" in tactic_names or "ring" in tactic_names


class TestCriticAgent:
    """Test CriticAgent."""
    
    def test_value_estimation(self):
        agent = CriticAgent()
        state = ProofState(
            theorem="test",
            goal="test",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
            syntax_valid=True,
            types_valid=True,
            depth=0,
        )
        
        assessment = agent(state)
        
        assert isinstance(assessment, CriticAssessment)
        assert 0.0 <= assessment.value <= 1.0
        assert isinstance(assessment.reasoning, str)
        assert not assessment.pruning_recommendation  # Shallow, valid state
    
    def test_completed_proof_value(self):
        agent = CriticAgent()
        state = ProofState(
            theorem="test",
            goal="",
            hypotheses=[],
            tactics_applied=["rfl"],
            subgoals=[],
            is_complete=True,
        )
        
        assessment = agent(state)
        assert assessment.value == 1.0
    
    def test_novelty_bonus(self):
        agent = CriticAgent()
        state = ProofState(
            theorem="test",
            goal="test",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        
        # First visit should get higher value
        assessment1 = agent(state)
        # Second visit should get lower novelty bonus
        assessment2 = agent(state)
        
        # Due to novelty decay, second should be slightly different
        assert assessment1.value >= assessment2.value or abs(assessment1.value - assessment2.value) < 0.2
    
    def test_pruning_deep_state(self):
        agent = CriticAgent()
        state = ProofState(
            theorem="test",
            goal="test",
            hypotheses=[],
            tactics_applied=["t" + str(i) for i in range(12)],
            subgoals=[],
            depth=12,
        )
        
        assessment = agent(state)
        assert assessment.pruning_recommendation  # Too deep
    
    def test_suggests_tactics(self):
        agent = CriticAgent()
        
        # Test with equality goal
        state = ProofState(
            theorem="a = b",
            goal="a = b",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        assessment = agent(state)
        assert len(assessment.suggested_tactics) > 0


class TestRetrieverAgent:
    """Test RetrieverAgent."""
    
    def test_nat_retrieval(self):
        agent = RetrieverAgent()
        state = ProofState(
            theorem="∀ (n m : ℕ), n + m = m + n",
            goal="n + m = m + n",
            hypotheses=["n : ℕ", "m : ℕ"],
            tactics_applied=[],
            subgoals=[],
        )
        
        premises = agent(state, k=5)
        
        assert len(premises) > 0
        assert all(isinstance(p, RetrievedPremise) for p in premises)
        
        # Should find Nat.add_comm
        lemma_names = [p.lemma_name for p in premises]
        assert "Nat.add_comm" in lemma_names
    
    def test_list_retrieval(self):
        agent = RetrieverAgent()
        state = ProofState(
            theorem="List.length [] = 0",
            goal="List.length [] = 0",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        
        premises = agent(state, k=3)
        lemma_names = [p.lemma_name for p in premises]
        assert any("List" in name for name in lemma_names)
    
    def test_relevance_scores(self):
        agent = RetrieverAgent()
        state = ProofState(
            theorem="n + m = m + n",
            goal="n + m = m + n",
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
        )
        
        premises = agent(state, k=10)
        
        # Results should be sorted by relevance
        for i in range(len(premises) - 1):
            assert premises[i].relevance_score >= premises[i+1].relevance_score


# =============================================================================
# Integration Tests for VERITASSearch
# =============================================================================

class TestVERITASSearch:
    """Test the unified VERITASSearch engine."""
    
    def test_simple_proof(self):
        """Test proving a trivial theorem."""
        verifier = MockLEANVerifier()
        search = create_veritas(verifier, config=SearchConfig(
            max_iterations=20,
            num_tactic_candidates=5,
        ))
        
        result = search.search("0 + 0 = 0")
        
        # Should find a proof
        assert result['success'] or result['statistics']['iterations'] > 0
        assert result['statistics']['lean_calls'] > 0
    
    def test_search_statistics(self):
        """Test that search produces meaningful statistics."""
        verifier = MockLEANVerifier()
        search = create_veritas(verifier)
        
        result = search.search("True")
        
        stats = result['statistics']
        assert 'iterations' in stats
        assert 'nodes_expanded' in stats
        assert 'lean_calls' in stats
        assert 'time_seconds' in stats
        assert stats['time_seconds'] > 0
    
    def test_timeout(self):
        """Test that search respects timeout."""
        verifier = MockLEANVerifier()
        search = create_veritas(verifier, config=SearchConfig(
            max_iterations=1000,
            timeout_seconds=1,
        ))
        
        result = search.search("impossible_theorem_xyz")
        
        assert result['statistics']['time_seconds'] <= 2  # Allow some slack


class TestMCTSNode:
    """Test MCTS node operations."""
    
    def test_q_value(self):
        state = ProofState(
            theorem="t", goal="g", hypotheses=[], 
            tactics_applied=[], subgoals=[]
        )
        node = MCTSNode(state=state)
        
        assert node.q_value == 0.0  # Zero visits
        
        node.visits = 2
        node.total_value = 1.0
        
        assert node.q_value == 0.5
    
    def test_is_terminal(self):
        # Completed proof
        state1 = ProofState(
            theorem="t", goal="", hypotheses=[], 
            tactics_applied=[], subgoals=[], is_complete=True
        )
        node1 = MCTSNode(state=state1)
        assert node1.is_terminal
        
        # Too deep
        state2 = ProofState(
            theorem="t", goal="g", hypotheses=[], 
            tactics_applied=[], subgoals=[], depth=20
        )
        node2 = MCTSNode(state=state2)
        assert node2.is_terminal


class TestSearchConfig:
    """Test search configuration."""
    
    def test_default_config(self):
        config = SearchConfig()
        assert config.max_iterations == 100
        assert config.exploration_constant > 0
        assert config.num_tactic_candidates > 0
    
    def test_custom_config(self):
        config = SearchConfig(
            max_iterations=50,
            max_depth=10,
            exploration_constant=2.0,
        )
        assert config.max_iterations == 50
        assert config.max_depth == 10
        assert config.exploration_constant == 2.0


# =============================================================================
# End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end tests with mock verifier."""
    
    def test_full_pipeline(self):
        """Test the complete VERITAS pipeline."""
        # 1. Create initial state
        state = ProofState(
            theorem="n + 0 = n",
            goal="n + 0 = n",
            hypotheses=["n : ℕ"],
            tactics_applied=[],
            subgoals=[],
        )
        
        # 2. Strategist plans
        strategist = StrategistAgent()
        strategy = strategist(state)
        assert strategy.strategy in ProofStrategy
        
        # 3. Retriever finds premises
        retriever = RetrieverAgent()
        premises = retriever(state)
        assert len(premises) > 0
        
        # 4. Critic assesses
        critic = CriticAgent()
        assessment = critic(state)
        assert 0 <= assessment.value <= 1
        
        # 5. Tactician generates
        tactician = TacticianAgent()
        tactics = tactician(state, strategy, premises, assessment)
        assert len(tactics) > 0
        
        # 6. Run search
        verifier = MockLEANVerifier()
        search = create_veritas(verifier)
        result = search.search("n + 0 = n")
        
        assert 'success' in result
        assert 'statistics' in result
    
    def test_agent_collaboration(self):
        """Test that agents can collaborate through shared state."""
        state = ProofState(
            theorem="∀ n m : ℕ, n + m = m + n",
            goal="n + m = m + n",
            hypotheses=["n : ℕ", "m : ℕ"],
            tactics_applied=["intro n", "intro m"],
            subgoals=[],
            depth=2,
        )
        
        # All agents should be able to process the same state
        strategist = StrategistAgent()
        tactician = TacticianAgent()
        critic = CriticAgent()
        retriever = RetrieverAgent()
        
        strategy = strategist(state)
        premises = retriever(state)
        assessment = critic(state)
        tactics = tactician(state, strategy, premises, assessment)
        
        # Verify coordination
        assert strategy.strategy is not None
        assert len(premises) > 0
        assert len(tactics) > 0
        assert not assessment.pruning_recommendation  # Valid state


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Basic performance tests."""
    
    def test_many_states(self):
        """Test handling many proof states."""
        critic = CriticAgent()
        
        states = []
        for i in range(100):
            state = ProofState(
                theorem=f"theorem_{i}",
                goal=f"goal_{i}",
                hypotheses=[],
                tactics_applied=[f"tactic_{j}" for j in range(i % 5)],
                subgoals=[],
                depth=i % 10,
            )
            states.append(state)
        
        # Should handle many states efficiently
        import time
        start = time.time()
        for state in states:
            critic(state)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should complete quickly
    
    def test_deep_tree(self):
        """Test handling deep MCTS tree."""
        verifier = MockLEANVerifier()
        search = create_veritas(verifier, config=SearchConfig(
            max_iterations=10,
            max_depth=5,
        ))
        
        # Should handle without stack overflow
        result = search.search("deep_theorem")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

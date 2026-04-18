"""
VERITAS Tests for Agents
"""

import pytest
from unittest.mock import Mock, patch

from src.agents.lean_generator import LEANGenerator, GeneratedProofStep
from src.agents.proof_validator import ProofValidator, ValidationResult, ValidationStatus
from src.agents.reflector import Reflector, ReflectionResult, ReflectionType


class TestLEANGenerator:
    """Tests for LEAN Generator agent."""
    
    def test_init(self):
        """Test generator initialization."""
        with patch.object(LEANGenerator, '_load_model'):
            generator = LEANGenerator(
                model_name="test-model",
                num_return_sequences=3,
            )
            
            assert generator.model_name == "test-model"
            assert generator.num_return_sequences == 3
    
    def test_build_initial_prompt(self):
        """Test initial prompt construction."""
        with patch.object(LEANGenerator, '_load_model'):
            generator = LEANGenerator()
            
            prompt = generator._build_initial_prompt(
                theorem="theorem test : 1 + 1 = 2",
                context="import Mathlib",
            )
            
            assert "theorem test" in prompt
            assert "import Mathlib" in prompt
    
    def test_build_subgoal_prompt(self):
        """Test subgoal prompt construction."""
        with patch.object(LEANGenerator, '_load_model'):
            generator = LEANGenerator()
            
            prompt = generator._build_subgoal_prompt(
                goal="n + 0 = n",
                proof_state={"hypotheses": ["n : ℕ"]},
                history=["intro n"],
            )
            
            assert "n + 0 = n" in prompt
            assert "intro n" in prompt
    
    def test_parse_output(self):
        """Test output parsing."""
        with patch.object(LEANGenerator, '_load_model'):
            generator = LEANGenerator()
            
            output = """```lean4
simp
```
Reasoning: Simplification should close this goal."""
            
            result = generator._parse_output(output)
            
            assert result.code == "simp"
            assert "Simplification" in result.reasoning


class TestProofValidator:
    """Tests for Proof Validator agent."""
    
    def test_validation_status_enum(self):
        """Test validation status values."""
        assert ValidationStatus.SUCCESS.value == "success"
        assert ValidationStatus.SYNTAX_ERROR.value == "syntax_error"
    
    def test_validation_result_signal(self):
        """Test validation result signal computation."""
        result = ValidationResult(
            status=ValidationStatus.SUCCESS,
            is_valid=True,
            remaining_goals=[],
            error_message=None,
            proof_state=None,
            signal_A_syntax=1.0,
            signal_B_typecheck=1.0,
            signal_C_progress=1.0,
            signal_D_complete=1.0,
        )
        
        assert result.total_signal == 1.0
    
    def test_partial_signals(self):
        """Test partial signal computation."""
        result = ValidationResult(
            status=ValidationStatus.TACTIC_FAILED,
            is_valid=False,
            remaining_goals=["goal"],
            error_message="tactic failed",
            proof_state=None,
            signal_A_syntax=1.0,
            signal_B_typecheck=1.0,
            signal_C_progress=0.2,
            signal_D_complete=0.0,
        )
        
        expected = 0.25 * (1.0 + 1.0 + 0.2 + 0.0)
        assert abs(result.total_signal - expected) < 0.001


class TestReflector:
    """Tests for Reflector agent."""
    
    def test_reflection_types(self):
        """Test reflection type enum."""
        assert ReflectionType.SUCCESS.value == "success"
        assert ReflectionType.BACKTRACK.value == "backtrack"
    
    def test_should_prune_stuck(self):
        """Test pruning decision for stuck nodes."""
        with patch.object(Reflector, '_load_model'):
            reflector = Reflector()
            
            # Create stuck reflection history
            stuck_reflections = [
                ReflectionResult(
                    reflection_type=ReflectionType.STUCK,
                    analysis="stuck",
                    suggestions=[],
                    confidence=0.5,
                    should_backtrack=True,
                    priority_tactics=[],
                    reflection_signal=0.2,
                )
                for _ in range(3)
            ]
            
            should_prune = reflector.should_prune(
                node_value=0.1,
                visit_count=5,
                reflection_history=stuck_reflections,
            )
            
            assert should_prune is True
    
    def test_should_not_prune_early(self):
        """Test that early nodes are not pruned."""
        with patch.object(Reflector, '_load_model'):
            reflector = Reflector()
            
            should_prune = reflector.should_prune(
                node_value=0.5,
                visit_count=2,
                reflection_history=[],
            )
            
            assert should_prune is False
    
    def test_get_alternative_tactics(self):
        """Test alternative tactic suggestions."""
        with patch.object(Reflector, '_load_model'):
            reflector = Reflector()
            
            alternatives = reflector._get_alternative_tactics("simp", "goal")
            
            assert "simp" not in alternatives
            assert len(alternatives) > 0
    
    def test_get_priority_tactics_forall(self):
        """Test priority tactics for universal goals."""
        with patch.object(Reflector, '_load_model'):
            reflector = Reflector()
            
            tactics = reflector._get_priority_tactics("∀ n, P n")
            
            assert "intro" in tactics or "intros" in tactics


class TestIntegration:
    """Integration tests for agent coordination."""
    
    def test_generator_validator_flow(self):
        """Test generator -> validator flow."""
        # Mock the flow
        generated = GeneratedProofStep(
            code="simp",
            tactic="simp",
            confidence=0.8,
            reasoning="test",
        )
        
        validation = ValidationResult(
            status=ValidationStatus.SUCCESS,
            is_valid=True,
            remaining_goals=[],
            error_message=None,
            proof_state=None,
            signal_A_syntax=1.0,
            signal_B_typecheck=1.0,
            signal_C_progress=1.0,
            signal_D_complete=1.0,
        )
        
        # Verify the data structures are compatible
        assert generated.code == "simp"
        assert validation.total_signal == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

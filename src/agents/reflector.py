"""
Reflector Agent

Analyzes generator output and validator feedback to guide MCTS search.
Combines LLM reasoning with symbolic information from LEAN.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.agents.lean_generator import GeneratedProofStep
from src.agents.proof_validator import ValidationResult, ValidationStatus


class ReflectionType(Enum):
    """Type of reflection feedback."""
    SUCCESS = "success"
    SYNTAX_FIX = "syntax_fix"
    STRATEGY_CHANGE = "strategy_change"
    GOAL_DECOMPOSITION = "goal_decomposition"
    BACKTRACK = "backtrack"
    STUCK = "stuck"


@dataclass
class ReflectionResult:
    """Result of reflection analysis."""
    reflection_type: ReflectionType
    analysis: str
    suggestions: List[str]
    confidence: float
    should_backtrack: bool
    priority_tactics: List[str]
    
    # Signal contribution for MCTS
    reflection_signal: float


class Reflector:
    """
    Hybrid reflection agent combining LLM analysis with LEAN feedback.
    
    Responsibilities:
    - Analyze why a proof step failed
    - Suggest corrections or alternative approaches
    - Recommend when to backtrack in MCTS
    - Provide structured guidance for next expansion
    """
    
    REFLECTION_PROMPT = """You are a mathematical reasoning expert analyzing a proof attempt.

Given:
1. The current goal
2. The attempted proof step
3. The validation result from LEAN
4. The proof history

Analyze what went wrong (if anything) and suggest improvements.

Consider:
- Is the approach fundamentally sound?
- Is there a simpler tactic that would work?
- Should we decompose the goal differently?
- Is backtracking necessary?

Provide structured feedback:
- Analysis: Brief explanation of the situation
- Suggestion: Concrete next step to try
- Confidence: How confident are you (0-1)?
- Priority tactics: Which tactics are most promising?
"""

    def __init__(
        self,
        model_name: str = "llemma-7b",
        device: str = "cuda",
        max_reflection_rounds: int = 3,
    ):
        """
        Initialize the Reflector.
        
        Args:
            model_name: LLM for reflection
            device: Compute device
            max_reflection_rounds: Maximum reflection iterations
        """
        self.model_name = model_name
        self.device = device
        self.max_reflection_rounds = max_reflection_rounds
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the reflection model."""
        # Placeholder for model loading
        pass
    
    def reflect(
        self,
        goal: str,
        generated_step: GeneratedProofStep,
        validation_result: ValidationResult,
        proof_history: List[str],
        attempt_count: int = 1,
    ) -> ReflectionResult:
        """
        Reflect on a proof attempt.
        
        Args:
            goal: The current proof goal
            generated_step: The generated proof step
            validation_result: Result from the validator
            proof_history: Previous proof steps
            attempt_count: Number of attempts on this goal
            
        Returns:
            ReflectionResult with analysis and suggestions
        """
        # Quick path for successful validation
        if validation_result.is_valid:
            return self._success_reflection(generated_step)
        
        # Analyze based on error type
        if validation_result.status == ValidationStatus.SYNTAX_ERROR:
            return self._syntax_error_reflection(
                goal, generated_step, validation_result
            )
        
        elif validation_result.status == ValidationStatus.TYPE_ERROR:
            return self._type_error_reflection(
                goal, generated_step, validation_result
            )
        
        elif validation_result.status == ValidationStatus.TACTIC_FAILED:
            return self._tactic_failure_reflection(
                goal, generated_step, validation_result, proof_history
            )
        
        elif validation_result.status == ValidationStatus.TIMEOUT:
            return self._timeout_reflection(goal, generated_step)
        
        else:
            return self._general_reflection(
                goal, generated_step, validation_result, proof_history, attempt_count
            )
    
    def should_prune(
        self,
        node_value: float,
        visit_count: int,
        reflection_history: List[ReflectionResult],
    ) -> bool:
        """
        Determine if an MCTS node should be pruned.
        
        Args:
            node_value: Current value estimate of the node
            visit_count: Number of times node was visited
            reflection_history: Past reflections for this path
            
        Returns:
            True if node should be pruned
        """
        # Prune if consistently stuck
        if len(reflection_history) >= 3:
            stuck_count = sum(
                1 for r in reflection_history[-3:]
                if r.reflection_type == ReflectionType.STUCK
            )
            if stuck_count >= 2:
                return True
        
        # Prune if value too low after sufficient visits
        if visit_count >= 5 and node_value < 0.1:
            return True
        
        # Prune if backtracking strongly recommended
        if reflection_history and reflection_history[-1].should_backtrack:
            if reflection_history[-1].confidence > 0.8:
                return True
        
        return False
    
    def compute_exploration_bonus(
        self,
        reflection: ReflectionResult,
        current_goal: str,
    ) -> float:
        """
        Compute exploration bonus for MCTS based on reflection.
        
        Args:
            reflection: The reflection result
            current_goal: Current proof goal
            
        Returns:
            Bonus value to add to UCB calculation
        """
        base_bonus = 0.0
        
        # Bonus for strategy change suggestions
        if reflection.reflection_type == ReflectionType.STRATEGY_CHANGE:
            base_bonus += 0.2
        
        # Bonus for goal decomposition (often leads to progress)
        if reflection.reflection_type == ReflectionType.GOAL_DECOMPOSITION:
            base_bonus += 0.3
        
        # Scale by confidence
        return base_bonus * reflection.confidence
    
    def _success_reflection(
        self,
        generated_step: GeneratedProofStep,
    ) -> ReflectionResult:
        """Create reflection for successful validation."""
        return ReflectionResult(
            reflection_type=ReflectionType.SUCCESS,
            analysis="Proof step validated successfully.",
            suggestions=["Continue with the next goal if any remain."],
            confidence=1.0,
            should_backtrack=False,
            priority_tactics=[],
            reflection_signal=1.0,
        )
    
    def _syntax_error_reflection(
        self,
        goal: str,
        step: GeneratedProofStep,
        result: ValidationResult,
    ) -> ReflectionResult:
        """Reflect on syntax errors."""
        # Extract specific syntax issue
        error_msg = result.error_message or "Unknown syntax error"
        
        suggestions = [
            "Check for typos in tactic name",
            "Verify parentheses and brackets match",
            "Ensure proper LEAN 4 syntax (not LEAN 3)",
        ]
        
        # LLM-based fix suggestion would go here
        
        return ReflectionResult(
            reflection_type=ReflectionType.SYNTAX_FIX,
            analysis=f"Syntax error in generated code: {error_msg}",
            suggestions=suggestions,
            confidence=0.7,
            should_backtrack=False,
            priority_tactics=["simp", "rfl", "decide"],
            reflection_signal=0.2,
        )
    
    def _type_error_reflection(
        self,
        goal: str,
        step: GeneratedProofStep,
        result: ValidationResult,
    ) -> ReflectionResult:
        """Reflect on type errors."""
        error_msg = result.error_message or "Type mismatch"
        
        suggestions = [
            "Check that hypothesis types match goal",
            "Consider using type conversion tactics",
            "Try 'exact?' to find matching terms",
        ]
        
        return ReflectionResult(
            reflection_type=ReflectionType.STRATEGY_CHANGE,
            analysis=f"Type error: {error_msg}. The tactic may be applied incorrectly.",
            suggestions=suggestions,
            confidence=0.6,
            should_backtrack=False,
            priority_tactics=["exact?", "apply?", "convert"],
            reflection_signal=0.3,
        )
    
    def _tactic_failure_reflection(
        self,
        goal: str,
        step: GeneratedProofStep,
        result: ValidationResult,
        history: List[str],
    ) -> ReflectionResult:
        """Reflect on tactic failures."""
        # Analyze if we should try different tactics or backtrack
        tactic_used = step.tactic or "unknown"
        
        # Suggest alternative tactics based on what was tried
        alternatives = self._get_alternative_tactics(tactic_used, goal)
        
        should_backtrack = len(history) > 5 and not any(
            t in step.code for t in ["have", "suffices", "obtain"]
        )
        
        return ReflectionResult(
            reflection_type=ReflectionType.STRATEGY_CHANGE,
            analysis=f"Tactic '{tactic_used}' failed. Goal may need different approach.",
            suggestions=[f"Try {t} instead" for t in alternatives[:3]],
            confidence=0.5,
            should_backtrack=should_backtrack,
            priority_tactics=alternatives,
            reflection_signal=0.4,
        )
    
    def _timeout_reflection(
        self,
        goal: str,
        step: GeneratedProofStep,
    ) -> ReflectionResult:
        """Reflect on timeout issues."""
        return ReflectionResult(
            reflection_type=ReflectionType.STRATEGY_CHANGE,
            analysis="Tactic timed out. This often indicates the approach is too complex.",
            suggestions=[
                "Try breaking the goal into smaller steps",
                "Use simpler tactics",
                "Add intermediate lemmas",
            ],
            confidence=0.6,
            should_backtrack=True,
            priority_tactics=["have", "suffices", "calc"],
            reflection_signal=0.1,
        )
    
    def _general_reflection(
        self,
        goal: str,
        step: GeneratedProofStep,
        result: ValidationResult,
        history: List[str],
        attempt_count: int,
    ) -> ReflectionResult:
        """General reflection using LLM."""
        # Build prompt for LLM reflection
        prompt = self._build_reflection_prompt(
            goal, step, result, history, attempt_count
        )
        
        # For now, return heuristic-based reflection
        # LLM call would go here
        
        if attempt_count >= 3:
            return ReflectionResult(
                reflection_type=ReflectionType.STUCK,
                analysis="Multiple attempts failed. Consider fundamental approach change.",
                suggestions=["Try goal decomposition", "Look for auxiliary lemmas"],
                confidence=0.4,
                should_backtrack=True,
                priority_tactics=["have", "obtain", "cases"],
                reflection_signal=0.2,
            )
        
        return ReflectionResult(
            reflection_type=ReflectionType.STRATEGY_CHANGE,
            analysis="Proof step unsuccessful. Trying alternative approaches.",
            suggestions=["Try different tactic", "Decompose the goal"],
            confidence=0.5,
            should_backtrack=False,
            priority_tactics=self._get_priority_tactics(goal),
            reflection_signal=0.3,
        )
    
    def _build_reflection_prompt(
        self,
        goal: str,
        step: GeneratedProofStep,
        result: ValidationResult,
        history: List[str],
        attempt_count: int,
    ) -> str:
        """Build prompt for LLM reflection."""
        prompt = f"{self.REFLECTION_PROMPT}\n\n"
        prompt += f"Current goal: {goal}\n"
        prompt += f"Attempted step: {step.code}\n"
        prompt += f"Validation status: {result.status.value}\n"
        
        if result.error_message:
            prompt += f"Error: {result.error_message}\n"
        
        if history:
            prompt += f"Proof history: {history}\n"
        
        prompt += f"Attempt #{attempt_count}\n"
        
        return prompt
    
    def _get_alternative_tactics(
        self,
        failed_tactic: str,
        goal: str,
    ) -> List[str]:
        """Get alternative tactics based on what failed."""
        # Tactic categories
        simplification = ["simp", "ring", "linarith", "norm_num"]
        automation = ["aesop", "decide", "trivial"]
        structural = ["cases", "induction", "rcases"]
        construction = ["exact", "apply", "refine"]
        decomposition = ["constructor", "ext", "funext"]
        
        alternatives = []
        
        if failed_tactic in simplification:
            alternatives = automation + construction
        elif failed_tactic in automation:
            alternatives = simplification + structural
        elif failed_tactic in structural:
            alternatives = construction + decomposition
        else:
            alternatives = simplification + automation
        
        # Remove the failed tactic
        alternatives = [t for t in alternatives if t != failed_tactic]
        
        return alternatives
    
    def _get_priority_tactics(self, goal: str) -> List[str]:
        """Get priority tactics based on goal structure."""
        goal_lower = goal.lower()
        
        if "∀" in goal or "forall" in goal_lower:
            return ["intro", "intros", "ext"]
        elif "∃" in goal or "exists" in goal_lower:
            return ["use", "refine ⟨_, _⟩", "constructor"]
        elif "∧" in goal or "and" in goal_lower:
            return ["constructor", "And.intro"]
        elif "∨" in goal or "or" in goal_lower:
            return ["left", "right"]
        elif "=" in goal:
            return ["rfl", "ring", "simp", "linarith"]
        elif "≤" in goal or "<" in goal:
            return ["linarith", "norm_num", "omega"]
        else:
            return ["simp", "aesop", "decide"]

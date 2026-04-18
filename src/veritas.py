"""
VERITAS: Verification-Enhanced Reasoning with Integrated Tactic Agents and Search
===================================================================================

A novel multi-agent framework for neural theorem proving that combines:
1. Hierarchical proof search (strategy → tactics)
2. Critic-guided MCTS with learned value estimation
3. Verification-aware backpropagation
4. Retrieval-augmented premise selection

Key Innovation: Unlike prior work that treats LLM generation and tree search as 
separate components, VERITAS uses specialized agents that collaborate through
a shared proof state representation, with the Critic agent providing value 
estimates that guide MCTS exploration.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VERITAS Framework                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐ │
│  │  Strategist │   │  Tactician  │   │   Critic    │   │   Retriever     │ │
│  │   Agent     │──▶│   Agent     │──▶│   Agent     │◀──│   Agent         │ │
│  │ (High-level)│   │ (Low-level) │   │ (Value Est) │   │ (Premise RAG)   │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────────┘ │
│         │                 │                 │                   │          │
│         └─────────────────┴─────────────────┴───────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Critic-Guided MCTS with Verification Feedback          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │   │
│  │  │ Selection │─▶│ Expansion │─▶│Simulation │─▶│ Backpropagation │  │   │
│  │  │ (UCB+V)   │  │(Tactician)│  │  (LEAN)   │  │  (Structured)   │  │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LEAN 4 Verification Oracle                       │   │
│  │         Provides ground-truth signals for all agent decisions       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

References:
- DeepSeek-Prover V1.5: Intrinsic rewards, RLPAF
- AlphaProof: RL from verification, self-improvement
- ReProver: Retrieval-augmented premise selection
- PACT: Proof artifact signals
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time
import math
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class ProofState:
    """
    Unified proof state representation shared across all agents.
    
    This is a key design decision: all agents operate on the same state
    representation, enabling coherent multi-agent collaboration.
    """
    theorem: str                          # Original theorem statement
    goal: str                             # Current goal to prove
    hypotheses: List[str]                 # Available hypotheses
    tactics_applied: List[str]            # Proof history
    subgoals: List[str]                   # Remaining subgoals
    context: Optional[str] = None         # Additional LEAN context
    
    # Verification signals (from LEAN oracle)
    syntax_valid: bool = True
    types_valid: bool = True
    goal_progress: float = 0.0            # 0-1 progress toward goal
    is_complete: bool = False
    
    # Metadata for search
    depth: int = 0
    parent_tactic: Optional[str] = None
    
    def to_prompt(self) -> str:
        """Convert to prompt format for LLM agents."""
        return f"""
Theorem: {self.theorem}
Current Goal: {self.goal}
Hypotheses: {', '.join(self.hypotheses) if self.hypotheses else 'None'}
Proof so far: {' '.join(self.tactics_applied) if self.tactics_applied else 'None'}
Remaining subgoals: {len(self.subgoals)}
"""
    
    def hash(self) -> str:
        """Hash for state deduplication."""
        state_str = f"{self.goal}|{'|'.join(self.tactics_applied)}"
        return hashlib.md5(state_str.encode()).hexdigest()


class ProofStrategy(Enum):
    """High-level proof strategies identified by Strategist agent."""
    DIRECT = "direct"              # Direct computation (rfl, decide, norm_num)
    INDUCTION = "induction"        # Structural induction
    CONTRADICTION = "contradiction" # Proof by contradiction
    CASES = "cases"                # Case analysis
    REWRITE = "rewrite"            # Term rewriting
    APPLY_LEMMA = "apply_lemma"    # Apply library lemma
    CUSTOM = "custom"              # Domain-specific strategy


@dataclass
class StrategyPlan:
    """Output from Strategist agent."""
    strategy: ProofStrategy
    target_lemmas: List[str]       # Lemmas likely needed
    estimated_depth: int           # Expected proof depth
    subgoal_decomposition: List[str]  # How to break down the proof
    confidence: float              # Strategy confidence


@dataclass
class TacticCandidate:
    """Output from Tactician agent."""
    tactic: str                    # LEAN tactic code
    reasoning: str                 # Why this tactic
    prior: float                   # Prior probability
    strategy_alignment: float      # How well it aligns with strategy


@dataclass
class CriticAssessment:
    """Output from Critic agent."""
    value: float                   # Estimated proof success probability
    policy_prior: Dict[str, float] # Prior over tactics
    reasoning: str                 # Explanation
    suggested_tactics: List[str]   # Tactics to prioritize
    pruning_recommendation: bool   # Should this branch be pruned?


@dataclass
class RetrievedPremise:
    """Output from Retriever agent."""
    lemma_name: str
    lemma_statement: str
    relevance_score: float
    usage_hint: str                # How to apply this lemma


# =============================================================================
# Agent Interfaces
# =============================================================================

class BaseAgent(ABC):
    """Base class for all VERITAS agents."""
    
    @abstractmethod
    def __call__(self, state: ProofState, **kwargs) -> Any:
        pass


class StrategistAgent(BaseAgent):
    """
    High-level strategy planning agent.
    
    Unlike prior work that directly generates tactics, the Strategist first
    identifies the proof approach, guiding subsequent tactic generation.
    This hierarchical decomposition improves search efficiency.
    """
    
    STRATEGY_PROMPT = """
You are an expert mathematician planning a proof strategy.

{state}

Analyze this proof goal and determine:
1. What high-level strategy should be used? (direct/induction/contradiction/cases/rewrite/apply_lemma)
2. What lemmas or theorems might be needed?
3. How should the proof be decomposed into subgoals?
4. What is your confidence in this strategy?

Output format:
STRATEGY: <strategy_name>
LEMMAS: <comma-separated list>
DECOMPOSITION: <step1>; <step2>; ...
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def __call__(self, state: ProofState, **kwargs) -> StrategyPlan:
        """Generate high-level proof strategy."""
        # For now, use heuristic strategy selection
        # In full implementation, use LLM
        
        goal_lower = state.goal.lower()
        
        # Heuristic strategy selection based on goal structure
        if any(x in goal_lower for x in ['= 0', '= 1', '= true', '= false', 'rfl']):
            strategy = ProofStrategy.DIRECT
            depth = 1
        elif 'nat' in goal_lower or 'list' in goal_lower or '∀ n' in goal_lower:
            strategy = ProofStrategy.INDUCTION
            depth = 3
        elif '∨' in state.goal or 'or' in goal_lower:
            strategy = ProofStrategy.CASES
            depth = 2
        elif '¬' in state.goal or 'false' in goal_lower:
            strategy = ProofStrategy.CONTRADICTION
            depth = 3
        elif '=' in state.goal:
            strategy = ProofStrategy.REWRITE
            depth = 2
        else:
            strategy = ProofStrategy.APPLY_LEMMA
            depth = 2
        
        return StrategyPlan(
            strategy=strategy,
            target_lemmas=[],
            estimated_depth=depth,
            subgoal_decomposition=[],
            confidence=0.7,
        )


class TacticianAgent(BaseAgent):
    """
    Low-level tactic generation agent.
    
    Generates specific LEAN tactics conditioned on:
    1. Current proof state
    2. Strategy from Strategist
    3. Retrieved premises from Retriever
    4. Value estimates from Critic
    
    This conditioning makes tactic generation more focused and effective.
    """
    
    # Tactics organized by strategy
    STRATEGY_TACTICS = {
        ProofStrategy.DIRECT: [
            "rfl", "trivial", "decide", "native_decide", "norm_num", 
            "ring", "omega", "simp", "rfl"
        ],
        ProofStrategy.INDUCTION: [
            "induction {var} with", "induction {var}", "cases {var}",
            "induction {var} using Nat.strong_induction_on"
        ],
        ProofStrategy.CONTRADICTION: [
            "by_contra h", "exfalso", "absurd", "contradiction"
        ],
        ProofStrategy.CASES: [
            "cases {h}", "rcases {h} with ⟨_, _⟩", "obtain ⟨_, _⟩ := {h}",
            "by_cases h : {cond}", "split"
        ],
        ProofStrategy.REWRITE: [
            "rw [{lemma}]", "simp [{lemma}]", "conv => {tactic}",
            "calc", "rw [← {lemma}]"
        ],
        ProofStrategy.APPLY_LEMMA: [
            "apply {lemma}", "exact {lemma}", "refine {lemma} ?_",
            "have h := {lemma}", "use {term}"
        ],
        ProofStrategy.CUSTOM: [
            "simp_all", "aesop", "tauto", "decide"
        ],
    }
    
    # General tactics that work across strategies
    GENERAL_TACTICS = [
        "intro", "intros", "constructor", "ext", "funext",
        "simp", "simp_all", "ring", "linarith", "nlinarith",
        "omega", "positivity", "norm_num", "field_simp"
    ]
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def __call__(
        self, 
        state: ProofState, 
        strategy: StrategyPlan,
        premises: List[RetrievedPremise] = None,
        critic_guidance: CriticAssessment = None,
        num_candidates: int = 5,
        **kwargs
    ) -> List[TacticCandidate]:
        """Generate tactic candidates."""
        
        candidates = []
        
        # Get strategy-specific tactics
        strategy_tactics = self.STRATEGY_TACTICS.get(
            strategy.strategy, self.STRATEGY_TACTICS[ProofStrategy.CUSTOM]
        )
        
        # Add tactics aligned with strategy
        for tactic in strategy_tactics[:num_candidates]:
            # Fill in template variables if needed
            filled_tactic = self._fill_template(tactic, state, premises)
            candidates.append(TacticCandidate(
                tactic=filled_tactic,
                reasoning=f"Aligned with {strategy.strategy.value} strategy",
                prior=0.6,
                strategy_alignment=0.9,
            ))
        
        # Add general tactics
        for tactic in self.GENERAL_TACTICS[:3]:
            candidates.append(TacticCandidate(
                tactic=tactic,
                reasoning="General-purpose tactic",
                prior=0.3,
                strategy_alignment=0.3,
            ))
        
        # Add premise-based tactics if available
        if premises:
            for premise in premises[:2]:
                for template in ["apply {lemma}", "exact {lemma}", "rw [{lemma}]"]:
                    candidates.append(TacticCandidate(
                        tactic=template.format(lemma=premise.lemma_name),
                        reasoning=f"Using retrieved premise: {premise.usage_hint}",
                        prior=premise.relevance_score * 0.8,
                        strategy_alignment=0.7,
                    ))
        
        # If critic provides guidance, boost those tactics
        if critic_guidance and critic_guidance.suggested_tactics:
            for tactic in critic_guidance.suggested_tactics:
                candidates.append(TacticCandidate(
                    tactic=tactic,
                    reasoning="Critic-suggested tactic",
                    prior=0.8,
                    strategy_alignment=0.8,
                ))
        
        # Sort by prior and deduplicate
        seen = set()
        unique_candidates = []
        for c in sorted(candidates, key=lambda x: -x.prior):
            if c.tactic not in seen:
                seen.add(c.tactic)
                unique_candidates.append(c)
        
        return unique_candidates[:num_candidates]
    
    def _fill_template(
        self, 
        template: str, 
        state: ProofState,
        premises: List[RetrievedPremise] = None
    ) -> str:
        """Fill in tactic template variables."""
        
        # Extract variable names from hypotheses
        var_names = []
        for hyp in state.hypotheses:
            if ':' in hyp:
                var_name = hyp.split(':')[0].strip()
                var_names.append(var_name)
        
        # Fill templates
        if '{var}' in template and var_names:
            template = template.replace('{var}', var_names[0])
        if '{h}' in template and var_names:
            template = template.replace('{h}', var_names[0] if var_names else 'h')
        if '{cond}' in template:
            template = template.replace('{cond}', 'True')  # Placeholder
        if '{lemma}' in template:
            if premises:
                template = template.replace('{lemma}', premises[0].lemma_name)
            else:
                template = template.replace('{lemma}', '*')  # Wildcard
        if '{term}' in template:
            template = template.replace('{term}', '0')  # Placeholder
        if '{tactic}' in template:
            template = template.replace('{tactic}', 'rfl')
        
        return template


class CriticAgent(BaseAgent):
    """
    Value estimation and policy guidance agent.
    
    KEY INNOVATION: Unlike prior work that uses a single value/policy network,
    the Critic agent provides structured guidance through:
    1. Value estimate (probability of proof success)
    2. Policy prior (distribution over tactics)
    3. Pruning recommendations (which branches to abandon)
    4. Tactic suggestions (what to try next)
    
    The Critic is trained on proof traces with verification feedback,
    similar to RLPAF but integrated into the multi-agent framework.
    """
    
    def __init__(self, value_network=None):
        self.value_network = value_network
        # State visit counts for intrinsic reward
        self.state_visits: Dict[str, int] = defaultdict(int)
    
    def __call__(
        self, 
        state: ProofState,
        tactics: List[TacticCandidate] = None,
        **kwargs
    ) -> CriticAssessment:
        """Assess proof state and provide guidance."""
        
        # Compute base value from state features
        base_value = self._compute_value(state)
        
        # Add intrinsic bonus for novelty (RMaxTS-inspired)
        state_hash = state.hash()
        visits = self.state_visits[state_hash]
        self.state_visits[state_hash] += 1
        
        novelty_bonus = 0.1 / (1.0 + visits)
        value = min(1.0, base_value + novelty_bonus)
        
        # Compute policy prior over tactics
        policy_prior = {}
        if tactics:
            total = sum(t.prior * t.strategy_alignment for t in tactics)
            for t in tactics:
                score = t.prior * t.strategy_alignment
                policy_prior[t.tactic] = score / total if total > 0 else 1/len(tactics)
        
        # Determine if branch should be pruned
        should_prune = (
            state.depth > 10 or  # Too deep
            (visits > 5 and value < 0.1) or  # Visited often with low value
            (not state.syntax_valid)  # Invalid state
        )
        
        # Suggest tactics based on state analysis
        suggested = self._suggest_tactics(state)
        
        return CriticAssessment(
            value=value,
            policy_prior=policy_prior,
            reasoning=f"Value={value:.2f}, visits={visits}, depth={state.depth}",
            suggested_tactics=suggested,
            pruning_recommendation=should_prune,
        )
    
    def _compute_value(self, state: ProofState) -> float:
        """
        Compute value estimate from proof state features.
        
        In full implementation, this would use a trained neural network.
        For now, use heuristic features.
        """
        value = 0.5  # Base value
        
        # Positive signals
        if state.is_complete:
            return 1.0
        if state.goal_progress > 0:
            value += 0.2 * state.goal_progress
        if state.syntax_valid:
            value += 0.1
        if state.types_valid:
            value += 0.1
        if len(state.subgoals) == 0:
            value += 0.1
        
        # Negative signals
        if state.depth > 5:
            value -= 0.05 * (state.depth - 5)
        if not state.syntax_valid:
            value -= 0.3
        if len(state.subgoals) > 3:
            value -= 0.1
        
        return max(0.0, min(1.0, value))
    
    def _suggest_tactics(self, state: ProofState) -> List[str]:
        """Suggest tactics based on goal structure."""
        suggestions = []
        goal = state.goal.lower()
        
        # Pattern-based suggestions
        if '=' in state.goal:
            suggestions.extend(['rfl', 'ring', 'simp', 'omega'])
        if '∀' in state.goal or 'forall' in goal:
            suggestions.extend(['intro', 'intros'])
        if '∃' in state.goal or 'exists' in goal:
            suggestions.extend(['use', 'exists'])
        if '∧' in state.goal or 'and' in goal:
            suggestions.extend(['constructor', 'And.intro'])
        if '→' in state.goal or 'implies' in goal:
            suggestions.extend(['intro', 'fun'])
        if 'nat' in goal or 'ℕ' in state.goal:
            suggestions.extend(['omega', 'norm_num', 'decide'])
        
        return suggestions[:5]


class RetrieverAgent(BaseAgent):
    """
    Premise retrieval agent.
    
    Retrieves relevant lemmas/theorems from a library (e.g., Mathlib).
    Uses semantic similarity between the current goal and lemma statements.
    
    Inspired by ReProver but integrated into the multi-agent framework.
    """
    
    # Small set of common Mathlib lemmas for demonstration
    BUILTIN_LEMMAS = [
        ("Nat.add_comm", "∀ (n m : ℕ), n + m = m + n", "commutativity"),
        ("Nat.add_assoc", "∀ (n m k : ℕ), (n + m) + k = n + (m + k)", "associativity"),
        ("Nat.mul_comm", "∀ (n m : ℕ), n * m = m * n", "commutativity"),
        ("Nat.add_zero", "∀ (n : ℕ), n + 0 = n", "identity"),
        ("Nat.zero_add", "∀ (n : ℕ), 0 + n = n", "identity"),
        ("Nat.succ_pos", "∀ (n : ℕ), 0 < Nat.succ n", "positivity"),
        ("Nat.le_refl", "∀ (n : ℕ), n ≤ n", "reflexivity"),
        ("Nat.lt_succ_self", "∀ (n : ℕ), n < n + 1", "successor"),
        ("Int.add_comm", "∀ (a b : ℤ), a + b = b + a", "commutativity"),
        ("Real.add_comm", "∀ (a b : ℝ), a + b = b + a", "commutativity"),
        ("List.length_nil", "List.length [] = 0", "base case"),
        ("List.length_cons", "∀ (a : α) (l : List α), (a :: l).length = l.length + 1", "recursive"),
        ("And.intro", "∀ (a b : Prop), a → b → a ∧ b", "conjunction"),
        ("Or.inl", "∀ (a b : Prop), a → a ∨ b", "disjunction"),
        ("Or.inr", "∀ (a b : Prop), b → a ∨ b", "disjunction"),
    ]
    
    def __init__(self, index_path: str = None):
        self.index_path = index_path
        # In full implementation, load vector index of Mathlib
    
    def __call__(
        self, 
        state: ProofState, 
        k: int = 5,
        **kwargs
    ) -> List[RetrievedPremise]:
        """Retrieve relevant premises for the current goal."""
        
        goal_lower = state.goal.lower()
        scored_lemmas = []
        
        for name, statement, hint in self.BUILTIN_LEMMAS:
            # Simple keyword-based relevance scoring
            score = 0.0
            
            # Check for type matches
            if 'nat' in goal_lower and 'Nat' in name:
                score += 0.3
            if 'int' in goal_lower and 'Int' in name:
                score += 0.3
            if 'real' in goal_lower and 'Real' in name:
                score += 0.3
            if 'list' in goal_lower and 'List' in name:
                score += 0.3
            
            # Check for operation matches
            if '+' in state.goal and 'add' in name.lower():
                score += 0.2
            if '*' in state.goal and 'mul' in name.lower():
                score += 0.2
            if '=' in state.goal and 'eq' in hint:
                score += 0.1
            if '≤' in state.goal or '<' in state.goal:
                if 'le' in name.lower() or 'lt' in name.lower():
                    score += 0.2
            
            # Check for structural matches
            if '∧' in state.goal and 'And' in name:
                score += 0.3
            if '∨' in state.goal and 'Or' in name:
                score += 0.3
            
            if score > 0:
                scored_lemmas.append((name, statement, hint, score))
        
        # Sort by score and return top k
        scored_lemmas.sort(key=lambda x: -x[3])
        
        return [
            RetrievedPremise(
                lemma_name=name,
                lemma_statement=statement,
                relevance_score=score,
                usage_hint=f"Use for {hint}"
            )
            for name, statement, hint, score in scored_lemmas[:k]
        ]


# =============================================================================
# Unified VERITAS Search Engine
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for VERITAS search."""
    max_iterations: int = 100
    max_depth: int = 15
    exploration_constant: float = 1.5
    num_tactic_candidates: int = 8
    num_premises: int = 5
    value_weight: float = 0.3        # Weight for Critic value in UCB
    intrinsic_weight: float = 0.1    # Weight for novelty bonus
    strategy_bonus: float = 0.2      # Bonus for strategy-aligned tactics
    timeout_seconds: Optional[int] = None


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: ProofState
    tactic: Optional[str] = None      # Tactic that led to this state
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    
    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.5
    
    # Agent outputs cached
    strategy: Optional[StrategyPlan] = None
    critic_assessment: Optional[CriticAssessment] = None
    
    @property
    def q_value(self) -> float:
        """Average value."""
        return self.total_value / max(1, self.visits)
    
    @property
    def is_terminal(self) -> bool:
        """Check if node is terminal."""
        return self.state.is_complete or self.state.depth >= 15


class VERITASSearch:
    """
    Main VERITAS search engine.
    
    Coordinates the four agents through Critic-Guided MCTS:
    1. Strategist plans high-level approach
    2. Retriever fetches relevant premises
    3. Tactician generates candidates
    4. Critic guides selection and backpropagation
    """
    
    def __init__(
        self,
        strategist: StrategistAgent,
        tactician: TacticianAgent,
        critic: CriticAgent,
        retriever: RetrieverAgent,
        verifier,  # LEAN verification oracle
        config: SearchConfig = None,
    ):
        self.strategist = strategist
        self.tactician = tactician
        self.critic = critic
        self.retriever = retriever
        self.verifier = verifier
        self.config = config or SearchConfig()
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'nodes_expanded': 0,
            'lean_calls': 0,
            'proof_found': False,
            'proof_depth': 0,
            'best_signal_A': 0.0,
            'best_signal_B': 0.0,
            'best_signal_C': 0.0,
            'best_signal_D': 0.0,
        }
    
    def search(self, theorem: str, context: str = None) -> Dict[str, Any]:
        """
        Run VERITAS search to find a proof.
        
        Returns dict with:
        - success: bool
        - proof: List[str] or None
        - statistics: Dict
        """
        start_time = time.time()
        
        # Initialize root state
        initial_state = ProofState(
            theorem=theorem,
            goal=theorem,  # Initial goal is the theorem itself
            hypotheses=[],
            tactics_applied=[],
            subgoals=[],
            context=context,
            depth=0,
        )
        
        # Create root node
        root = MCTSNode(state=initial_state)
        
        # Get initial strategy
        root.strategy = self.strategist(initial_state)
        
        # Main search loop
        for iteration in range(self.config.max_iterations):
            self.stats['iterations'] = iteration + 1
            
            # Check timeout
            if self.config.timeout_seconds:
                if time.time() - start_time > self.config.timeout_seconds:
                    logger.info(f"Timeout after {iteration} iterations")
                    break
            
            # Selection: traverse tree to leaf using UCB + Critic value
            leaf = self._select(root)
            
            if leaf.is_terminal:
                if leaf.state.is_complete:
                    # Proof found!
                    self.stats['proof_found'] = True
                    self.stats['proof_depth'] = leaf.state.depth
                    proof = self._extract_proof(leaf)
                    return {
                        'success': True,
                        'proof': proof,
                        'statistics': self._get_statistics(start_time),
                    }
                continue
            
            # Expansion: generate children using agents
            self._expand(leaf)

            # Immediately check if expansion found a complete proof.
            # This catches proofs discovered during batch validation before
            # a timeout can fire at the top of the next iteration.
            for child in leaf.children:
                if child.state.is_complete:
                    self.stats['proof_found'] = True
                    self.stats['proof_depth'] = child.state.depth
                    return {
                        'success': True,
                        'proof': self._extract_proof(child),
                        'statistics': self._get_statistics(start_time),
                    }

            # Simulation: evaluate expanded node
            value = self._simulate(leaf)

            # Backpropagation: update tree with structured signals
            self._backpropagate(leaf, value)
        
        # No proof found
        return {
            'success': False,
            'proof': None,
            'statistics': self._get_statistics(start_time),
        }
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Select leaf node using UCB with Critic guidance.
        
        UCB formula: Q + V_critic * w_v + c * sqrt(ln(N_parent) / N) + novelty
        """
        node = root
        
        while node.children:
            # Get Critic assessment for value estimate
            if node.critic_assessment is None:
                node.critic_assessment = self.critic(node.state)
            
            # Check pruning recommendation
            if node.critic_assessment.pruning_recommendation:
                if node.parent:
                    return node.parent
                break
            
            # Select best child using enhanced UCB
            best_child = None
            best_score = -float('inf')
            
            for child in node.children:
                if child.visits == 0:
                    score = float('inf')  # Prioritize unvisited
                else:
                    # Standard UCB
                    exploit = child.q_value
                    explore = self.config.exploration_constant * math.sqrt(
                        math.log(node.visits + 1) / child.visits
                    )
                    
                    # Critic value bonus
                    if child.critic_assessment:
                        value_bonus = self.config.value_weight * child.critic_assessment.value
                    else:
                        value_bonus = 0
                    
                    # Strategy alignment bonus
                    if node.strategy and child.tactic:
                        strategy_tactics = TacticianAgent.STRATEGY_TACTICS.get(
                            node.strategy.strategy, []
                        )
                        if any(child.tactic.startswith(t.split()[0]) for t in strategy_tactics):
                            strategy_bonus = self.config.strategy_bonus
                        else:
                            strategy_bonus = 0
                    else:
                        strategy_bonus = 0
                    
                    score = exploit + explore + value_bonus + strategy_bonus
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child is None:
                break
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode):
        """
        Expand node using all agents.

        1. Get strategy (inherited or new)
        2. Retrieve relevant premises
        3. Get Critic guidance
        4. Generate tactic candidates
        5. Validate and create children (batch if verifier supports it)
        """
        self.stats['nodes_expanded'] += 1

        # Get or inherit strategy
        if node.strategy is None:
            if node.parent and node.parent.strategy:
                node.strategy = node.parent.strategy
            else:
                node.strategy = self.strategist(node.state)

        # Retrieve premises
        premises = self.retriever(node.state, k=self.config.num_premises)

        # Get Critic guidance
        node.critic_assessment = self.critic(node.state)

        # Generate tactic candidates
        tactics = self.tactician(
            node.state,
            strategy=node.strategy,
            premises=premises,
            critic_guidance=node.critic_assessment,
            num_candidates=self.config.num_tactic_candidates,
        )

        # Use batch validation when available: 6 tactics → 1 Lean call instead of 6
        if hasattr(self.verifier, 'validate_batch'):
            self._expand_batch(node, tactics)
        else:
            self._expand_sequential(node, tactics)

    def _expand_batch(self, node: MCTSNode, tactics: List[TacticCandidate]):
        """Validate all candidates in one Lean call. Updates node.children in place."""
        tactic_strings = [t.tactic for t in tactics]
        results = self.verifier.validate_batch(
            theorem=node.state.theorem,
            prefix_tactics=node.state.tactics_applied,
            candidate_tactics=tactic_strings,
        )
        self.stats['lean_calls'] += 1  # one Lean process for the whole batch

        for tactic_candidate, result in zip(tactics, results):
            # Track best signals seen across all validations
            self.stats['best_signal_A'] = max(self.stats['best_signal_A'], result.signal_A_syntax)
            self.stats['best_signal_B'] = max(self.stats['best_signal_B'], result.signal_B_typecheck)
            self.stats['best_signal_C'] = max(self.stats['best_signal_C'], result.signal_C_progress)
            self.stats['best_signal_D'] = max(self.stats['best_signal_D'], result.signal_D_complete)

            if not (result.signal_A_syntax > 0.5):
                continue
            proof_steps = node.state.tactics_applied + [tactic_candidate.tactic]
            # Use extracted remaining_goal; fall back to parent goal only if empty
            new_goal = result.remaining_goal if result.remaining_goal else node.state.goal
            new_state = ProofState(
                theorem=node.state.theorem,
                goal=new_goal,
                hypotheses=result.hypotheses if result.hypotheses else node.state.hypotheses,
                tactics_applied=proof_steps,
                subgoals=result.subgoals if result.subgoals else [],
                context=node.state.context,
                syntax_valid=result.signal_A_syntax > 0.5,
                types_valid=result.signal_B_typecheck > 0.5,
                goal_progress=result.signal_C_progress,
                is_complete=result.signal_D_complete > 0.5,
                depth=node.state.depth + 1,
                parent_tactic=tactic_candidate.tactic,
            )
            node.children.append(MCTSNode(
                state=new_state,
                tactic=tactic_candidate.tactic,
                parent=node,
                prior=tactic_candidate.prior,
            ))

    def _expand_sequential(self, node: MCTSNode, tactics: List[TacticCandidate]):
        """Original sequential expansion: one Lean call per candidate."""
        for tactic_candidate in tactics:
            self.stats['lean_calls'] += 1
            new_state = self._verify_tactic(node.state, tactic_candidate.tactic)
            if new_state is not None and new_state.syntax_valid:
                node.children.append(MCTSNode(
                    state=new_state,
                    tactic=tactic_candidate.tactic,
                    parent=node,
                    prior=tactic_candidate.prior,
                ))
    
    def _verify_tactic(self, state: ProofState, tactic: str) -> Optional[ProofState]:
        """Verify tactic using LEAN oracle and return new state."""
        
        # Build proof script
        proof_steps = state.tactics_applied + [tactic]
        
        # Call LEAN verifier
        result = self.verifier.validate(
            theorem=state.theorem,
            proof_steps=proof_steps,
            context=state.context,
        )
        
        # Create new state from result
        new_state = ProofState(
            theorem=state.theorem,
            goal=result.remaining_goal if hasattr(result, 'remaining_goal') else state.goal,
            hypotheses=result.hypotheses if hasattr(result, 'hypotheses') else state.hypotheses,
            tactics_applied=proof_steps,
            subgoals=result.subgoals if hasattr(result, 'subgoals') else [],
            context=state.context,
            syntax_valid=result.signal_A_syntax > 0.5 if hasattr(result, 'signal_A_syntax') else True,
            types_valid=result.signal_B_typecheck > 0.5 if hasattr(result, 'signal_B_typecheck') else True,
            goal_progress=result.signal_C_progress if hasattr(result, 'signal_C_progress') else 0,
            is_complete=result.signal_D_complete > 0.5 if hasattr(result, 'signal_D_complete') else False,
            depth=state.depth + 1,
            parent_tactic=tactic,
        )
        
        return new_state
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate from node using Critic value estimate.
        
        Unlike random rollouts, we use the Critic's learned value function
        to estimate the probability of finding a proof from this state.
        """
        if node.critic_assessment is None:
            node.critic_assessment = self.critic(node.state)
        
        # Base value from Critic
        value = node.critic_assessment.value
        
        # Bonus for proof completion
        if node.state.is_complete:
            value = 1.0
        
        # Bonus for LEAN verification signals
        if node.state.syntax_valid:
            value += 0.05
        if node.state.types_valid:
            value += 0.05
        value += 0.1 * node.state.goal_progress
        
        return min(1.0, value)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value through tree.
        
        Uses structured signals from LEAN to provide richer feedback.
        """
        current = node
        
        while current is not None:
            current.visits += 1
            
            # Decay value as we go up (distant nodes get less credit)
            current.total_value += value
            value *= 0.95  # Discount factor
            
            current = current.parent
    
    def _extract_proof(self, node: MCTSNode) -> List[str]:
        """Extract proof from successful node."""
        return node.state.tactics_applied
    
    def _get_statistics(self, start_time: float) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            **self.stats,
            'time_seconds': time.time() - start_time,
            'efficiency': self.stats['iterations'] / max(1, self.stats['lean_calls']),
            'best_signal_A': self.stats['best_signal_A'],
            'best_signal_B': self.stats['best_signal_B'],
            'best_signal_C': self.stats['best_signal_C'],
            'best_signal_D': self.stats['best_signal_D'],
        }


# =============================================================================
# Factory function for creating VERITAS
# =============================================================================

def create_veritas(verifier, config: SearchConfig = None) -> VERITASSearch:
    """
    Create a VERITAS search engine with all agents.
    
    Args:
        verifier: LEAN verification oracle
        config: Search configuration
        
    Returns:
        Configured VERITASSearch instance
    """
    return VERITASSearch(
        strategist=StrategistAgent(),
        tactician=TacticianAgent(),
        critic=CriticAgent(),
        retriever=RetrieverAgent(),
        verifier=verifier,
        config=config or SearchConfig(),
    )

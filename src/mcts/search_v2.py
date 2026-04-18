"""
MCTS Search V2

Enhanced MCTS with intrinsic rewards, progressive widening, and LLM-guided search.
Based on DeepSeek-Prover RMaxTS and AlphaProof techniques.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import math
import hashlib
import time

from src.mcts.node import MCTSNode
from src.mcts.tree import MCTSTree
from src.agents.lean_generator import LEANGenerator, GeneratedProofStep
from src.agents.proof_validator import ProofValidator, ValidationResult
from src.agents.reflector import Reflector, ReflectionResult

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for MCTS search."""
    num_simulations: int = 100
    max_depth: int = 20
    exploration_constant: float = 2.0  # Higher for more exploration
    max_reflection_rounds: int = 3
    
    # Intrinsic reward parameters (RMaxTS-style)
    intrinsic_reward_weight: float = 0.1  # Weight for novelty bonus
    progress_reward_weight: float = 0.3   # Weight for LEAN signals
    
    # Progressive widening parameters
    widening_alpha: float = 0.5  # Controls expansion rate
    widening_k: float = 1.0      # Initial children count
    
    # Value function parameters
    use_learned_value: bool = False  # Use neural value network
    value_network_path: Optional[str] = None


@dataclass
class SearchResult:
    """Result of MCTS search."""
    success: bool
    proof: Optional[List[str]]
    tree: MCTSTree
    statistics: Dict[str, Any]


@dataclass
class SearchStatistics:
    """Detailed search statistics."""
    total_simulations: int = 0
    total_expansions: int = 0
    total_lean_calls: int = 0
    unique_states_visited: int = 0
    proof_found_at_simulation: Optional[int] = None
    time_to_proof: Optional[float] = None
    max_depth_reached: int = 0
    intrinsic_rewards_given: int = 0
    successful_reflections: int = 0


class StateHasher:
    """Hash proof states for novelty detection."""
    
    def __init__(self):
        self.seen_states: Dict[str, int] = defaultdict(int)
    
    def hash_state(self, proof_steps: List[str], goal: str) -> str:
        """Create hash of proof state."""
        state_str = f"{goal}||{'|'.join(proof_steps)}"
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get_visit_count(self, state_hash: str) -> int:
        """Get number of times state has been visited."""
        return self.seen_states[state_hash]
    
    def record_visit(self, state_hash: str):
        """Record a visit to state."""
        self.seen_states[state_hash] += 1


class ProgressiveWidening:
    """
    Progressive widening for continuous action spaces.
    Dynamically expands tree based on visit counts.
    """
    
    def __init__(self, alpha: float = 0.5, k: float = 1.0):
        """
        Args:
            alpha: Widening exponent (smaller = slower expansion)
            k: Initial number of children
        """
        self.alpha = alpha
        self.k = k
    
    def max_children(self, visits: int) -> int:
        """Maximum children allowed for given visit count."""
        return max(1, int(self.k * (visits ** self.alpha)))
    
    def should_expand(self, node: MCTSNode) -> bool:
        """Determine if node should be expanded with new child."""
        max_c = self.max_children(node.visit_count)
        return len(node.children) < max_c


class IntrinsicRewardCalculator:
    """
    Calculate intrinsic rewards for exploration.
    Based on DeepSeek-Prover RMaxTS.
    """
    
    def __init__(self, 
                 novelty_weight: float = 0.1,
                 progress_weight: float = 0.3):
        """
        Args:
            novelty_weight: Weight for state novelty bonus
            progress_weight: Weight for LEAN progress signals
        """
        self.novelty_weight = novelty_weight
        self.progress_weight = progress_weight
        self.state_hasher = StateHasher()
    
    def compute_reward(
        self,
        proof_steps: List[str],
        goal: str,
        validation_result: ValidationResult,
    ) -> float:
        """
        Compute combined reward with intrinsic and extrinsic components.
        
        Args:
            proof_steps: Current proof steps
            goal: Current goal
            validation_result: LEAN validation result
            
        Returns:
            Combined reward in [0, 1+]
        """
        # Extrinsic reward: proof completion
        extrinsic = 1.0 if validation_result.signal_D_complete else 0.0
        
        # Progress reward from LEAN signals
        progress = (
            0.15 * validation_result.signal_A_syntax +
            0.20 * validation_result.signal_B_typecheck +
            0.50 * validation_result.signal_C_progress +
            0.15 * validation_result.signal_D_complete
        )
        
        # Intrinsic reward: novelty bonus
        state_hash = self.state_hasher.hash_state(proof_steps, goal)
        visit_count = self.state_hasher.get_visit_count(state_hash)
        self.state_hasher.record_visit(state_hash)
        
        novelty = 1.0 / (1.0 + visit_count)  # Decreases with visits
        
        # Combined reward
        reward = (
            extrinsic + 
            self.progress_weight * progress +
            self.novelty_weight * novelty
        )
        
        return min(reward, 2.0)  # Cap at 2.0


class MCTSSearchV2:
    """
    Enhanced Monte Carlo Tree Search for theorem proving.
    
    Improvements over V1:
    1. Intrinsic rewards for exploration (RMaxTS-style)
    2. Progressive widening for better action space coverage
    3. Improved UCB with value estimates
    4. Better signal integration from LEAN
    """
    
    def __init__(
        self,
        generator: LEANGenerator,
        validator: ProofValidator,
        reflector: Reflector,
        config: Optional[SearchConfig] = None,
    ):
        """
        Initialize enhanced MCTS Search.
        
        Args:
            generator: LEAN code generator
            validator: Proof validator
            reflector: Reflection agent
            config: Search configuration
        """
        self.generator = generator
        self.validator = validator
        self.reflector = reflector
        self.config = config or SearchConfig()
        
        # Initialize components
        self.reward_calculator = IntrinsicRewardCalculator(
            novelty_weight=self.config.intrinsic_reward_weight,
            progress_weight=self.config.progress_reward_weight,
        )
        self.progressive_widening = ProgressiveWidening(
            alpha=self.config.widening_alpha,
            k=self.config.widening_k,
        )
        
        # Statistics
        self.stats = SearchStatistics()
    
    def search(
        self,
        theorem: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SearchResult:
        """
        Run enhanced MCTS search to find a proof.
        
        Args:
            theorem: Theorem to prove
            context: Additional LEAN context
            timeout: Optional timeout in seconds
            
        Returns:
            SearchResult with proof if found
        """
        logger.info(f"Starting enhanced MCTS search")
        start_time = time.time()
        
        # Reset statistics
        self.stats = SearchStatistics()
        
        # Get initial proof state
        initial_state = self.validator.get_proof_state(theorem, [], context)
        
        # Create search tree
        tree = MCTSTree(
            theorem=theorem,
            initial_state=initial_state,
            max_depth=self.config.max_depth,
            exploration_constant=self.config.exploration_constant,
        )
        
        # Run simulations
        for sim in range(self.config.num_simulations):
            self.stats.total_simulations = sim + 1
            
            # Check for existing proof
            if tree.proven_nodes:
                self.stats.proof_found_at_simulation = sim
                self.stats.time_to_proof = time.time() - start_time
                logger.info(f"Proof found at simulation {sim}")
                break
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.info(f"Timeout after {sim} simulations")
                break
            
            # Selection phase
            leaf = self._select_with_ucb(tree)
            
            if leaf.is_terminal:
                continue
            
            # Expansion & Simulation
            should_expand = self.progressive_widening.should_expand(leaf)
            
            if should_expand:
                value, signals = self._expand_and_simulate(
                    leaf, theorem, context, tree
                )
                self.stats.total_expansions += 1
            else:
                # Use existing best child
                value, signals = self._simulate_existing(leaf)
            
            # Backpropagation with intrinsic rewards
            tree.backpropagate(leaf, value, signals)
            
            # Update statistics
            self.stats.max_depth_reached = max(
                self.stats.max_depth_reached,
                leaf.depth
            )
            
            # Periodic logging
            if sim > 0 and sim % 10 == 0:
                logger.debug(
                    f"Sim {sim}: nodes={tree.stats.total_nodes}, "
                    f"max_depth={self.stats.max_depth_reached}"
                )
        
        # Get result
        proof = tree.get_best_proof()
        
        self.stats.unique_states_visited = len(
            self.reward_calculator.state_hasher.seen_states
        )
        
        return SearchResult(
            success=proof is not None,
            proof=proof,
            tree=tree,
            statistics=self._get_statistics(),
        )
    
    def _select_with_ucb(self, tree: MCTSTree) -> MCTSNode:
        """
        Select leaf node using UCB with intrinsic value.
        
        Uses modified UCB formula:
        UCB = (Q + intrinsic) / N + c * sqrt(ln(parent_N) / N)
        """
        return tree.select()  # Use tree's built-in selection
    
    def _expand_and_simulate(
        self,
        node: MCTSNode,
        theorem: str,
        context: Optional[str],
        tree: MCTSTree,
    ) -> tuple[float, Dict[str, float]]:
        """
        Expand node and run simulation with intrinsic rewards.
        """
        # Get current proof state
        proof_steps = node.get_proof_steps()
        
        # Generate candidate tactics
        generated_steps = self.generator.generate_subgoal(
            current_goal=node.goal,
            proof_state=node.proof_state,
            history=proof_steps,
        )
        
        best_value = 0.0
        best_signals = {"A": 0, "B": 0, "C": 0, "D": 0}
        
        tactics = []
        priors = []
        states = []
        
        for step in generated_steps:
            # Validate
            self.stats.total_lean_calls += 1
            result = self.validator.validate(
                proof_step=step.code,
                theorem=theorem,
                prior_steps=proof_steps,
                context=context,
            )
            
            # Compute reward with intrinsic component
            new_proof_steps = proof_steps + [step.code]
            reward = self.reward_calculator.compute_reward(
                proof_steps=new_proof_steps,
                goal=node.goal,
                validation_result=result,
            )
            
            if reward > 0.1 + best_value:  # Novelty bonus threshold
                self.stats.intrinsic_rewards_given += 1
            
            signals = {
                "A": result.signal_A_syntax,
                "B": result.signal_B_typecheck,
                "C": result.signal_C_progress,
                "D": result.signal_D_complete,
            }
            
            # Apply reflection if needed
            if not result.is_valid and result.signal_A_syntax > 0:
                reflection = self.reflector.reflect(
                    goal=node.goal,
                    generated_step=step,
                    validation_result=result,
                    proof_history=proof_steps,
                )
                
                reward, signals, step = self._apply_reflection_v2(
                    step, result, reflection, theorem, proof_steps, context, node.goal
                )
            
            # Track best
            if reward > best_value:
                best_value = reward
                best_signals = signals
            
            # Add valid candidates
            if result.signal_A_syntax > 0:
                tactics.append(step.code)
                priors.append(step.confidence * (1 + reward))  # Boost by reward
                
                new_state = self.validator.get_proof_state(
                    theorem, proof_steps + [step.code], context
                )
                states.append(new_state)
        
        # Expand tree
        if tactics:
            tree.expand(node, tactics, priors, states)
        else:
            node.is_terminal = True
        
        return best_value, best_signals
    
    def _simulate_existing(self, node: MCTSNode) -> tuple[float, Dict[str, float]]:
        """Simulate using existing children."""
        if not node.children:
            return 0.0, {"A": 0, "B": 0, "C": 0, "D": 0}
        
        best_child = max(node.children, key=lambda c: c.value / max(1, c.visit_count))
        return best_child.value, best_child.signals if hasattr(best_child, 'signals') else {}
    
    def _apply_reflection_v2(
        self,
        step: GeneratedProofStep,
        result: ValidationResult,
        reflection: ReflectionResult,
        theorem: str,
        proof_steps: List[str],
        context: Optional[str],
        goal: str,
    ) -> tuple[float, Dict[str, float], GeneratedProofStep]:
        """
        Apply reflection with intrinsic reward tracking.
        """
        best_value = self.reward_calculator.compute_reward(
            proof_steps=proof_steps + [step.code],
            goal=goal,
            validation_result=result,
        )
        best_signals = {
            "A": result.signal_A_syntax,
            "B": result.signal_B_typecheck,
            "C": result.signal_C_progress,
            "D": result.signal_D_complete,
        }
        best_step = step
        
        for round_num in range(min(self.config.max_reflection_rounds, 
                                   len(reflection.priority_tactics))):
            suggested_tactic = reflection.priority_tactics[round_num]
            
            self.stats.total_lean_calls += 1
            new_result = self.validator.validate(
                proof_step=suggested_tactic,
                theorem=theorem,
                prior_steps=proof_steps,
                context=context,
            )
            
            new_reward = self.reward_calculator.compute_reward(
                proof_steps=proof_steps + [suggested_tactic],
                goal=goal,
                validation_result=new_result,
            )
            
            if new_reward > best_value:
                self.stats.successful_reflections += 1
                best_value = new_reward
                best_signals = {
                    "A": new_result.signal_A_syntax,
                    "B": new_result.signal_B_typecheck,
                    "C": new_result.signal_C_progress,
                    "D": new_result.signal_D_complete,
                }
                best_step = GeneratedProofStep(
                    code=suggested_tactic,
                    tactic=suggested_tactic,
                    confidence=reflection.confidence,
                    reasoning=reflection.analysis,
                )
            
            if new_result.signal_D_complete == 1.0:
                break
        
        return best_value, best_signals, best_step
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        return {
            "total_simulations": self.stats.total_simulations,
            "total_expansions": self.stats.total_expansions,
            "total_lean_calls": self.stats.total_lean_calls,
            "unique_states_visited": self.stats.unique_states_visited,
            "proof_found_at": self.stats.proof_found_at_simulation,
            "time_to_proof": self.stats.time_to_proof,
            "max_depth_reached": self.stats.max_depth_reached,
            "intrinsic_rewards_given": self.stats.intrinsic_rewards_given,
            "successful_reflections": self.stats.successful_reflections,
            "efficiency": self.stats.total_simulations / max(1, self.stats.total_lean_calls),
        }

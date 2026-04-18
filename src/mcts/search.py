"""
MCTS Search

Main search algorithm coordinating generator, validator, and reflector.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from src.mcts.node import MCTSNode
from src.mcts.tree import MCTSTree
from src.agents.lean_generator import LEANGenerator, GeneratedProofStep
from src.agents.proof_validator import ProofValidator, ValidationResult
from src.agents.reflector import Reflector, ReflectionResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of MCTS search."""
    success: bool
    proof: Optional[List[str]]
    tree: MCTSTree
    statistics: Dict[str, Any]


class MCTSSearch:
    """
    Monte Carlo Tree Search for theorem proving.
    
    Coordinates the three agents (Generator, Validator, Reflector)
    with MCTS to find proofs.
    """
    
    def __init__(
        self,
        generator: LEANGenerator,
        validator: ProofValidator,
        reflector: Reflector,
        num_simulations: int = 100,
        max_depth: int = 20,
        exploration_constant: float = 1.414,
        max_reflection_rounds: int = 3,
    ):
        """
        Initialize MCTS Search.
        
        Args:
            generator: LEAN code generator
            validator: Proof validator
            reflector: Reflection agent
            num_simulations: Number of MCTS simulations
            max_depth: Maximum proof depth
            exploration_constant: UCB exploration parameter
            max_reflection_rounds: Maximum reflection iterations
        """
        self.generator = generator
        self.validator = validator
        self.reflector = reflector
        
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.max_reflection_rounds = max_reflection_rounds
    
    def search(
        self,
        theorem: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SearchResult:
        """
        Run MCTS search to find a proof.
        
        Args:
            theorem: Theorem to prove
            context: Additional LEAN context
            timeout: Optional timeout in seconds
            
        Returns:
            SearchResult with proof if found
        """
        logger.info(f"Starting MCTS search for theorem")
        
        # Get initial proof state
        initial_state = self.validator.get_proof_state(theorem, [], context)
        
        # Create search tree
        tree = MCTSTree(
            theorem=theorem,
            initial_state=initial_state,
            max_depth=self.max_depth,
            exploration_constant=self.exploration_constant,
        )
        
        # Run simulations
        for sim in range(self.num_simulations):
            if tree.proven_nodes:
                logger.info(f"Proof found at simulation {sim}")
                break
            
            # Selection
            leaf = tree.select()
            
            if leaf.is_terminal:
                continue
            
            # Expansion & Simulation
            value, signals = self._expand_and_simulate(
                leaf, theorem, context, tree
            )
            
            # Backpropagation
            tree.backpropagate(leaf, value, signals)
            
            # Periodic pruning
            if sim > 0 and sim % 20 == 0:
                self._prune_tree(tree)
            
            logger.debug(f"Simulation {sim}: value={value:.3f}")
        
        # Get result
        proof = tree.get_best_proof()
        
        return SearchResult(
            success=proof is not None,
            proof=proof,
            tree=tree,
            statistics=self._get_statistics(tree),
        )
    
    def _expand_and_simulate(
        self,
        node: MCTSNode,
        theorem: str,
        context: Optional[str],
        tree: MCTSTree,
    ) -> tuple[float, Dict[str, float]]:
        """
        Expand a node and run simulation.
        
        Args:
            node: Node to expand
            theorem: Theorem being proved
            context: LEAN context
            tree: The search tree
            
        Returns:
            Tuple of (value, signals)
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
        
        # Validate each candidate
        tactics = []
        priors = []
        states = []
        
        for step in generated_steps:
            # Validate the step
            result = self.validator.validate(
                proof_step=step.code,
                theorem=theorem,
                prior_steps=proof_steps,
                context=context,
            )
            
            signals = {
                "A": result.signal_A_syntax,
                "B": result.signal_B_typecheck,
                "C": result.signal_C_progress,
                "D": result.signal_D_complete,
            }
            
            value = result.total_signal
            
            # Reflect if needed
            if not result.is_valid:
                reflection = self.reflector.reflect(
                    goal=node.goal,
                    generated_step=step,
                    validation_result=result,
                    proof_history=proof_steps,
                )
                
                # Apply reflection to refine
                value, signals, step = self._apply_reflection(
                    step, result, reflection, theorem, proof_steps, context
                )
            
            # Track best result
            if value > best_value:
                best_value = value
                best_signals = signals
            
            # Add to expansion candidates
            if result.signal_A_syntax > 0:  # At least syntactically valid
                tactics.append(step.code)
                priors.append(step.confidence)
                
                # Get resulting state
                new_state = self.validator.get_proof_state(
                    theorem, proof_steps + [step.code], context
                )
                states.append(new_state)
        
        # Expand tree with valid candidates
        if tactics:
            tree.expand(node, tactics, priors, states)
        else:
            node.is_terminal = True
        
        return best_value, best_signals
    
    def _apply_reflection(
        self,
        step: GeneratedProofStep,
        result: ValidationResult,
        reflection: ReflectionResult,
        theorem: str,
        proof_steps: List[str],
        context: Optional[str],
    ) -> tuple[float, Dict[str, float], GeneratedProofStep]:
        """
        Apply reflection to improve proof step.
        
        Args:
            step: Original generated step
            result: Validation result
            reflection: Reflection analysis
            theorem: Theorem being proved
            proof_steps: Current proof steps
            context: LEAN context
            
        Returns:
            Tuple of (value, signals, refined_step)
        """
        best_value = result.total_signal
        best_signals = {
            "A": result.signal_A_syntax,
            "B": result.signal_B_typecheck,
            "C": result.signal_C_progress,
            "D": result.signal_D_complete,
        }
        best_step = step
        
        # Try priority tactics from reflection
        for round_num in range(min(self.max_reflection_rounds, len(reflection.priority_tactics))):
            suggested_tactic = reflection.priority_tactics[round_num]
            
            new_result = self.validator.validate(
                proof_step=suggested_tactic,
                theorem=theorem,
                prior_steps=proof_steps,
                context=context,
            )
            
            if new_result.total_signal > best_value:
                best_value = new_result.total_signal
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
            
            # Stop if proof is complete
            if new_result.signal_D_complete == 1.0:
                break
        
        return best_value, best_signals, best_step
    
    def _prune_tree(self, tree: MCTSTree):
        """Prune unpromising branches from the tree."""
        def should_prune(node: MCTSNode) -> bool:
            return self.reflector.should_prune(
                node.value,
                node.visit_count,
                node.reflection_history,
            )
        
        pruned = tree.prune(should_prune)
        if pruned > 0:
            logger.debug(f"Pruned {pruned} nodes")
    
    def _get_statistics(self, tree: MCTSTree) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            "total_nodes": tree.stats.total_nodes,
            "max_depth": tree.stats.max_depth,
            "total_simulations": tree.stats.total_simulations,
            "proven_paths": tree.stats.proven_paths,
            "pruned_nodes": tree.stats.pruned_nodes,
            "success_rate": tree.stats.proven_paths / max(1, tree.stats.total_simulations),
        }

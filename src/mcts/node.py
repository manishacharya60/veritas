"""
MCTS Node

Represents a node in the Monte Carlo Tree Search.
Each node corresponds to a proof state after applying a tactic.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import math


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree representing a proof state.
    
    Attributes:
        proof_state: Current LEAN proof state (goals, hypotheses)
        tactic: Tactic that led to this state (None for root)
        parent: Parent node
        children: Child nodes
        visit_count: Number of times this node was visited
        value_sum: Sum of values from simulations through this node
        prior: Prior probability from generator
    """
    
    # State information
    proof_state: Dict[str, Any] = field(default_factory=dict)
    tactic: Optional[str] = None
    goal: str = ""
    
    # Tree structure
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    
    # MCTS statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    # Validation signals (A-D)
    signal_A: float = 0.0  # Syntax validity
    signal_B: float = 0.0  # Type check
    signal_C: float = 0.0  # Goal progress
    signal_D: float = 0.0  # Proof complete
    
    # Reflection information
    reflection_history: List[Any] = field(default_factory=list)
    is_terminal: bool = False
    is_proven: bool = False
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def combined_signal(self) -> float:
        """Combined signal from validation (A-D)."""
        return 0.25 * (self.signal_A + self.signal_B + self.signal_C + self.signal_D)
    
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB1 score for node selection.
        
        Args:
            exploration_constant: Controls exploration vs exploitation
            
        Returns:
            UCB1 score
        """
        if self.visit_count == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visit_count == 0:
            return self.value
        
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        
        return exploitation + exploration + self.prior
    
    def puct_score(
        self,
        exploration_constant: float = 1.414,
        prior_weight: float = 1.0,
    ) -> float:
        """
        Calculate PUCT score (used in AlphaZero-style MCTS).
        
        Args:
            exploration_constant: Controls exploration
            prior_weight: Weight for prior probability
            
        Returns:
            PUCT score
        """
        if self.parent is None:
            return 0.0
        
        exploitation = self.value
        exploration = (
            exploration_constant 
            * self.prior 
            * math.sqrt(self.parent.visit_count) 
            / (1 + self.visit_count)
        )
        
        return exploitation + prior_weight * exploration
    
    def expand(
        self,
        tactics: List[str],
        priors: List[float],
        proof_states: List[Dict[str, Any]],
    ) -> List['MCTSNode']:
        """
        Expand this node with child nodes.
        
        Args:
            tactics: List of candidate tactics
            priors: Prior probabilities for each tactic
            proof_states: Resulting proof states
            
        Returns:
            List of new child nodes
        """
        for tactic, prior, state in zip(tactics, priors, proof_states):
            child = MCTSNode(
                proof_state=state,
                tactic=tactic,
                goal=state.get("goals", [""])[0] if state.get("goals") else "",
                parent=self,
                prior=prior,
            )
            self.children.append(child)
        
        return self.children
    
    def update(self, value: float, signals: Dict[str, float] = None):
        """
        Update node statistics after simulation.
        
        Args:
            value: Value from simulation
            signals: Optional A-D signals from validation
        """
        self.visit_count += 1
        self.value_sum += value
        
        if signals:
            # Exponential moving average for signals
            alpha = 0.3
            self.signal_A = alpha * signals.get('A', 0) + (1 - alpha) * self.signal_A
            self.signal_B = alpha * signals.get('B', 0) + (1 - alpha) * self.signal_B
            self.signal_C = alpha * signals.get('C', 0) + (1 - alpha) * self.signal_C
            self.signal_D = alpha * signals.get('D', 0) + (1 - alpha) * self.signal_D
    
    def backpropagate(self, value: float, signals: Dict[str, float] = None):
        """
        Backpropagate value up the tree.
        
        Args:
            value: Value to propagate
            signals: Optional validation signals
        """
        node = self
        while node is not None:
            node.update(value, signals)
            node = node.parent
    
    def best_child(self, exploration_constant: float = 1.414) -> Optional['MCTSNode']:
        """
        Select best child according to UCB.
        
        Args:
            exploration_constant: UCB exploration parameter
            
        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None
        
        return max(
            self.children,
            key=lambda c: c.ucb_score(exploration_constant)
        )
    
    def most_visited_child(self) -> Optional['MCTSNode']:
        """Select child with most visits (for final move selection)."""
        if not self.children:
            return None
        
        return max(self.children, key=lambda c: c.visit_count)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    def depth(self) -> int:
        """Get depth of this node in the tree."""
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth
    
    def path_to_root(self) -> List['MCTSNode']:
        """Get path from this node to root."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def get_proof_steps(self) -> List[str]:
        """Get sequence of tactics from root to this node."""
        path = self.path_to_root()
        return [node.tactic for node in path if node.tactic is not None]
    
    def __repr__(self) -> str:
        return (
            f"MCTSNode(tactic={self.tactic}, "
            f"visits={self.visit_count}, "
            f"value={self.value:.3f}, "
            f"children={len(self.children)})"
        )

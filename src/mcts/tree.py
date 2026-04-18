"""
MCTS Tree

Manages the Monte Carlo Tree Search tree structure.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json

from src.mcts.node import MCTSNode


@dataclass
class TreeStatistics:
    """Statistics about the MCTS tree."""
    total_nodes: int = 0
    max_depth: int = 0
    total_simulations: int = 0
    proven_paths: int = 0
    pruned_nodes: int = 0


class MCTSTree:
    """
    Monte Carlo Tree Search tree for theorem proving.
    
    Manages the tree structure, node creation, and pruning.
    """
    
    def __init__(
        self,
        theorem: str,
        initial_state: Dict[str, Any],
        max_depth: int = 20,
        exploration_constant: float = 1.414,
    ):
        """
        Initialize the MCTS tree.
        
        Args:
            theorem: The theorem to prove
            initial_state: Initial LEAN proof state
            max_depth: Maximum depth of the tree
            exploration_constant: UCB exploration parameter
        """
        self.theorem = theorem
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        
        # Create root node
        self.root = MCTSNode(
            proof_state=initial_state,
            goal=initial_state.get("goals", [""])[0] if initial_state.get("goals") else theorem,
        )
        
        # Statistics
        self.stats = TreeStatistics(total_nodes=1)
        
        # Proven proofs found
        self.proven_nodes: List[MCTSNode] = []
    
    def select(self, node: MCTSNode = None) -> MCTSNode:
        """
        Select a leaf node for expansion using UCB.
        
        Args:
            node: Starting node (default: root)
            
        Returns:
            Selected leaf node
        """
        if node is None:
            node = self.root
        
        while not node.is_leaf() and not node.is_terminal:
            node = node.best_child(self.exploration_constant)
            if node is None:
                break
        
        return node
    
    def expand(
        self,
        node: MCTSNode,
        tactics: List[str],
        priors: List[float],
        proof_states: List[Dict[str, Any]],
    ) -> List[MCTSNode]:
        """
        Expand a node with new children.
        
        Args:
            node: Node to expand
            tactics: Candidate tactics
            priors: Prior probabilities
            proof_states: Resulting states
            
        Returns:
            List of new child nodes
        """
        if node.depth() >= self.max_depth:
            node.is_terminal = True
            return []
        
        children = node.expand(tactics, priors, proof_states)
        self.stats.total_nodes += len(children)
        self.stats.max_depth = max(self.stats.max_depth, node.depth() + 1)
        
        return children
    
    def backpropagate(
        self,
        node: MCTSNode,
        value: float,
        signals: Dict[str, float] = None,
    ):
        """
        Backpropagate value from node to root.
        
        Args:
            node: Starting node
            value: Value to propagate
            signals: Validation signals (A-D)
        """
        node.backpropagate(value, signals)
        self.stats.total_simulations += 1
        
        # Track if proof was found
        if signals and signals.get('D', 0) == 1.0:
            node.is_proven = True
            node.is_terminal = True
            self.proven_nodes.append(node)
            self.stats.proven_paths += 1
    
    def prune(
        self,
        should_prune: Callable[[MCTSNode], bool],
    ) -> int:
        """
        Prune nodes based on criteria.
        
        Args:
            should_prune: Function that returns True if node should be pruned
            
        Returns:
            Number of nodes pruned
        """
        pruned = 0
        
        def prune_recursive(node: MCTSNode):
            nonlocal pruned
            
            # Check children first (post-order)
            remaining_children = []
            for child in node.children:
                if should_prune(child):
                    pruned += 1 + self._count_descendants(child)
                    child.is_terminal = True
                else:
                    prune_recursive(child)
                    remaining_children.append(child)
            
            node.children = remaining_children
        
        prune_recursive(self.root)
        self.stats.pruned_nodes += pruned
        
        return pruned
    
    def _count_descendants(self, node: MCTSNode) -> int:
        """Count all descendants of a node."""
        count = len(node.children)
        for child in node.children:
            count += self._count_descendants(child)
        return count
    
    def get_best_proof(self) -> Optional[List[str]]:
        """
        Get the best proven proof found.
        
        Returns:
            List of tactics forming the proof, or None
        """
        if not self.proven_nodes:
            return None
        
        # Return shortest proof
        best_node = min(self.proven_nodes, key=lambda n: n.depth())
        return best_node.get_proof_steps()
    
    def get_best_partial_path(self) -> List[str]:
        """
        Get the most promising partial proof path.
        
        Returns:
            List of tactics for best partial path
        """
        # Follow most visited children from root
        path = []
        node = self.root
        
        while not node.is_leaf():
            best_child = node.most_visited_child()
            if best_child is None:
                break
            path.append(best_child.tactic)
            node = best_child
        
        return path
    
    def get_all_paths(
        self,
        min_visits: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get all paths in the tree with statistics.
        
        Args:
            min_visits: Minimum visits to include
            
        Returns:
            List of path dictionaries
        """
        paths = []
        
        def collect_paths(node: MCTSNode, current_path: List[str]):
            if node.visit_count >= min_visits:
                paths.append({
                    "tactics": current_path.copy(),
                    "visits": node.visit_count,
                    "value": node.value,
                    "signals": {
                        "A": node.signal_A,
                        "B": node.signal_B,
                        "C": node.signal_C,
                        "D": node.signal_D,
                    },
                    "is_proven": node.is_proven,
                })
            
            for child in node.children:
                new_path = current_path + [child.tactic] if child.tactic else current_path
                collect_paths(child, new_path)
        
        collect_paths(self.root, [])
        return paths
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        def node_to_dict(node: MCTSNode) -> Dict[str, Any]:
            return {
                "tactic": node.tactic,
                "goal": node.goal,
                "visits": node.visit_count,
                "value": node.value,
                "prior": node.prior,
                "signals": {
                    "A": node.signal_A,
                    "B": node.signal_B,
                    "C": node.signal_C,
                    "D": node.signal_D,
                },
                "is_proven": node.is_proven,
                "is_terminal": node.is_terminal,
                "children": [node_to_dict(c) for c in node.children],
            }
        
        return {
            "theorem": self.theorem,
            "statistics": {
                "total_nodes": self.stats.total_nodes,
                "max_depth": self.stats.max_depth,
                "total_simulations": self.stats.total_simulations,
                "proven_paths": self.stats.proven_paths,
                "pruned_nodes": self.stats.pruned_nodes,
            },
            "tree": node_to_dict(self.root),
        }
    
    def save(self, path: str):
        """Save tree to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self) -> str:
        return (
            f"MCTSTree(nodes={self.stats.total_nodes}, "
            f"depth={self.stats.max_depth}, "
            f"simulations={self.stats.total_simulations}, "
            f"proven={self.stats.proven_paths})"
        )

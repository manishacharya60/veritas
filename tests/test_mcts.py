"""
VERITAS Tests for MCTS
"""

import pytest

from src.mcts.node import MCTSNode
from src.mcts.tree import MCTSTree


class TestMCTSNode:
    """Tests for MCTS Node."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = MCTSNode(
            goal="test goal",
            tactic="simp",
        )
        
        assert node.goal == "test goal"
        assert node.tactic == "simp"
        assert node.visit_count == 0
        assert node.value == 0.0
    
    def test_node_value(self):
        """Test node value computation."""
        node = MCTSNode()
        node.visit_count = 10
        node.value_sum = 5.0
        
        assert node.value == 0.5
    
    def test_node_update(self):
        """Test node statistics update."""
        node = MCTSNode()
        
        node.update(0.7, {"A": 1.0, "B": 0.5, "C": 0.3, "D": 0.0})
        
        assert node.visit_count == 1
        assert node.value_sum == 0.7
        assert node.signal_A > 0
    
    def test_ucb_score_unvisited(self):
        """Test UCB score for unvisited node."""
        parent = MCTSNode()
        parent.visit_count = 10
        
        child = MCTSNode(parent=parent)
        
        assert child.ucb_score() == float('inf')
    
    def test_ucb_score_visited(self):
        """Test UCB score for visited node."""
        parent = MCTSNode()
        parent.visit_count = 100
        
        child = MCTSNode(parent=parent)
        child.visit_count = 10
        child.value_sum = 5.0
        
        score = child.ucb_score(exploration_constant=1.414)
        
        # Should be exploitation + exploration
        assert score > 0.5  # At least the exploitation term
    
    def test_expand(self):
        """Test node expansion."""
        node = MCTSNode(goal="initial")
        
        tactics = ["simp", "ring", "linarith"]
        priors = [0.5, 0.3, 0.2]
        states = [{"goals": ["g1"]}, {"goals": ["g2"]}, {"goals": ["g3"]}]
        
        children = node.expand(tactics, priors, states)
        
        assert len(children) == 3
        assert all(c.parent == node for c in children)
        assert node.children[0].tactic == "simp"
    
    def test_backpropagate(self):
        """Test backpropagation."""
        root = MCTSNode()
        child1 = MCTSNode(parent=root, tactic="t1")
        root.children.append(child1)
        child2 = MCTSNode(parent=child1, tactic="t2")
        child1.children.append(child2)
        
        child2.backpropagate(0.8)
        
        assert root.visit_count == 1
        assert child1.visit_count == 1
        assert child2.visit_count == 1
        assert root.value == 0.8
    
    def test_best_child(self):
        """Test best child selection."""
        parent = MCTSNode()
        parent.visit_count = 100
        
        # Create children with different stats
        child1 = MCTSNode(parent=parent, tactic="t1")
        child1.visit_count = 10
        child1.value_sum = 3.0
        
        child2 = MCTSNode(parent=parent, tactic="t2")
        child2.visit_count = 5
        child2.value_sum = 4.0
        
        parent.children = [child1, child2]
        
        best = parent.best_child()
        
        # child2 has higher value and more exploration bonus
        assert best is not None
    
    def test_most_visited_child(self):
        """Test most visited child selection."""
        parent = MCTSNode()
        
        child1 = MCTSNode(parent=parent, tactic="t1")
        child1.visit_count = 100
        
        child2 = MCTSNode(parent=parent, tactic="t2")
        child2.visit_count = 50
        
        parent.children = [child1, child2]
        
        most_visited = parent.most_visited_child()
        
        assert most_visited == child1
    
    def test_path_to_root(self):
        """Test path extraction."""
        root = MCTSNode()
        child1 = MCTSNode(parent=root, tactic="t1")
        root.children.append(child1)
        child2 = MCTSNode(parent=child1, tactic="t2")
        child1.children.append(child2)
        
        path = child2.path_to_root()
        
        assert len(path) == 3
        assert path[0] == root
        assert path[-1] == child2
    
    def test_get_proof_steps(self):
        """Test proof steps extraction."""
        root = MCTSNode()
        child1 = MCTSNode(parent=root, tactic="intro n")
        root.children.append(child1)
        child2 = MCTSNode(parent=child1, tactic="simp")
        child1.children.append(child2)
        
        steps = child2.get_proof_steps()
        
        assert steps == ["intro n", "simp"]
    
    def test_depth(self):
        """Test depth calculation."""
        root = MCTSNode()
        child1 = MCTSNode(parent=root)
        root.children.append(child1)
        child2 = MCTSNode(parent=child1)
        child1.children.append(child2)
        
        assert root.depth() == 0
        assert child1.depth() == 1
        assert child2.depth() == 2


class TestMCTSTree:
    """Tests for MCTS Tree."""
    
    def test_tree_creation(self):
        """Test tree initialization."""
        tree = MCTSTree(
            theorem="theorem test : True",
            initial_state={"goals": ["True"]},
        )
        
        assert tree.theorem == "theorem test : True"
        assert tree.root is not None
        assert tree.stats.total_nodes == 1
    
    def test_select_unexpanded(self):
        """Test selection on unexpanded tree."""
        tree = MCTSTree(
            theorem="test",
            initial_state={},
        )
        
        selected = tree.select()
        
        assert selected == tree.root
    
    def test_expand_tree(self):
        """Test tree expansion."""
        tree = MCTSTree(
            theorem="test",
            initial_state={},
        )
        
        tactics = ["t1", "t2"]
        priors = [0.5, 0.5]
        states = [{}, {}]
        
        children = tree.expand(tree.root, tactics, priors, states)
        
        assert len(children) == 2
        assert tree.stats.total_nodes == 3
    
    def test_backpropagate_tree(self):
        """Test tree backpropagation."""
        tree = MCTSTree(
            theorem="test",
            initial_state={},
        )
        
        tree.backpropagate(tree.root, 0.5, {"A": 1, "B": 1, "C": 0, "D": 0})
        
        assert tree.stats.total_simulations == 1
        assert tree.root.visit_count == 1
    
    def test_proof_found(self):
        """Test proof discovery tracking."""
        tree = MCTSTree(
            theorem="test",
            initial_state={},
        )
        
        tree.backpropagate(tree.root, 1.0, {"A": 1, "B": 1, "C": 1, "D": 1})
        
        assert tree.stats.proven_paths == 1
        assert len(tree.proven_nodes) == 1
    
    def test_get_best_proof(self):
        """Test best proof extraction."""
        tree = MCTSTree(
            theorem="test",
            initial_state={},
        )
        
        # Expand and mark as proven
        tree.expand(
            tree.root,
            ["simp"],
            [1.0],
            [{"goals": []}],
        )
        
        child = tree.root.children[0]
        child.is_proven = True
        tree.proven_nodes.append(child)
        
        proof = tree.get_best_proof()
        
        assert proof == ["simp"]
    
    def test_serialization(self):
        """Test tree serialization."""
        tree = MCTSTree(
            theorem="test theorem",
            initial_state={"goals": ["goal"]},
        )
        
        tree.expand(
            tree.root,
            ["t1", "t2"],
            [0.5, 0.5],
            [{}, {}],
        )
        
        data = tree.to_dict()
        
        assert data["theorem"] == "test theorem"
        assert len(data["tree"]["children"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

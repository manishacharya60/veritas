"""
VERITAS: Verification-Enhanced Reasoning with Integrated Tactic Agents and Search
"""

__version__ = "0.2.0"
__author__ = ""

# Core framework
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

# Legacy agents
from src.agents import LEANGenerator, ProofValidator, Reflector

# Legacy MCTS
from src.mcts import MCTSSearch, MCTSNode as LegacyMCTSNode, MCTSTree

__all__ = [
    # Core VERITAS framework
    "ProofState",
    "ProofStrategy", 
    "StrategyPlan",
    "TacticCandidate",
    "CriticAssessment",
    "RetrievedPremise",
    "StrategistAgent",
    "TacticianAgent",
    "CriticAgent",
    "RetrieverAgent",
    "VERITASSearch",
    "SearchConfig",
    "MCTSNode",
    "create_veritas",
    # Legacy components
    "LEANGenerator",
    "ProofValidator", 
    "Reflector",
    "MCTSSearch",
    "LegacyMCTSNode",
    "MCTSTree",
]

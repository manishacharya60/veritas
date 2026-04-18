# VERITAS Design Improvements Based on Literature Review

## Executive Summary

After reviewing state-of-the-art theorem proving systems (DeepSeek-Prover, AlphaProof, LeanDojo/ReProver, Llemma, GPT-f), here are critical improvements for VERITAS.

---

## Literature Review Key Findings

### 1. DeepSeek-Prover V1.5 (SOTA: 63.5% miniF2F)
- **RMaxTS**: Modified MCTS using **intrinsic rewards** for exploration
- **RLPAF**: Reinforcement Learning from Proof Assistant Feedback
- **Key Insight**: Intrinsic reward encourages diverse proof paths, not just success
- **Training**: 8M synthetic theorem-proof pairs

### 2. AlphaProof (IMO Silver Medal)
- **AlphaZero-style RL**: Self-play with LEAN verification as ground truth
- **Formalizer Network**: Gemini fine-tuned for NL → LEAN translation
- **Key Insight**: Millions of problems for training, continuous self-improvement
- **Online Learning**: Reinforced proofs of problem variations during competition

### 3. LeanDojo + ReProver (48% → 51.2% with retrieval)
- **Premise Selection**: Critical bottleneck in theorem proving
- **Retrieval-Augmented**: Retrieves relevant lemmas from Mathlib
- **Key Insight**: Access to the right premises > better search
- **Efficient**: Only 1 GPU-week training

### 4. PACT (Proof Artifact Co-Training)
- **Self-supervised**: Extract training signal from proof terms
- **Key Insight**: Use LEAN's kernel-level proof terms as auxiliary training
- **Improvement**: 32% → 48% success rate

### 5. GPT-f / Llemma
- **Generative LM**: Generate entire tactic applications
- **Key Insight**: LLMs can generate novel mathematical terms
- **Llemma**: Math-specialized model outperforms general models

---

## Current VERITAS Weaknesses

| Component | Current State | Limitation |
|-----------|--------------|------------|
| MCTS | Standard UCB | No intrinsic rewards for exploration |
| Tactic Generation | Random from fixed list | No LLM-guided generation |
| Value Function | Heuristic (A-D signals) | Not learned |
| Premise Selection | None | Can't use library lemmas |
| Training | None | No RL loop |

---

## Proposed Improvements

### Improvement 1: RMaxTS-style Intrinsic Rewards

**Current**: 
```python
reward = 0.25*A + 0.25*B + 0.35*C + 0.15*D  # Static weights
```

**Improved**:
```python
# Intrinsic reward for exploration (DeepSeek-Prover style)
def compute_reward(node, result, visit_counts):
    # Extrinsic: proof success
    extrinsic = 1.0 if result["success"] else 0.0
    
    # Intrinsic: novelty bonus (encourage diverse paths)
    state_hash = hash(node.proof_state)
    novelty = 1.0 / (1.0 + visit_counts.get(state_hash, 0))
    
    # Progress signal from LEAN
    progress = 0.0
    if result["signal_A"]: progress += 0.2
    if result["signal_B"]: progress += 0.2
    if result["signal_C"]: progress += 0.4  # Goal progress most important
    
    # Combined reward
    return extrinsic + 0.1 * novelty + 0.3 * progress
```

### Improvement 2: LLM-Guided Tactic Generation

**Current**: Random selection from fixed tactic list

**Improved**: Use LLM to generate contextual tactics

```python
class LLMTacticGenerator:
    """Generate tactics using LLM based on proof state"""
    
    PROMPT = """
Given the current proof state in LEAN 4:
{proof_state}

Current goal: {goal}

Generate the next tactic to apply. Consider:
1. The goal type (equality, inequality, existence, etc.)
2. Available hypotheses
3. Relevant lemmas from the context

Output only the tactic, nothing else.
"""
    
    def generate_tactics(self, proof_state: str, goal: str, n: int = 5) -> List[str]:
        """Generate n candidate tactics"""
        prompt = self.PROMPT.format(proof_state=proof_state, goal=goal)
        
        # Use local model (Llemma-7B) or API
        responses = self.llm.generate(prompt, n=n, temperature=0.8)
        
        # Also include some standard tactics for exploration
        standard = random.sample(STANDARD_TACTICS, min(3, len(STANDARD_TACTICS)))
        
        return responses + standard
```

### Improvement 3: Premise Retrieval (ReProver-style)

**Key Insight**: Most proof failures come from not knowing which lemmas to use

```python
class PremiseRetriever:
    """Retrieve relevant premises from Mathlib"""
    
    def __init__(self, index_path: str):
        # Load pre-built index of Mathlib lemmas
        self.index = self.load_index(index_path)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, goal: str, k: int = 10) -> List[str]:
        """Retrieve k most relevant premises for the goal"""
        goal_embedding = self.encoder.encode(goal)
        
        # Similarity search
        scores = self.index.search(goal_embedding, k)
        
        return [self.index.get_lemma(i) for i, _ in scores]
    
    def augment_tactics(self, tactics: List[str], premises: List[str]) -> List[str]:
        """Augment tactics with retrieved premises"""
        augmented = tactics.copy()
        
        for premise in premises:
            augmented.append(f"apply {premise}")
            augmented.append(f"exact {premise}")
            augmented.append(f"rw [{premise}]")
            augmented.append(f"simp [{premise}]")
        
        return augmented
```

### Improvement 4: Learned Value Function

**Current**: Heuristic value from A-D signals

**Improved**: Train a value network

```python
class ProofValueNetwork(nn.Module):
    """Neural network to estimate proof success probability"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.value_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, proof_state: str) -> float:
        """Estimate probability of finding proof from this state"""
        tokens = self.tokenizer(proof_state, return_tensors="pt")
        embedding = self.encoder(**tokens).last_hidden_state[:, 0, :]
        return self.value_head(embedding)
```

### Improvement 5: Progressive Widening

**Current**: Fixed branching factor

**Improved**: Dynamically expand tree based on visits

```python
class ProgressiveWidening:
    """Progressive widening for infinite action spaces"""
    
    def __init__(self, alpha: float = 0.5, k: float = 1.0):
        self.alpha = alpha  # Widening rate
        self.k = k  # Initial width
    
    def should_expand(self, node: MCTSNode) -> bool:
        """Decide whether to expand with new action or select existing child"""
        max_children = int(self.k * (node.visits ** self.alpha))
        return len(node.children) < max_children
    
    def select_or_expand(self, node: MCTSNode, tactic_generator) -> MCTSNode:
        if self.should_expand(node):
            # Generate new tactic
            new_tactic = tactic_generator.generate_one()
            return node.add_child(new_tactic)
        else:
            # Select existing best child
            return node.best_child()
```

### Improvement 6: Proof State Encoding

**Key Insight**: Better state representation improves everything

```python
class ProofStateEncoder:
    """Rich encoding of LEAN proof state"""
    
    def encode(self, lean_state: dict) -> torch.Tensor:
        """
        Encode proof state with:
        - Goal type and structure
        - Available hypotheses
        - Tactics already applied
        - Depth in proof tree
        """
        components = []
        
        # Goal encoding
        goal_emb = self.encode_expr(lean_state["goal"])
        components.append(goal_emb)
        
        # Hypothesis encoding
        for hyp in lean_state["hypotheses"]:
            hyp_emb = self.encode_expr(hyp)
            components.append(hyp_emb)
        
        # Proof history
        history_emb = self.encode_history(lean_state["tactics_applied"])
        components.append(history_emb)
        
        # Aggregate
        return self.aggregate(components)
```

---

## Recommended Implementation Priority

| Priority | Improvement | Effort | Impact |
|----------|------------|--------|--------|
| 1 | Intrinsic Rewards | Low | High |
| 2 | LLM Tactic Generation | Medium | High |
| 3 | Progressive Widening | Low | Medium |
| 4 | Premise Retrieval | High | Very High |
| 5 | Learned Value Function | High | High |
| 6 | RLPAF Training Loop | Very High | Transformative |

---

## Updated Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VERITAS v2                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   LLM        │    │   Premise    │    │   Value Network      │  │
│  │   Generator  │───▶│   Retriever  │───▶│   (Learned)          │  │
│  │   (Llemma)   │    │   (ReProver) │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │               │
│         ▼                   ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    MCTS Search (RMaxTS)                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Progressive │  │ Intrinsic   │  │ UCB + Value         │  │   │
│  │  │ Widening    │  │ Rewards     │  │ Network             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LEAN 4 Verifier                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Signal A-D  │  │ Proof Term  │  │ Goal State          │  │   │
│  │  │ Extraction  │  │ Analysis    │  │ Extraction          │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Reflector Agent                          │   │
│  │  - Error analysis and tactic suggestion                     │   │
│  │  - Premise recommendation based on errors                   │   │
│  │  - Proof strategy adjustment                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Expected Performance Gains

| Benchmark | Current | With Improvements | Target |
|-----------|---------|-------------------|--------|
| miniF2F (MATHD) | 25% | 40-50% | 60%+ |
| miniF2F (AMC) | 0% | 15-25% | 35%+ |
| miniF2F (AIME) | 0% | 5-15% | 25%+ |
| miniF2F (overall) | 13% | 25-35% | 50%+ |

**Note**: DeepSeek-Prover V1.5 achieves 63.5% with full training. Our target is competitive with less compute.

---

## Next Steps

1. **Implement intrinsic rewards** (1-2 days)
2. **Integrate Llemma-7B for tactic generation** (3-5 days)
3. **Add progressive widening** (1 day)
4. **Build premise retrieval index** (1 week)
5. **Train value network on proof traces** (2 weeks)
6. **Implement RLPAF training loop** (1 month)

---

## References

1. DeepSeek-Prover-V1.5 (2024) - RMaxTS, RLPAF
2. AlphaProof (2024) - AlphaZero for proofs
3. LeanDojo/ReProver (2023) - Retrieval-augmented proving
4. PACT (2021) - Proof artifact co-training
5. Llemma (2023) - Math-specialized LLM
6. GPT-f (2020) - Generative theorem proving

"""
Shared benchmark infrastructure for VERITAS experiment runners.
Used by run_minif2f.py and run_leandojo.py.
"""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lean.lake_validator import LakeValidator, ValidationResult
from src.veritas import (
    ProofState, SearchConfig, VERITASSearch,
    StrategistAgent, TacticianAgent, CriticAgent, RetrieverAgent,
)

# ---------------------------------------------------------------------------
# Tactic Portfolios
# ---------------------------------------------------------------------------

PORTFOLIO_FAST = [
    "rfl", "norm_num", "decide", "native_decide", "omega",
    "ring", "simp", "aesop", "trivial", "linarith", "nlinarith",
]

PORTFOLIO_EXTENDED = [
    "rfl", "norm_num", "decide", "native_decide", "omega",
    "ring", "simp", "aesop", "trivial", "linarith", "nlinarith",
    "positivity", "norm_cast", "push_cast", "field_simp",
    "simp; ring", "norm_num; ring",
    "intro h; simp at h ⊢", "intro n; simp", "intro n; omega",
    "intro n m; ring", "intro n m; omega",
    "simp [mul_comm]", "simp [add_comm]",
    "ring_nf; norm_num", "simp; linarith", "simp; omega",
    "norm_num [Nat.factorial]", "simp [Nat.add_zero]",
    "intro h; linarith", "intro h; nlinarith [sq_nonneg h]",
    "constructor <;> norm_num", "simp only [ne_eq]; norm_num",
]

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TheoremResult:
    name: str
    category: str
    method: str
    solved: bool
    proof: Optional[List[str]]
    elapsed_seconds: float
    lean_calls: int
    failure_reason: Optional[str] = None
    best_signal_A: float = 0.0
    best_signal_B: float = 0.0
    best_signal_C: float = 0.0
    best_signal_D: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    method: str
    total: int
    solved: int
    solve_rate: float
    total_lean_calls: int
    total_time_seconds: float
    per_category: Dict[str, Dict[str, int]]
    per_theorem: List[Dict[str, Any]]
    timestamp: str
    config: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Method: {self.method}")
        print(f"Solved: {self.solved}/{self.total} ({self.solve_rate:.1%})")
        print(f"LEAN calls: {self.total_lean_calls}")
        print(f"Time: {self.total_time_seconds:.1f}s")
        print(f"\nPer-category breakdown:")
        for cat, counts in sorted(self.per_category.items()):
            solved = counts.get("solved", 0)
            total = counts.get("total", 0)
            pct = solved / total if total > 0 else 0
            print(f"  {cat:20s}: {solved:3d}/{total:3d}  ({pct:.0%})")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Portfolio Baseline
# ---------------------------------------------------------------------------

class PortfolioBaseline:
    def __init__(self, validator: LakeValidator, portfolio: List[str] = None):
        self.validator = validator
        self.portfolio = portfolio or PORTFOLIO_EXTENDED
        self._lock = threading.Lock()
        self._call_count = 0

    def prove(self, theorem, timeout_per_batch: int = 60) -> TheoremResult:
        start = time.time()
        winning_idx = self.validator.validate_portfolio(
            theorem_statement=theorem.statement,
            tactics=self.portfolio,
            timeout=timeout_per_batch,
        )
        with self._lock:
            self._call_count += 1
        elapsed = time.time() - start

        if winning_idx is not None:
            tactic_str = self.portfolio[winning_idx]
            proof = [t.strip() for t in tactic_str.replace(";", "\n").split("\n") if t.strip()]
            return TheoremResult(
                name=theorem.name, category=theorem.category, method="portfolio",
                solved=True, proof=proof, elapsed_seconds=elapsed, lean_calls=1,
                best_signal_A=1.0, best_signal_B=1.0, best_signal_C=1.0, best_signal_D=1.0,
            )
        return TheoremResult(
            name=theorem.name, category=theorem.category, method="portfolio",
            solved=False, proof=None, elapsed_seconds=elapsed, lean_calls=1,
            failure_reason="portfolio_exhausted",
        )

    @property
    def call_count(self):
        return self._call_count


# ---------------------------------------------------------------------------
# Best-of-N Claude Baseline
# ---------------------------------------------------------------------------

class BestOfNClaude:
    def __init__(
        self,
        validator: LakeValidator,
        n: int = 32,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.8,
    ):
        self.validator = validator
        self.n = n
        self.model = model
        self.temperature = temperature
        self._lock = threading.Lock()
        self._call_count = 0
        self.last_tactics: List[str] = []

        import anthropic
        self._client = anthropic.Anthropic()

    def _generate_tactics_single_call(self, theorem, k: int) -> List[str]:
        prompt = f"""Prove the following Lean 4 theorem. Suggest {k} different proof attempts.

```lean
{theorem.statement}
```

Output ONLY a JSON array of tactic strings. Each string can be a single tactic or multiple tactics joined by newlines. Do NOT include sorry.

Example:
["ring", "norm_num", "omega", "simp; ring", "intro n\\nomega"]

Respond with the JSON array only."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if arr_match:
            try:
                tactics = json.loads(arr_match.group(0))
                return [t for t in tactics if isinstance(t, str) and "sorry" not in t.lower()]
            except json.JSONDecodeError:
                pass
        fallback = re.findall(r'"([^"]+)"', raw)
        return [t for t in fallback if "sorry" not in t.lower()][:k]

    def _generate_tactics(self, theorem) -> List[str]:
        if self.n <= 50:
            return self._generate_tactics_single_call(theorem, self.n)
        batch_size = 50
        num_calls = (self.n + batch_size - 1) // batch_size
        all_tactics, seen = [], set()
        for _ in range(num_calls):
            for t in self._generate_tactics_single_call(theorem, batch_size):
                if t not in seen:
                    seen.add(t)
                    all_tactics.append(t)
            if len(all_tactics) >= self.n:
                break
        return all_tactics[:self.n]

    def prove(self, theorem, timeout: int = 120, batch_size: int = 50) -> TheoremResult:
        start = time.time()
        method_label = f"best_of_{self.n}_claude"

        try:
            tactics = self._generate_tactics(theorem)
        except Exception as e:
            return TheoremResult(
                name=theorem.name, category=theorem.category,
                method=method_label, solved=False, proof=None,
                elapsed_seconds=time.time() - start, lean_calls=0,
                failure_reason=f"Claude API error: {e}",
            )

        self.last_tactics = tactics
        if not tactics:
            return TheoremResult(
                name=theorem.name, category=theorem.category,
                method=method_label, solved=False, proof=None,
                elapsed_seconds=time.time() - start, lean_calls=0,
                failure_reason="no_tactics_generated",
            )

        lean_calls = 0
        for batch_start in range(0, len(tactics), batch_size):
            batch = tactics[batch_start:batch_start + batch_size]
            winning_idx = self.validator.validate_portfolio(
                theorem_statement=theorem.statement,
                tactics=batch,
                timeout=timeout,
            )
            lean_calls += 1
            with self._lock:
                self._call_count += 1

            if winning_idx is not None:
                tactic_str = batch[winning_idx]
                proof = [t.strip() for t in tactic_str.replace(";", "\n").split("\n") if t.strip()]
                return TheoremResult(
                    name=theorem.name, category=theorem.category,
                    method=method_label, solved=True, proof=proof,
                    elapsed_seconds=time.time() - start, lean_calls=lean_calls,
                    best_signal_A=1.0, best_signal_B=1.0, best_signal_C=1.0, best_signal_D=1.0,
                )

        return TheoremResult(
            name=theorem.name, category=theorem.category,
            method=method_label, solved=False, proof=None,
            elapsed_seconds=time.time() - start, lean_calls=lean_calls,
            failure_reason=f"all_{len(tactics)}_tactics_failed",
        )


# ---------------------------------------------------------------------------
# VERITAS adapter
# ---------------------------------------------------------------------------

class _LegacyResult:
    def __init__(self, r: ValidationResult):
        self.signal_A_syntax = r.signal_A_syntax
        self.signal_B_typecheck = r.signal_B_typecheck
        self.signal_C_progress = r.signal_C_progress
        self.signal_D_complete = r.signal_D_complete
        self.remaining_goal = r.remaining_goal
        self.hypotheses = r.hypotheses
        self.subgoals = r.subgoals
        self.stderr = r.stderr
        self.success = r.success


class VERITASVerifierAdapter:
    def __init__(self, validator: LakeValidator):
        self._v = validator
        self.call_count = 0
        self._lock = threading.Lock()

    def validate(self, theorem: str, proof_steps: List[str], context: str = None) -> Any:
        with self._lock:
            self.call_count += 1
        result = self._v.validate(theorem_statement=theorem, tactics=proof_steps)
        return _LegacyResult(result)


# ---------------------------------------------------------------------------
# VERITAS search runners
# ---------------------------------------------------------------------------

def run_veritas(
    theorem,
    validator: LakeValidator,
    max_iterations: int = 50,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    use_claude_api: bool = False,
    theorem_timeout: int = 300,
) -> TheoremResult:
    start = time.time()
    verifier_adapter = VERITASVerifierAdapter(validator)

    if use_claude_api:
        from src.agents.claude_tactician import ClaudeTacticianAgent, ClaudeCriticAgent
        tactician = ClaudeTacticianAgent()
        critic = ClaudeCriticAgent()
        method_name = "veritas_claude"
    elif model is not None:
        from experiments.simple_comparison import LLMTacticianAgent
        tactician = LLMTacticianAgent(model, tokenizer, device)
        critic = CriticAgent()
        method_name = "veritas_llm"
    else:
        tactician = TacticianAgent()
        critic = CriticAgent()
        method_name = "veritas"

    search = VERITASSearch(
        strategist=StrategistAgent(),
        tactician=tactician,
        critic=critic,
        retriever=RetrieverAgent(),
        verifier=verifier_adapter,
        config=SearchConfig(
            max_iterations=max_iterations, max_depth=10,
            num_tactic_candidates=6, exploration_constant=1.5,
            value_weight=0.3, intrinsic_weight=0.1,
            strategy_bonus=0.2, timeout_seconds=theorem_timeout,
        ),
    )

    result = search.search(theorem.statement)
    elapsed = time.time() - start
    return TheoremResult(
        name=theorem.name, category=theorem.category, method=method_name,
        solved=result.get("success", False), proof=result.get("proof"),
        elapsed_seconds=elapsed, lean_calls=verifier_adapter.call_count,
    )


def run_veritas_two_phase(
    theorem,
    validator: LakeValidator,
    max_iterations: int = 50,
    sweep_n: int = 5,
    theorem_timeout: int = 300,
) -> TheoremResult:
    from src.agents.claude_tactician import ClaudeTacticianAgent, ClaudeCriticAgent

    start = time.time()
    sweeper = BestOfNClaude(validator, n=sweep_n)
    phase1 = sweeper.prove(theorem, timeout=90)
    phase1_elapsed = time.time() - start

    if phase1.solved:
        return TheoremResult(
            name=theorem.name, category=theorem.category, method="veritas_two_phase",
            solved=True, proof=phase1.proof,
            elapsed_seconds=phase1_elapsed, lean_calls=phase1.lean_calls,
        )

    failed_tactics = sweeper.last_tactics
    remaining_timeout = theorem_timeout - phase1_elapsed

    if remaining_timeout < 30:
        return TheoremResult(
            name=theorem.name, category=theorem.category, method="veritas_two_phase",
            solved=False, proof=None,
            elapsed_seconds=time.time() - start, lean_calls=phase1.lean_calls,
            failure_reason="phase1_exhausted_budget",
        )

    tactician = ClaudeTacticianAgent(phase1_failed_tactics=failed_tactics)
    critic = ClaudeCriticAgent()
    verifier_adapter = VERITASVerifierAdapter(validator)

    search = VERITASSearch(
        strategist=StrategistAgent(),
        tactician=tactician,
        critic=critic,
        retriever=RetrieverAgent(),
        verifier=verifier_adapter,
        config=SearchConfig(
            max_iterations=max_iterations, max_depth=10,
            num_tactic_candidates=6, exploration_constant=1.5,
            value_weight=0.3, intrinsic_weight=0.1,
            strategy_bonus=0.2, timeout_seconds=remaining_timeout,
        ),
    )

    mcts_result = search.search(theorem.statement)
    elapsed = time.time() - start
    return TheoremResult(
        name=theorem.name, category=theorem.category, method="veritas_two_phase",
        solved=mcts_result.get("success", False), proof=mcts_result.get("proof"),
        elapsed_seconds=elapsed, lean_calls=phase1.lean_calls + verifier_adapter.call_count,
    )


def run_veritas_ablation(
    theorem,
    validator: LakeValidator,
    variant_config: Dict[str, Any],
    max_iterations: int = 50,
    theorem_timeout: int = 300,
) -> TheoremResult:
    start = time.time()
    verifier_adapter = VERITASVerifierAdapter(validator)

    strategist = None if variant_config.get("disable_strategist") else StrategistAgent()
    critic_agent = None if variant_config.get("disable_critic") else CriticAgent()
    retriever = None if variant_config.get("disable_retriever") else RetrieverAgent()

    if strategist is None:
        from src.veritas import StrategistAgent as SA, ProofStrategy, StrategyPlan
        class NullStrategist(SA):
            def __call__(self, state, **kw):
                return StrategyPlan(
                    strategy=ProofStrategy.CUSTOM, target_lemmas=[],
                    estimated_depth=2, subgoal_decomposition=[], confidence=0.5,
                )
        strategist = NullStrategist()

    if critic_agent is None:
        from src.veritas import CriticAgent as CA, CriticAssessment
        class NullCritic(CA):
            def __call__(self, state, **kw):
                return CriticAssessment(
                    value=0.5, policy_prior={}, reasoning="uniform",
                    suggested_tactics=[], pruning_recommendation=False,
                )
        critic_agent = NullCritic()

    if retriever is None:
        from src.veritas import RetrieverAgent as RA
        class NullRetriever(RA):
            def __call__(self, state, **kw):
                return []
        retriever = NullRetriever()

    search = VERITASSearch(
        strategist=strategist,
        tactician=TacticianAgent(),
        critic=critic_agent,
        retriever=retriever,
        verifier=verifier_adapter,
        config=SearchConfig(
            max_iterations=max_iterations, max_depth=10,
            num_tactic_candidates=6, exploration_constant=1.5,
            value_weight=0.3,
            intrinsic_weight=variant_config.get("intrinsic_bonus", 0.1),
            strategy_bonus=0.2, timeout_seconds=theorem_timeout,
        ),
    )

    result = search.search(theorem.statement)
    elapsed = time.time() - start
    return TheoremResult(
        name=theorem.name, category=theorem.category, method="veritas_ablation",
        solved=result.get("success", False), proof=result.get("proof"),
        elapsed_seconds=elapsed, lean_calls=verifier_adapter.call_count,
    )


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    theorems: list,
    method_fn,
    method_name: str,
    workers: int = 4,
    config: Dict[str, Any] = None,
) -> BenchmarkResult:
    print(f"\nRunning {method_name} on {len(theorems)} theorems (workers={workers})...")

    results: List[TheoremResult] = []
    solved = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(method_fn, t): t for t in theorems}
        for i, future in enumerate(as_completed(futures)):
            theorem = futures[future]
            try:
                r = future.result(timeout=300)
            except Exception as e:
                r = TheoremResult(
                    name=theorem.name, category=theorem.category, method=method_name,
                    solved=False, proof=None, elapsed_seconds=0.0, lean_calls=0,
                    failure_reason=str(e),
                )
            with lock:
                results.append(r)
                if r.solved:
                    solved += 1

            status = "✓" if r.solved else "✗"
            print(f"  [{i+1:3d}/{len(theorems)}] {status} {r.name[:40]:40s} "
                  f"({r.elapsed_seconds:.1f}s, {r.lean_calls} calls)")

    total_calls = sum(r.lean_calls for r in results)
    total_time = sum(r.elapsed_seconds for r in results)

    per_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r.category
        if cat not in per_category:
            per_category[cat] = {"total": 0, "solved": 0}
        per_category[cat]["total"] += 1
        if r.solved:
            per_category[cat]["solved"] += 1

    return BenchmarkResult(
        method=method_name,
        total=len(theorems),
        solved=solved,
        solve_rate=solved / max(1, len(theorems)),
        total_lean_calls=total_calls,
        total_time_seconds=total_time,
        per_category=per_category,
        per_theorem=[r.to_dict() for r in results],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config=config or {},
    )

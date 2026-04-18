"""
CombiBench Experiment Runner
==============================
Evaluates VERITAS and baselines on CombiBench (100 Lean 4 combinatorics
problems: IMO 2000–2024, Brualdi textbook, Hackmath, other competitions).

Usage:
    # Portfolio baseline (no LLM):
    python experiments/run_combibench.py --mode portfolio

    # Best-of-1 Claude:
    python experiments/run_combibench.py --mode best_of_1

    # Best-of-5 Claude:
    python experiments/run_combibench.py --mode best_of_5

    # VERITAS Two-Phase (main method):
    python experiments/run_combibench.py --mode veritas_two_phase

    # Limit to a subset:
    python experiments/run_combibench.py --mode portfolio --max-problems 20
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lean.lake_validator import LakeValidator
from src.lean.combibench_loader import load_combibench, CombiTheorem, get_category_breakdown
from src.veritas import (
    SearchConfig, VERITASSearch,
    StrategistAgent, TacticianAgent, CriticAgent, RetrieverAgent,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
LEAN_PROJECT = PROJECT_ROOT / "lean_project"
DATA_DIR = PROJECT_ROOT / "data" / "combibench"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Canonical eval set — all experiments must use this fixed order.
# Uses the solvable subset (55/98): excludes 43 problems with `_solution := sorry`
# (those require a pre-known answer value and are unsolvable by any tactic-based method).
EVAL_SET_FILE = DATA_DIR / "eval_set_solvable.json"

# CombiBench theorems open their own namespaces, so we avoid `open Finset` here
# to prevent Icc ambiguity with `open Set` in problem files.
COMBI_MATHLIB_HEADER = """\
import Mathlib
import Aesop

open BigOperators Real Nat Rat
"""

# ---------------------------------------------------------------------------
# Tactic Portfolio
# ---------------------------------------------------------------------------
PORTFOLIO = [
    "rfl", "norm_num", "decide", "native_decide", "omega",
    "ring", "simp", "aesop", "trivial", "linarith", "nlinarith",
    "positivity", "norm_cast", "push_cast", "field_simp",
    "simp; ring", "norm_num; ring",
    "intro h; simp at h ⊢", "intro n; simp", "intro n; omega",
    "intro n m; ring", "intro n m; omega",
    "simp [mul_comm]", "simp [add_comm]",
    "ring_nf; norm_num", "simp; linarith", "simp; omega",
    "intro h; linarith", "constructor <;> norm_num",
    "simp only [ne_eq]; norm_num",
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
            print(f"  {cat:25s}: {solved:3d}/{total:3d}  ({pct:.0%})")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Portfolio Baseline
# ---------------------------------------------------------------------------

class PortfolioBaseline:
    def __init__(self, validator: LakeValidator, portfolio: List[str] = None):
        self.validator = validator
        self.portfolio = portfolio or PORTFOLIO
        self._lock = threading.Lock()
        self._call_count = 0

    def prove(self, theorem: CombiTheorem, timeout: int = 60) -> TheoremResult:
        start = time.time()
        winning_idx = self.validator.validate_portfolio(
            theorem_statement=theorem.statement,
            tactics=self.portfolio,
            timeout=timeout,
            extra_imports=theorem.extra_imports,
            lean_header=COMBI_MATHLIB_HEADER,
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
        n: int = 5,
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

    def _generate_tactics(self, theorem: CombiTheorem) -> List[str]:
        import re

        # Include preamble context (custom defs, opens, etc.)
        preamble_context = ""
        if theorem.preamble.strip():
            preamble_context = f"\nPreamble (already in scope):\n```lean\n{theorem.preamble}\n```\n"

        prompt = f"""Prove the following Lean 4 combinatorics theorem. Suggest {self.n} different proof attempts.
{preamble_context}
Theorem to prove:
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

        # Try 1: JSON array in ```json fence
        m = re.search(r'```json\s*(\[.*?\])\s*```', raw, re.DOTALL)
        if m:
            try:
                tactics = json.loads(m.group(1))
                result = [t for t in tactics if isinstance(t, str) and "sorry" not in t.lower()]
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Try 2: bare JSON array anywhere in response
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            try:
                tactics = json.loads(m.group(0))
                result = [t for t in tactics if isinstance(t, str) and "sorry" not in t.lower()]
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Try 3: quoted strings (Claude returned prose with "tactic" mentions)
        fallback = re.findall(r'"([^"\n]{1,120})"', raw)
        result = [t for t in fallback if "sorry" not in t.lower()]
        if result:
            return result[:self.n]

        # Try 4: backtick-fenced tactics (e.g. `ring`)
        bt = re.findall(r'`([^`\n]{1,80})`', raw)
        result = [t for t in bt if "sorry" not in t.lower() and len(t.split()) <= 10]
        if result:
            return result[:self.n]

        return []

    def prove(self, theorem: CombiTheorem, timeout: int = 120, batch_size: int = 50) -> TheoremResult:
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
                extra_imports=theorem.extra_imports,
                lean_header=COMBI_MATHLIB_HEADER,
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
# VERITAS Two-Phase
# ---------------------------------------------------------------------------

class VERITASVerifierAdapter:
    """Wraps LakeValidator with extra_imports support for MCTS."""

    def __init__(self, validator: LakeValidator, extra_imports: str = ""):
        self._v = validator
        self._extra_imports = extra_imports
        self.call_count = 0
        self._lock = threading.Lock()

    def validate(self, theorem: str, proof_steps: List[str], context: str = None):
        with self._lock:
            self.call_count += 1
        result = self._v.validate(
            theorem_statement=theorem,
            tactics=proof_steps,
            extra_imports=self._extra_imports,
            lean_header=COMBI_MATHLIB_HEADER,
        )
        return _LegacyResult(result)

    def validate_batch(
        self,
        theorem: str,
        prefix_tactics: List[str],
        candidate_tactics: List[str],
    ) -> List["_LegacyResult"]:
        """Validate N candidates on top of prefix in one Lean call."""
        with self._lock:
            self.call_count += 1
        results = self._v.validate_batch_expansion(
            theorem_statement=theorem,
            prefix_tactics=prefix_tactics,
            candidate_tactics=candidate_tactics,
            extra_imports=self._extra_imports,
            lean_header=COMBI_MATHLIB_HEADER,
        )
        return [_LegacyResult(r) for r in results]


class _LegacyResult:
    def __init__(self, r):
        self.signal_A_syntax = r.signal_A_syntax
        self.signal_B_typecheck = r.signal_B_typecheck
        self.signal_C_progress = r.signal_C_progress
        self.signal_D_complete = r.signal_D_complete
        self.remaining_goal = r.remaining_goal
        self.hypotheses = r.hypotheses
        self.subgoals = r.subgoals
        self.stderr = r.stderr
        self.success = r.success


def run_veritas_two_phase(
    theorem: CombiTheorem,
    validator: LakeValidator,
    max_iterations: int = 50,
    sweep_n: int = 5,
    theorem_timeout: int = 300,
) -> TheoremResult:
    """VERITAS Two-Phase on CombiBench theorem."""
    from src.agents.claude_tactician import ClaudeTacticianAgent, ClaudeCriticAgent

    start = time.time()

    # Phase 1: Best-of-N sweep
    sweeper = BestOfNClaude(validator, n=sweep_n)
    phase1 = sweeper.prove(theorem, timeout=90)
    phase1_elapsed = time.time() - start

    if phase1.solved:
        return TheoremResult(
            name=theorem.name, category=theorem.category,
            method="veritas_two_phase", solved=True, proof=phase1.proof,
            elapsed_seconds=phase1_elapsed, lean_calls=phase1.lean_calls,
        )

    failed_tactics = sweeper.last_tactics
    remaining_timeout = theorem_timeout - phase1_elapsed

    if remaining_timeout < 30:
        return TheoremResult(
            name=theorem.name, category=theorem.category,
            method="veritas_two_phase", solved=False, proof=None,
            elapsed_seconds=time.time() - start, lean_calls=phase1.lean_calls,
            failure_reason="phase1_exhausted_budget",
        )

    # Phase 2: MCTS
    tactician = ClaudeTacticianAgent(phase1_failed_tactics=failed_tactics)
    critic = ClaudeCriticAgent()
    verifier_adapter = VERITASVerifierAdapter(validator, extra_imports=theorem.extra_imports)

    search = VERITASSearch(
        strategist=StrategistAgent(),
        tactician=tactician,
        critic=critic,
        retriever=RetrieverAgent(),
        verifier=verifier_adapter,
        config=SearchConfig(
            max_iterations=max_iterations,
            max_depth=10,
            num_tactic_candidates=6,
            exploration_constant=1.5,
            value_weight=0.3,
            intrinsic_weight=0.1,
            strategy_bonus=0.2,
            timeout_seconds=remaining_timeout,
        ),
    )

    mcts_result = search.search(theorem.statement)
    elapsed = time.time() - start
    total_lean_calls = phase1.lean_calls + verifier_adapter.call_count
    mcts_stats = mcts_result.get("statistics", {})

    return TheoremResult(
        name=theorem.name, category=theorem.category,
        method="veritas_two_phase",
        solved=mcts_result.get("success", False),
        proof=mcts_result.get("proof"),
        elapsed_seconds=elapsed, lean_calls=total_lean_calls,
        best_signal_A=mcts_stats.get("best_signal_A", 0.0),
        best_signal_B=mcts_stats.get("best_signal_B", 0.0),
        best_signal_C=mcts_stats.get("best_signal_C", 0.0),
        best_signal_D=mcts_stats.get("best_signal_D", 0.0),
    )


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    theorems: List[CombiTheorem],
    method_fn,
    method_name: str,
    workers: int = 4,
    config: Dict[str, Any] = None,
    canonical_order: List[str] = None,
) -> BenchmarkResult:
    print(f"\nRunning {method_name} on {len(theorems)} theorems (workers={workers})...")

    results: List[TheoremResult] = []
    completed = 0
    solved = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(method_fn, t): t for t in theorems}

        for future in as_completed(futures):
            theorem = futures[future]
            try:
                r = future.result(timeout=600)
            except Exception as e:
                r = TheoremResult(
                    name=theorem.name, category=theorem.category,
                    method=method_name, solved=False, proof=None,
                    elapsed_seconds=0.0, lean_calls=0, failure_reason=str(e),
                )

            with lock:
                results.append(r)
                completed += 1
                if r.solved:
                    solved += 1

            status = "✓" if r.solved else "✗"
            print(f"  [{completed:3d}/{len(theorems)}] {status} {r.name[:45]:45s} "
                  f"({r.elapsed_seconds:.1f}s, {r.lean_calls} calls)")

    # Sort results by canonical order so all method JSONs are aligned
    if canonical_order:
        order_map = {name: i for i, name in enumerate(canonical_order)}
        results.sort(key=lambda r: order_map.get(r.name, 999999))

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


def save_result(result: BenchmarkResult, suffix: str = "") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"combibench_{result.method}_{ts}{suffix}.json"
    path = RESULTS_DIR / fname
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nResult saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="VERITAS CombiBench Experiment")
    parser.add_argument(
        "--mode",
        choices=["portfolio", "best_of_1", "best_of_5", "veritas_two_phase"],
        required=True,
    )
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of theorems to evaluate")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers")
    parser.add_argument("--theorem-timeout", type=int, default=300,
                        help="Per-theorem timeout in seconds (two-phase)")
    parser.add_argument("--sweep-n", type=int, default=5,
                        help="N for Best-of-N sweep (two-phase phase 1)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load canonical eval set (fixed order, used by all methods)
    if EVAL_SET_FILE.exists():
        with open(EVAL_SET_FILE) as f:
            eval_meta = json.load(f)
        canonical_names = eval_meta["problems"]
        print(f"Using canonical eval set: {len(canonical_names)} problems ({EVAL_SET_FILE.name})")
        # Load all problems then filter to canonical set in canonical order
        all_theorems = {t.name: t for t in load_combibench(DATA_DIR)}
        theorems = [all_theorems[n] for n in canonical_names if n in all_theorems]
    else:
        print(f"WARNING: No canonical eval set found at {EVAL_SET_FILE}. "
              f"Using dynamic load (order may vary between runs).")
        theorems = load_combibench(DATA_DIR, max_problems=args.max_problems)
        canonical_names = [t.name for t in theorems]

    if args.max_problems:
        theorems = theorems[:args.max_problems]
        canonical_names = canonical_names[:args.max_problems]

    print(f"Loaded {len(theorems)} theorems")
    print(f"Category breakdown: {get_category_breakdown(theorems)}")

    validator = LakeValidator(project_dir=str(LEAN_PROJECT), timeout=60, first_call_timeout=120)

    if args.mode == "portfolio":
        baseline = PortfolioBaseline(validator)
        result = run_benchmark(
            theorems,
            method_fn=lambda t: baseline.prove(t),
            method_name="portfolio",
            workers=args.workers,
            config={"portfolio_size": len(baseline.portfolio)},
            canonical_order=canonical_names,
        )

    elif args.mode == "best_of_1":
        bon = BestOfNClaude(validator, n=1)
        result = run_benchmark(
            theorems,
            method_fn=lambda t: bon.prove(t),
            method_name="best_of_1_claude",
            workers=args.workers,
            config={"n": 1, "model": bon.model},
            canonical_order=canonical_names,
        )

    elif args.mode == "best_of_5":
        bon = BestOfNClaude(validator, n=5)
        result = run_benchmark(
            theorems,
            method_fn=lambda t: bon.prove(t),
            method_name="best_of_5_claude",
            workers=args.workers,
            config={"n": 5, "model": bon.model},
            canonical_order=canonical_names,
        )

    elif args.mode == "veritas_two_phase":
        result = run_benchmark(
            theorems,
            method_fn=lambda t: run_veritas_two_phase(
                t, validator,
                sweep_n=args.sweep_n,
                theorem_timeout=args.theorem_timeout,
            ),
            method_name="veritas_two_phase",
            workers=args.workers,
            config={
                "sweep_n": args.sweep_n,
                "theorem_timeout": args.theorem_timeout,
            },
            canonical_order=canonical_names,
        )

    result.print_summary()
    save_result(result)


if __name__ == "__main__":
    main()

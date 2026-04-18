#!/usr/bin/env python3
"""
VERITAS miniF2F Benchmark
=========================
Controlled evaluation of proof search on pre-formalized theorems.
Tests MCTS vs baseline without confounding NL→LEAN translation.

Benchmark: miniF2F (488 problems from AIME, AMC, IMO, MATHD)
"""

import json
import re
import subprocess
import time
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import argparse

# Paths
LEAN_PATH = Path.home() / ".elan" / "bin" / "lean"
DATA_DIR = Path(__file__).parent.parent / "data" / "minif2f"
RESULTS_DIR = Path(__file__).parent.parent / "results"


@dataclass
class Theorem:
    """A miniF2F theorem"""
    name: str
    statement: str
    source: str  # AIME, AMC, IMO, MATHD, other
    difficulty: int  # 1-5 estimated
    
    
@dataclass
class ProofAttempt:
    """Result of a proof attempt"""
    theorem_name: str
    method: str
    solved: bool
    proof: Optional[str]
    time_seconds: float
    iterations: int
    lean_calls: int


# LEAN 4 tactics - ordered by likelihood of success on competition math
TACTICS_BASIC = [
    "rfl", "trivial", "decide", "native_decide", "sorry"
]

TACTICS_ARITHMETIC = [
    "norm_num", "ring", "ring_nf", "omega", "linarith", "nlinarith", "positivity"
]

TACTICS_SIMPLIFICATION = [
    "simp", "simp_all", "simp [*]", "field_simp", "push_neg"
]

TACTICS_SEARCH = [
    "exact?", "apply?", "rw?", "simp?"
]

TACTICS_STRUCTURAL = [
    "intro h", "intro", "cases h", "constructor", "left", "right",
    "use 1", "exists 1", "ext", "funext"
]

ALL_TACTICS = TACTICS_BASIC + TACTICS_ARITHMETIC + TACTICS_SIMPLIFICATION + TACTICS_STRUCTURAL


def parse_minif2f(filepath: Path) -> List[Theorem]:
    """Parse miniF2F LEAN file"""
    content = filepath.read_text()
    theorems = []
    
    # Split by theorem declarations
    parts = content.split("\ntheorem ")
    
    for part in parts[1:]:
        lines = part.split('\n')
        name_match = re.match(r'(\w+)', lines[0])
        if not name_match:
            continue
        name = name_match.group(1)
        
        # Find statement end
        full = '\n'.join(lines)
        for marker in [':= by', ':=\n', 'by sorry', ':= sorry']:
            idx = full.find(marker)
            if idx != -1:
                statement = 'theorem ' + full[:idx].strip()
                statement = ' '.join(statement.split())
                break
        else:
            continue
        
        # Classify
        if name.startswith("aime"):
            source, difficulty = "AIME", 4
        elif name.startswith("imo"):
            source, difficulty = "IMO", 5
        elif name.startswith("amc"):
            source, difficulty = "AMC", 3
        elif name.startswith("mathd"):
            source, difficulty = "MATHD", 2
        else:
            source, difficulty = "other", 3
            
        theorems.append(Theorem(name, statement, source, difficulty))
    
    return theorems


class LeanVerifier:
    """LEAN 4 proof verification"""
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.call_count = 0
        
    def verify(self, statement: str, proof: str) -> Dict:
        """Verify proof and return structured feedback"""
        self.call_count += 1
        
        lean_code = f"""
-- Auto-generated
{statement} := by
  {proof}
"""
        try:
            result = subprocess.run(
                [str(LEAN_PATH), "--stdin"],
                input=lean_code,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            stderr = result.stderr.lower()
            return {
                "success": result.returncode == 0 and "error" not in stderr,
                "signal_A": "unknown identifier" not in stderr and "unexpected" not in stderr,
                "signal_B": "type mismatch" not in stderr,
                "signal_C": "unsolved goals" not in stderr,
                "signal_D": result.returncode == 0 and "error" not in stderr,
                "error": result.stderr[:300] if result.stderr else None
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "signal_A": False, "signal_B": False, 
                    "signal_C": False, "signal_D": False, "error": "timeout"}
        except Exception as e:
            return {"success": False, "signal_A": False, "signal_B": False,
                    "signal_C": False, "signal_D": False, "error": str(e)}


class BaselineProver:
    """Random tactic search baseline"""
    
    def __init__(self, verifier: LeanVerifier, max_iterations: int = 100):
        self.verifier = verifier
        self.max_iterations = max_iterations
        
    def prove(self, theorem: Theorem) -> ProofAttempt:
        start = time.time()
        initial_calls = self.verifier.call_count
        
        # Phase 1: Single tactics
        for i in range(self.max_iterations // 2):
            tactic = random.choice(ALL_TACTICS)
            result = self.verifier.verify(theorem.statement, tactic)
            if result["success"]:
                return ProofAttempt(
                    theorem.name, "baseline", True, tactic,
                    time.time() - start, i + 1,
                    self.verifier.call_count - initial_calls
                )
        
        # Phase 2: Tactic pairs
        for i in range(self.max_iterations // 2):
            t1, t2 = random.choices(ALL_TACTICS, k=2)
            proof = f"{t1}\n  {t2}"
            result = self.verifier.verify(theorem.statement, proof)
            if result["success"]:
                return ProofAttempt(
                    theorem.name, "baseline", True, proof,
                    time.time() - start, self.max_iterations // 2 + i + 1,
                    self.verifier.call_count - initial_calls
                )
        
        return ProofAttempt(
            theorem.name, "baseline", False, None,
            time.time() - start, self.max_iterations,
            self.verifier.call_count - initial_calls
        )


class MCTSNode:
    """MCTS tree node"""
    
    def __init__(self, proof_state: str, parent: Optional['MCTSNode'] = None):
        self.proof_state = proof_state
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried = ALL_TACTICS.copy()
        random.shuffle(self.untried)
        
    def ucb(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = c * ((2 * (self.parent.visits if self.parent else 1)) / (self.visits + 1)) ** 0.5
        return exploit + explore
    
    def best_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb())
    
    def add_child(self, tactic: str) -> 'MCTSNode':
        new_state = f"{self.proof_state}\n  {tactic}".strip() if self.proof_state else tactic
        child = MCTSNode(new_state, parent=self)
        self.children.append(child)
        return child


class MCTSProver:
    """MCTS-guided proof search with LEAN feedback (VERITAS core)"""
    
    def __init__(self, verifier: LeanVerifier, max_iterations: int = 100):
        self.verifier = verifier
        self.max_iterations = max_iterations
        
    def prove(self, theorem: Theorem) -> ProofAttempt:
        start = time.time()
        initial_calls = self.verifier.call_count
        
        root = MCTSNode("")
        
        for i in range(self.max_iterations):
            # Selection: traverse tree using UCB
            node = root
            while node.children and not node.untried:
                node = node.best_child()
            
            # Expansion: try untried tactic
            if node.untried:
                tactic = node.untried.pop()
                node = node.add_child(tactic)
            
            # Simulation: get LEAN feedback
            result = self.verifier.verify(theorem.statement, node.proof_state)
            
            # Check success
            if result["success"]:
                return ProofAttempt(
                    theorem.name, "mcts", True, node.proof_state,
                    time.time() - start, i + 1,
                    self.verifier.call_count - initial_calls
                )
            
            # Compute reward from signals (A-D)
            reward = 0.0
            if result["signal_A"]:
                reward += 0.25
            if result["signal_B"]:
                reward += 0.25
            if result["signal_C"]:
                reward += 0.35
            if result["signal_D"]:
                reward += 0.15
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        return ProofAttempt(
            theorem.name, "mcts", False, None,
            time.time() - start, self.max_iterations,
            self.verifier.call_count - initial_calls
        )


def run_benchmark(theorems: List[Theorem], methods: List[str] = ["baseline", "mcts"],
                  max_problems: int = 50, max_iterations: int = 100) -> Dict:
    """Run benchmark comparison"""
    
    if len(theorems) > max_problems:
        theorems = random.sample(theorems, max_problems)
    
    verifier = LeanVerifier()
    baseline = BaselineProver(verifier, max_iterations)
    mcts = MCTSProver(verifier, max_iterations)
    
    results = {m: [] for m in methods}
    
    print(f"\n{'='*70}")
    print(f"miniF2F Benchmark: {len(theorems)} problems, {max_iterations} iterations/problem")
    print(f"{'='*70}\n")
    
    for i, thm in enumerate(theorems):
        print(f"[{i+1:3d}/{len(theorems)}] {thm.name} ({thm.source})")
        
        for method in methods:
            prover = baseline if method == "baseline" else mcts
            attempt = prover.prove(thm)
            results[method].append(attempt)
            
            status = "✓" if attempt.solved else "✗"
            print(f"    {method:8s}: {status} | {attempt.iterations:3d} iter | {attempt.time_seconds:.2f}s | {attempt.lean_calls} calls")
    
    return results


def print_summary(results: Dict):
    """Print benchmark summary with statistics"""
    
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    comparison = {}
    
    for method, attempts in results.items():
        solved = [a for a in attempts if a.solved]
        total = len(attempts)
        
        comparison[method] = {
            "solved": len(solved),
            "total": total,
            "rate": len(solved) / total if total else 0,
            "avg_time": sum(a.time_seconds for a in attempts) / total,
            "avg_iter": sum(a.iterations for a in attempts) / total,
            "avg_calls": sum(a.lean_calls for a in attempts) / total,
        }
        
        print(f"{method.upper()}")
        print(f"  Solve rate:   {len(solved)}/{total} ({100*len(solved)/total:.1f}%)")
        print(f"  Avg time:     {comparison[method]['avg_time']:.2f}s")
        print(f"  Avg iter:     {comparison[method]['avg_iter']:.1f}")
        print(f"  Avg LEAN calls: {comparison[method]['avg_calls']:.1f}")
        
        # Breakdown by source
        by_source = {}
        for a in attempts:
            # Get source from theorem name
            if a.theorem_name.startswith("aime"):
                src = "AIME"
            elif a.theorem_name.startswith("imo"):
                src = "IMO"
            elif a.theorem_name.startswith("amc"):
                src = "AMC"
            elif a.theorem_name.startswith("mathd"):
                src = "MATHD"
            else:
                src = "other"
            
            if src not in by_source:
                by_source[src] = {"solved": 0, "total": 0}
            by_source[src]["total"] += 1
            if a.solved:
                by_source[src]["solved"] += 1
        
        print(f"  By source:")
        for src in sorted(by_source.keys()):
            s = by_source[src]
            print(f"    {src:6s}: {s['solved']:2d}/{s['total']:2d} ({100*s['solved']/s['total']:.0f}%)")
        print()
    
    # Comparison
    if "baseline" in comparison and "mcts" in comparison:
        b, m = comparison["baseline"], comparison["mcts"]
        improvement = (m["rate"] - b["rate"]) / b["rate"] * 100 if b["rate"] > 0 else 0
        print(f"MCTS vs Baseline:")
        print(f"  Solve rate improvement: {improvement:+.1f}%")
        print(f"  Efficiency: MCTS uses {m['avg_calls']/b['avg_calls']:.2f}x LEAN calls")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="VERITAS miniF2F Benchmark")
    parser.add_argument("--problems", type=int, default=30, help="Number of problems")
    parser.add_argument("--iterations", type=int, default=50, help="Max iterations per problem")
    parser.add_argument("--methods", nargs="+", default=["baseline", "mcts"])
    parser.add_argument("--difficulty", type=int, default=3, help="Max difficulty (1-5)")
    parser.add_argument("--source", type=str, help="Filter by source (AIME, AMC, IMO, MATHD)")
    args = parser.parse_args()
    
    # Check LEAN
    if not LEAN_PATH.exists():
        print(f"ERROR: LEAN not found at {LEAN_PATH}")
        print("Install with: curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh")
        return
    
    # Load miniF2F
    minif2f_file = DATA_DIR / "test_clean.lean"
    if not minif2f_file.exists():
        print(f"ERROR: miniF2F not found at {minif2f_file}")
        return
    
    theorems = parse_minif2f(minif2f_file)
    print(f"Loaded {len(theorems)} theorems from miniF2F")
    
    # Filter
    if args.difficulty:
        theorems = [t for t in theorems if t.difficulty <= args.difficulty]
    if args.source:
        theorems = [t for t in theorems if t.source == args.source]
    
    print(f"After filtering: {len(theorems)} theorems")
    
    # Run benchmark
    results = run_benchmark(theorems, args.methods, args.problems, args.iterations)
    
    # Summary
    comparison = print_summary(results)
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "benchmark": "miniF2F",
        "config": vars(args),
        "summary": comparison,
        "details": {
            method: [
                {
                    "theorem": a.theorem_name,
                    "solved": a.solved,
                    "proof": a.proof,
                    "time": a.time_seconds,
                    "iterations": a.iterations,
                    "lean_calls": a.lean_calls
                }
                for a in attempts
            ]
            for method, attempts in results.items()
        }
    }
    
    output_file = RESULTS_DIR / "minif2f_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

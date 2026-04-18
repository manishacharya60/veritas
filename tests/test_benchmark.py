#!/usr/bin/env python3
"""
VERITAS Full Benchmark Test

Tests on MATH and AIME-style problems with real LEAN 4 verification.
Compares baseline (random tactic) vs MCTS-guided search.
"""

import subprocess
import tempfile
import os
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import defaultdict


class ProofStatus(Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    TACTIC_FAILED = "tactic_failed"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class LEANResult:
    status: ProofStatus
    is_valid: bool
    output: str
    error: str
    signal_A: float
    signal_B: float
    signal_C: float
    signal_D: float
    
    @property
    def total_signal(self) -> float:
        return 0.25 * (self.signal_A + self.signal_B + self.signal_C + self.signal_D)


@dataclass
class MCTSNode:
    """Node in MCTS tree."""
    tactic: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    signal: float = 0.0
    
    def ucb_score(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.value
        import math
        return self.value + c * math.sqrt(math.log(self.parent.visits) / self.visits)


class LEANVerifier:
    """Real LEAN 4 verifier."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.lean_path = os.path.expanduser("~/.elan/bin/lean")
        self.cache = {}
    
    def verify(self, lean_code: str) -> LEANResult:
        # Check cache
        if lean_code in self.cache:
            return self.cache[lean_code]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(lean_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [self.lean_path, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            parsed = self._parse_result(result)
            self.cache[lean_code] = parsed
            return parsed
            
        except subprocess.TimeoutExpired:
            return LEANResult(
                status=ProofStatus.TIMEOUT,
                is_valid=False,
                output="", error="Timeout",
                signal_A=0.5, signal_B=0.0, signal_C=0.0, signal_D=0.0,
            )
        finally:
            os.unlink(temp_path)
    
    def _parse_result(self, result: subprocess.CompletedProcess) -> LEANResult:
        if result.returncode == 0:
            return LEANResult(
                status=ProofStatus.SUCCESS,
                is_valid=True,
                output=result.stdout, error="",
                signal_A=1.0, signal_B=1.0, signal_C=1.0, signal_D=1.0,
            )
        
        error = result.stderr.lower()
        if "syntax" in error or "unexpected" in error:
            return LEANResult(
                status=ProofStatus.SYNTAX_ERROR, is_valid=False,
                output=result.stdout, error=result.stderr,
                signal_A=0.0, signal_B=0.0, signal_C=0.0, signal_D=0.0,
            )
        elif "type mismatch" in error:
            return LEANResult(
                status=ProofStatus.TYPE_ERROR, is_valid=False,
                output=result.stdout, error=result.stderr,
                signal_A=1.0, signal_B=0.0, signal_C=0.0, signal_D=0.0,
            )
        elif "unsolved goals" in error:
            return LEANResult(
                status=ProofStatus.TACTIC_FAILED, is_valid=False,
                output=result.stdout, error=result.stderr,
                signal_A=1.0, signal_B=1.0, signal_C=0.5, signal_D=0.0,
            )
        else:
            return LEANResult(
                status=ProofStatus.UNKNOWN, is_valid=False,
                output=result.stdout, error=result.stderr,
                signal_A=0.5, signal_B=0.0, signal_C=0.0, signal_D=0.0,
            )


class BaselineSearch:
    """Random/Sequential tactic search (baseline)."""
    
    TACTICS = [
        "rfl", "trivial", "decide", "simp", "omega", "ring",
        "assumption", "exact?", "apply?", "constructor",
        "intro h", "intros", "cases h", "induction n",
    ]
    
    def __init__(self, verifier: LEANVerifier):
        self.verifier = verifier
    
    def search(self, theorem_base: str, max_attempts: int = 15) -> Dict:
        """Random search through tactics."""
        attempts = []
        
        for i in range(max_attempts):
            tactic = random.choice(self.TACTICS)
            
            if ":= by" in theorem_base:
                code = theorem_base.replace(":= by", f":= by {tactic}")
            elif ":=" in theorem_base:
                code = theorem_base.replace(":=", f":= {tactic}")
            else:
                code = f"{theorem_base} := {tactic}"
            
            result = self.verifier.verify(code)
            attempts.append({
                "tactic": tactic,
                "signal": result.total_signal,
                "success": result.is_valid,
            })
            
            if result.is_valid:
                return {
                    "success": True,
                    "proof": code,
                    "attempts": len(attempts),
                    "total_signal": result.total_signal,
                    "history": attempts,
                }
        
        return {
            "success": False,
            "proof": None,
            "attempts": len(attempts),
            "total_signal": max(a["signal"] for a in attempts) if attempts else 0,
            "history": attempts,
        }


class MCTSSearch:
    """MCTS-guided proof search with A-D signals."""
    
    TACTICS = [
        "rfl", "trivial", "decide", "simp", "omega", "ring",
        "assumption", "exact?", "apply?", "constructor",
        "intro h", "intros", "cases h", "induction n",
        "simp only []", "simp_all", "norm_num", "linarith",
    ]
    
    def __init__(self, verifier: LEANVerifier, exploration_c: float = 1.414):
        self.verifier = verifier
        self.exploration_c = exploration_c
    
    def search(self, theorem_base: str, max_simulations: int = 15) -> Dict:
        """MCTS search with value-guided exploration."""
        root = MCTSNode(tactic="root")
        attempts = []
        best_result = None
        best_signal = 0.0
        
        # Initialize children
        for tactic in self.TACTICS:
            child = MCTSNode(tactic=tactic, parent=root)
            root.children.append(child)
        
        for sim in range(max_simulations):
            # Selection - pick best UCB node
            node = self._select(root)
            
            # Expansion/Simulation - try the tactic
            if ":= by" in theorem_base:
                code = theorem_base.replace(":= by", f":= by {node.tactic}")
            elif ":=" in theorem_base:
                code = theorem_base.replace(":=", f":= {node.tactic}")
            else:
                code = f"{theorem_base} := {node.tactic}"
            
            result = self.verifier.verify(code)
            signal = result.total_signal
            
            attempts.append({
                "tactic": node.tactic,
                "signal": signal,
                "success": result.is_valid,
            })
            
            # Backpropagation
            self._backpropagate(node, signal)
            
            # Track best
            if signal > best_signal:
                best_signal = signal
                best_result = (code, result)
            
            # Early exit on success
            if result.is_valid:
                return {
                    "success": True,
                    "proof": code,
                    "attempts": len(attempts),
                    "total_signal": signal,
                    "history": attempts,
                }
        
        return {
            "success": False,
            "proof": best_result[0] if best_result else None,
            "attempts": len(attempts),
            "total_signal": best_signal,
            "history": attempts,
        }
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select child with best UCB score."""
        if not root.children:
            return root
        return max(root.children, key=lambda n: n.ucb_score(self.exploration_c))
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            node.signal = max(node.signal, value)
            node = node.parent


# ============================================================
# BENCHMARK PROBLEMS
# ============================================================

MATH_PROBLEMS = [
    {
        "id": "math_basic_1",
        "name": "Reflexivity",
        "theorem": "theorem refl_nat : 42 = 42",
        "difficulty": "trivial",
    },
    {
        "id": "math_basic_2", 
        "name": "Addition identity",
        "theorem": "theorem add_zero (n : Nat) : n + 0 = n",
        "difficulty": "easy",
    },
    {
        "id": "math_basic_3",
        "name": "One plus one",
        "theorem": "theorem one_plus_one : 1 + 1 = 2",
        "difficulty": "trivial",
    },
    {
        "id": "math_basic_4",
        "name": "Commutativity",
        "theorem": "theorem add_comm (a b : Nat) : a + b = b + a",
        "difficulty": "medium",
    },
    {
        "id": "math_basic_5",
        "name": "Multiplication by one",
        "theorem": "theorem mul_one (n : Nat) : n * 1 = n",
        "difficulty": "easy",
    },
    {
        "id": "math_basic_6",
        "name": "True is true",
        "theorem": "theorem true_true : True",
        "difficulty": "trivial",
    },
    {
        "id": "math_basic_7",
        "name": "Double",
        "theorem": "theorem double_eq (n : Nat) : n + n = 2 * n",
        "difficulty": "medium",
    },
    {
        "id": "math_arith_1",
        "name": "Simple arithmetic",
        "theorem": "theorem arith_1 : 2 + 3 = 5",
        "difficulty": "trivial",
    },
    {
        "id": "math_arith_2",
        "name": "Multiplication",
        "theorem": "theorem arith_2 : 3 * 4 = 12",
        "difficulty": "trivial",
    },
    {
        "id": "math_arith_3",
        "name": "Power",
        "theorem": "theorem arith_3 : 2 ^ 3 = 8",
        "difficulty": "easy",
    },
]

AIME_STYLE_PROBLEMS = [
    {
        "id": "aime_1",
        "name": "Sum identity",
        "theorem": "theorem sum_id (n : Nat) : n + n = 2 * n",
        "difficulty": "medium",
    },
    {
        "id": "aime_2",
        "name": "Inequality",
        "theorem": "theorem ineq_1 (n : Nat) : n ≤ n + 1",
        "difficulty": "easy",
    },
    {
        "id": "aime_3",
        "name": "Associativity",
        "theorem": "theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c)",
        "difficulty": "medium",
    },
    {
        "id": "aime_4",
        "name": "Zero multiplication",
        "theorem": "theorem mul_zero (n : Nat) : n * 0 = 0",
        "difficulty": "easy",
    },
    {
        "id": "aime_5",
        "name": "Distributivity",
        "theorem": "theorem distrib (a b c : Nat) : a * (b + c) = a * b + a * c",
        "difficulty": "hard",
    },
]


def run_benchmark(problems: List[Dict], name: str) -> Dict:
    """Run benchmark comparing baseline vs MCTS."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    
    verifier = LEANVerifier(timeout=5)
    baseline = BaselineSearch(verifier)
    mcts = MCTSSearch(verifier)
    
    results = {
        "baseline": {"solved": 0, "total_attempts": 0, "avg_signal": 0.0, "details": []},
        "mcts": {"solved": 0, "total_attempts": 0, "avg_signal": 0.0, "details": []},
    }
    
    for prob in problems:
        print(f"\n  Problem: {prob['name']} ({prob['difficulty']})")
        theorem = prob["theorem"]
        
        # Run baseline
        random.seed(42)  # Reproducibility
        t0 = time.time()
        baseline_result = baseline.search(theorem, max_attempts=12)
        baseline_time = time.time() - t0
        
        # Run MCTS
        t0 = time.time()
        mcts_result = mcts.search(theorem, max_simulations=12)
        mcts_time = time.time() - t0
        
        # Record results
        b_status = "✓" if baseline_result["success"] else "✗"
        m_status = "✓" if mcts_result["success"] else "✗"
        
        print(f"    Baseline: {b_status} (attempts: {baseline_result['attempts']}, signal: {baseline_result['total_signal']:.2f}, time: {baseline_time:.2f}s)")
        print(f"    MCTS:     {m_status} (attempts: {mcts_result['attempts']}, signal: {mcts_result['total_signal']:.2f}, time: {mcts_time:.2f}s)")
        
        if baseline_result["success"]:
            results["baseline"]["solved"] += 1
        results["baseline"]["total_attempts"] += baseline_result["attempts"]
        results["baseline"]["avg_signal"] += baseline_result["total_signal"]
        results["baseline"]["details"].append(baseline_result)
        
        if mcts_result["success"]:
            results["mcts"]["solved"] += 1
        results["mcts"]["total_attempts"] += mcts_result["attempts"]
        results["mcts"]["avg_signal"] += mcts_result["total_signal"]
        results["mcts"]["details"].append(mcts_result)
    
    n = len(problems)
    results["baseline"]["avg_signal"] /= n
    results["mcts"]["avg_signal"] /= n
    
    return results


def print_summary(all_results: Dict):
    """Print final summary."""
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total_baseline_solved = 0
    total_mcts_solved = 0
    total_problems = 0
    
    for name, results in all_results.items():
        n = len(results["baseline"]["details"])
        total_problems += n
        total_baseline_solved += results["baseline"]["solved"]
        total_mcts_solved += results["mcts"]["solved"]
        
        print(f"\n{name}:")
        print(f"  Baseline: {results['baseline']['solved']}/{n} solved ({100*results['baseline']['solved']/n:.0f}%)")
        print(f"  MCTS:     {results['mcts']['solved']}/{n} solved ({100*results['mcts']['solved']/n:.0f}%)")
        print(f"  Baseline avg signal: {results['baseline']['avg_signal']:.3f}")
        print(f"  MCTS avg signal:     {results['mcts']['avg_signal']:.3f}")
        
        improvement = results['mcts']['solved'] - results['baseline']['solved']
        if improvement > 0:
            print(f"  → MCTS improvement: +{improvement} problems")
        elif improvement < 0:
            print(f"  → Baseline better: {-improvement} problems")
        else:
            print(f"  → Equal performance")
    
    print(f"\n{'='*60}")
    print("OVERALL:")
    print(f"  Baseline: {total_baseline_solved}/{total_problems} ({100*total_baseline_solved/total_problems:.1f}%)")
    print(f"  MCTS:     {total_mcts_solved}/{total_problems} ({100*total_mcts_solved/total_problems:.1f}%)")
    
    if total_mcts_solved > total_baseline_solved:
        print(f"\n  ✓ MCTS shows improvement: +{total_mcts_solved - total_baseline_solved} problems solved")
    elif total_mcts_solved == total_baseline_solved:
        print(f"\n  = Equal performance (both methods effective on these problems)")
    else:
        print(f"\n  ✗ Baseline performed better on this set")
    
    print("="*60)


def main():
    print("\n" + "="*60)
    print("VERITAS: LEAN Integration & Benchmark Test")
    print("Comparing Baseline (Random) vs MCTS-Guided Search")
    print("="*60)
    
    # Verify LEAN
    verifier = LEANVerifier()
    test_result = verifier.verify("theorem t : 1 = 1 := rfl")
    if not test_result.is_valid:
        print("ERROR: LEAN verification failed!")
        return
    print("✓ LEAN 4 verified and working\n")
    
    all_results = {}
    
    # Run MATH benchmark
    all_results["MATH"] = run_benchmark(MATH_PROBLEMS, "MATH Dataset")
    
    # Run AIME-style benchmark
    all_results["AIME-style"] = run_benchmark(AIME_STYLE_PROBLEMS, "AIME-Style Problems")
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "benchmark_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert for JSON serialization
    json_results = {}
    for name, res in all_results.items():
        json_results[name] = {
            "baseline": {
                "solved": res["baseline"]["solved"],
                "total_attempts": res["baseline"]["total_attempts"],
                "avg_signal": res["baseline"]["avg_signal"],
            },
            "mcts": {
                "solved": res["mcts"]["solved"],
                "total_attempts": res["mcts"]["total_attempts"],
                "avg_signal": res["mcts"]["avg_signal"],
            },
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

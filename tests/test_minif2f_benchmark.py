#!/usr/bin/env python3
"""
VERITAS Benchmark Runner for miniF2F
=====================================
Runs theorem proving benchmarks using real miniF2F problems (AIME, AMC, IMO formalized in LEAN 4)
"""

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import random
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LEAN_PATH = Path.home() / ".elan" / "bin" / "lean"
MINIF2F_PATH = Path(__file__).parent.parent / "data" / "minif2f" / "test_clean.lean"


@dataclass
class MinIF2FProblem:
    """A miniF2F theorem to prove"""
    name: str
    statement: str
    source: str  # aime, amc, imo, mathd, etc.
    difficulty: int  # estimated 1-5


@dataclass 
class ProofResult:
    """Result of attempting to prove a theorem"""
    problem: MinIF2FProblem
    solved: bool
    proof: Optional[str]
    time_seconds: float
    attempts: int
    method: str  # "baseline" or "mcts"


def parse_minif2f_theorems(filepath: Path) -> list[MinIF2FProblem]:
    """Parse miniF2F LEAN file to extract theorem statements"""
    content = filepath.read_text()
    
    problems = []
    
    # Split by "theorem " and process each
    parts = content.split("\ntheorem ")
    
    for part in parts[1:]:  # Skip first (header)
        lines = part.split('\n')
        
        # Get theorem name
        first_line = lines[0]
        name_match = re.match(r'(\w+)', first_line)
        if not name_match:
            continue
        name = name_match.group(1)
        
        # Find the full statement (until := or by)
        full_text = '\n'.join(lines)
        
        # Find where the proof starts
        proof_start = None
        for marker in [':= by', ':=\n', 'by sorry', ':= sorry']:
            idx = full_text.find(marker)
            if idx != -1:
                if proof_start is None or idx < proof_start:
                    proof_start = idx
        
        if proof_start is None:
            continue
            
        statement = 'theorem ' + full_text[:proof_start].strip()
        # Clean up whitespace
        statement = ' '.join(statement.split())
        
        # Determine source from name
        if name.startswith("aime"):
            source = "AIME"
            difficulty = 4
        elif name.startswith("imo"):
            source = "IMO"
            difficulty = 5
        elif name.startswith("amc"):
            source = "AMC"
            difficulty = 3
        elif name.startswith("mathd"):
            source = "MATHD"
            difficulty = 2
        else:
            source = "other"
            difficulty = 3
        
        problems.append(MinIF2FProblem(
            name=name,
            statement=statement,
            source=source,
            difficulty=difficulty
        ))
    
    return problems


# Standard LEAN 4 tactics to try
TACTICS = [
    "rfl",
    "trivial", 
    "decide",
    "native_decide",
    "simp",
    "ring",
    "omega",
    "norm_num",
    "linarith",
    "nlinarith",
    "field_simp",
    "positivity",
    "simp_all",
    "exact?",
    "simp [*]",
    "ring_nf",
    "norm_num [*]",
]


def verify_lean_proof(statement: str, proof: str) -> tuple[bool, str]:
    """Verify a proof using LEAN 4"""
    lean_code = f"""
-- Auto-generated proof verification
{statement} := by
  {proof}
"""
    
    try:
        result = subprocess.run(
            [str(LEAN_PATH), "--stdin"],
            input=lean_code,
            capture_output=True,
            text=True,
            timeout=30
        )
        success = result.returncode == 0 and "error" not in result.stderr.lower()
        return success, result.stderr
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def baseline_prove(problem: MinIF2FProblem, max_attempts: int = 50) -> ProofResult:
    """Baseline: Random tactic search"""
    start = time.time()
    attempts = 0
    
    # Try single tactics
    for _ in range(max_attempts):
        attempts += 1
        tactic = random.choice(TACTICS)
        success, _ = verify_lean_proof(problem.statement, tactic)
        if success:
            return ProofResult(
                problem=problem,
                solved=True,
                proof=tactic,
                time_seconds=time.time() - start,
                attempts=attempts,
                method="baseline"
            )
    
    # Try tactic combinations
    for _ in range(max_attempts):
        attempts += 1
        t1, t2 = random.sample(TACTICS, 2)
        proof = f"{t1}\n  {t2}"
        success, _ = verify_lean_proof(problem.statement, proof)
        if success:
            return ProofResult(
                problem=problem,
                solved=True,
                proof=proof,
                time_seconds=time.time() - start,
                attempts=attempts,
                method="baseline"
            )
    
    return ProofResult(
        problem=problem,
        solved=False,
        proof=None,
        time_seconds=time.time() - start,
        attempts=attempts,
        method="baseline"
    )


class MCTSNode:
    """MCTS node for proof search"""
    def __init__(self, state: str, parent=None):
        self.state = state  # Current proof script
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_tactics = TACTICS.copy()
        random.shuffle(self.untried_tactics)
    
    def ucb_score(self, exploration=1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = exploration * (2 * (self.parent.visits + 1) / (self.visits + 1)) ** 0.5
        return exploit + explore
    
    def best_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.ucb_score())


def get_lean_feedback(statement: str, proof: str) -> dict:
    """Get structured feedback from LEAN"""
    lean_code = f"""
{statement} := by
  {proof}
"""
    
    try:
        result = subprocess.run(
            [str(LEAN_PATH), "--stdin"],
            input=lean_code,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        stderr = result.stderr.lower()
        return {
            "success": result.returncode == 0 and "error" not in stderr,
            "syntax_ok": "unknown identifier" not in stderr and "unexpected" not in stderr,
            "type_ok": "type mismatch" not in stderr,
            "goals_remain": "unsolved goals" in stderr,
            "raw": result.stderr[:200]
        }
    except:
        return {"success": False, "syntax_ok": False, "type_ok": False, "goals_remain": True, "raw": ""}


def mcts_prove(problem: MinIF2FProblem, max_iterations: int = 100) -> ProofResult:
    """MCTS-guided proof search with LEAN feedback"""
    start = time.time()
    
    root = MCTSNode("")
    best_proof = None
    iterations = 0
    
    for _ in range(max_iterations):
        iterations += 1
        
        # Selection: traverse to leaf using UCB
        node = root
        while node.children and not node.untried_tactics:
            node = node.best_child()
        
        # Expansion: try an untried tactic
        if node.untried_tactics:
            tactic = node.untried_tactics.pop()
            new_state = f"{node.state}\n  {tactic}".strip() if node.state else tactic
            child = MCTSNode(new_state, parent=node)
            node.children.append(child)
            node = child
        
        # Simulation: get LEAN feedback
        feedback = get_lean_feedback(problem.statement, node.state)
        
        # Check if solved
        if feedback["success"]:
            return ProofResult(
                problem=problem,
                solved=True,
                proof=node.state,
                time_seconds=time.time() - start,
                attempts=iterations,
                method="mcts"
            )
        
        # Compute reward based on LEAN signals (A-D from VERITAS)
        reward = 0.0
        if feedback["syntax_ok"]:
            reward += 0.3  # Signal A
        if feedback["type_ok"]:
            reward += 0.3  # Signal B
        if not feedback["goals_remain"]:
            reward += 0.3  # Signal C (progress)
        
        # Backpropagation
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    return ProofResult(
        problem=problem,
        solved=False,
        proof=None,
        time_seconds=time.time() - start,
        attempts=iterations,
        method="mcts"
    )


def run_benchmark(problems: list[MinIF2FProblem], method: str = "both", 
                  max_problems: int = 50) -> dict:
    """Run benchmark on a set of problems"""
    
    # Sample problems if too many
    if len(problems) > max_problems:
        problems = random.sample(problems, max_problems)
    
    results = {"baseline": [], "mcts": []}
    
    print(f"\n{'='*60}")
    print(f"Running benchmark on {len(problems)} miniF2F problems")
    print(f"{'='*60}\n")
    
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem.name} ({problem.source}, difficulty={problem.difficulty})")
        
        if method in ["baseline", "both"]:
            result = baseline_prove(problem)
            results["baseline"].append(result)
            status = "✓" if result.solved else "✗"
            print(f"  Baseline: {status} ({result.attempts} attempts, {result.time_seconds:.2f}s)")
        
        if method in ["mcts", "both"]:
            result = mcts_prove(problem)
            results["mcts"].append(result)
            status = "✓" if result.solved else "✗"
            print(f"  MCTS:     {status} ({result.attempts} attempts, {result.time_seconds:.2f}s)")
    
    return results


def print_summary(results: dict):
    """Print benchmark summary"""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    for method, method_results in results.items():
        if not method_results:
            continue
            
        solved = sum(1 for r in method_results if r.solved)
        total = len(method_results)
        avg_time = sum(r.time_seconds for r in method_results) / total
        avg_attempts = sum(r.attempts for r in method_results) / total
        
        print(f"{method.upper()}:")
        print(f"  Solved: {solved}/{total} ({100*solved/total:.1f}%)")
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Avg attempts: {avg_attempts:.1f}")
        
        # Breakdown by source
        by_source = {}
        for r in method_results:
            src = r.problem.source
            if src not in by_source:
                by_source[src] = {"solved": 0, "total": 0}
            by_source[src]["total"] += 1
            if r.solved:
                by_source[src]["solved"] += 1
        
        print(f"  By source:")
        for src, stats in sorted(by_source.items()):
            pct = 100 * stats["solved"] / stats["total"]
            print(f"    {src}: {stats['solved']}/{stats['total']} ({pct:.1f}%)")
        print()


def main():
    """Main benchmark entry point"""
    print("VERITAS miniF2F Benchmark")
    print("=" * 40)
    
    # Check LEAN
    if not LEAN_PATH.exists():
        print(f"ERROR: LEAN not found at {LEAN_PATH}")
        return
    
    # Load problems
    if not MINIF2F_PATH.exists():
        print(f"ERROR: miniF2F not found at {MINIF2F_PATH}")
        return
    
    problems = parse_minif2f_theorems(MINIF2F_PATH)
    print(f"Loaded {len(problems)} theorems from miniF2F")
    
    # Filter by difficulty for quick test
    easy_problems = [p for p in problems if p.difficulty <= 3]
    print(f"Found {len(easy_problems)} easier problems (MATHD, AMC)")
    
    # Run benchmark
    results = run_benchmark(easy_problems, method="both", max_problems=20)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "minif2f_benchmark.json"
    output_path.parent.mkdir(exist_ok=True)
    
    serializable = {
        method: [
            {
                "name": r.problem.name,
                "source": r.problem.source,
                "solved": r.solved,
                "proof": r.proof,
                "time": r.time_seconds,
                "attempts": r.attempts
            }
            for r in method_results
        ]
        for method, method_results in results.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

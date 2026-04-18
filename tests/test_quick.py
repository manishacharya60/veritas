#!/usr/bin/env python3
"""
VERITAS Quick Benchmark - Fast tests with timeout protection.
"""

import subprocess
import tempfile
import os
import json
import time
import random
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Result:
    is_valid: bool
    signal: float


class LEANVerifier:
    def __init__(self, timeout: int = 3):
        self.timeout = timeout
        self.lean_path = os.path.expanduser("~/.elan/bin/lean")
        self.cache = {}
    
    def verify(self, code: str) -> Result:
        if code in self.cache:
            return self.cache[code]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [self.lean_path, temp_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            is_valid = result.returncode == 0
            signal = 1.0 if is_valid else 0.1
            res = Result(is_valid=is_valid, signal=signal)
            self.cache[code] = res
            return res
        except subprocess.TimeoutExpired:
            return Result(is_valid=False, signal=0.05)
        finally:
            os.unlink(temp_path)


# Fast tactics that work in LEAN 4
FAST_TACTICS = ["rfl", "trivial", "decide", "omega", "simp", "native_decide", "ring"]

PROBLEMS = [
    ("m1", "42 = 42", "theorem t : 42 = 42"),
    ("m2", "1 + 1 = 2", "theorem t : 1 + 1 = 2"),
    ("m3", "n + 0 = n", "theorem t (n : Nat) : n + 0 = n"),
    ("m4", "True", "theorem t : True"),
    ("m5", "2 * 3 = 6", "theorem t : 2 * 3 = 6"),
    ("m6", "5 = 5", "theorem t : 5 = 5"),
    ("m7", "0 + 1 = 1", "theorem t : 0 + 1 = 1"),
    ("m8", "n * 0 = 0", "theorem t (n : Nat) : n * 0 = 0"),
    ("m9", "100 = 100", "theorem t : 100 = 100"),
    ("m10", "n + m = m + n", "theorem t (n m : Nat) : n + m = m + n"),
]


def baseline_search(verifier, theorem: str, max_tries: int = 7) -> Dict:
    """Random search."""
    tactics = FAST_TACTICS.copy()
    random.shuffle(tactics)
    
    for i, tactic in enumerate(tactics[:max_tries]):
        code = f"{theorem} := by {tactic}"
        result = verifier.verify(code)
        if result.is_valid:
            return {"success": True, "attempts": i + 1, "tactic": tactic}
    
    return {"success": False, "attempts": max_tries, "tactic": None}


def mcts_search(verifier, theorem: str, max_tries: int = 7) -> Dict:
    """UCB-guided search."""
    tactics = FAST_TACTICS.copy()
    visits = {t: 0 for t in tactics}
    values = {t: 0.0 for t in tactics}
    total_visits = 0
    
    for i in range(max_tries):
        # UCB selection
        best_tactic = None
        best_ucb = -1
        
        for t in tactics:
            if visits[t] == 0:
                ucb = float('inf')
            else:
                ucb = values[t] / visits[t] + 1.5 * math.sqrt(math.log(total_visits + 1) / visits[t])
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_tactic = t
        
        # Try tactic
        code = f"{theorem} := by {best_tactic}"
        result = verifier.verify(code)
        
        # Update
        visits[best_tactic] += 1
        values[best_tactic] += result.signal
        total_visits += 1
        
        if result.is_valid:
            return {"success": True, "attempts": i + 1, "tactic": best_tactic}
    
    return {"success": False, "attempts": max_tries, "tactic": None}


def main():
    print("="*50)
    print("VERITAS Quick Benchmark")
    print("="*50)
    
    verifier = LEANVerifier(timeout=2)
    
    # Test LEAN
    test = verifier.verify("theorem t : 1 = 1 := rfl")
    if not test.is_valid:
        print("ERROR: LEAN not working")
        return
    print("✓ LEAN 4 working\n")
    
    baseline_solved = 0
    mcts_solved = 0
    baseline_attempts = []
    mcts_attempts = []
    
    print(f"{'Problem':<20} {'Baseline':<15} {'MCTS':<15} {'Winner':<10}")
    print("-"*60)
    
    for pid, name, theorem in PROBLEMS:
        random.seed(42)
        b = baseline_search(verifier, theorem)
        m = mcts_search(verifier, theorem)
        
        b_status = f"✓ ({b['attempts']})" if b['success'] else f"✗ ({b['attempts']})"
        m_status = f"✓ ({m['attempts']})" if m['success'] else f"✗ ({m['attempts']})"
        
        if b['success']:
            baseline_solved += 1
            baseline_attempts.append(b['attempts'])
        if m['success']:
            mcts_solved += 1
            mcts_attempts.append(m['attempts'])
        
        # Determine winner
        if m['success'] and not b['success']:
            winner = "MCTS"
        elif b['success'] and not m['success']:
            winner = "Baseline"
        elif m['success'] and b['success']:
            if m['attempts'] < b['attempts']:
                winner = "MCTS"
            elif b['attempts'] < m['attempts']:
                winner = "Baseline"
            else:
                winner = "Tie"
        else:
            winner = "None"
        
        print(f"{name:<20} {b_status:<15} {m_status:<15} {winner:<10}")
    
    print("-"*60)
    n = len(PROBLEMS)
    print(f"\nRESULTS:")
    print(f"  Baseline: {baseline_solved}/{n} solved ({100*baseline_solved/n:.0f}%)")
    print(f"  MCTS:     {mcts_solved}/{n} solved ({100*mcts_solved/n:.0f}%)")
    
    if baseline_attempts:
        print(f"  Baseline avg attempts: {sum(baseline_attempts)/len(baseline_attempts):.1f}")
    if mcts_attempts:
        print(f"  MCTS avg attempts:     {sum(mcts_attempts)/len(mcts_attempts):.1f}")
    
    print()
    if mcts_solved > baseline_solved:
        print(f"✓ MCTS wins: +{mcts_solved - baseline_solved} problems")
    elif baseline_solved > mcts_solved:
        print(f"✗ Baseline wins: +{baseline_solved - mcts_solved} problems")
    else:
        if mcts_attempts and baseline_attempts:
            m_avg = sum(mcts_attempts)/len(mcts_attempts)
            b_avg = sum(baseline_attempts)/len(baseline_attempts)
            if m_avg < b_avg:
                print(f"= Same problems solved, but MCTS is faster ({m_avg:.1f} vs {b_avg:.1f} attempts)")
            elif b_avg < m_avg:
                print(f"= Same problems solved, but Baseline is faster")
            else:
                print("= Equal performance")
    
    print("="*50)


if __name__ == "__main__":
    main()

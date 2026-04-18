"""
Simple Comparison Test: VERITAS vs Baseline
============================================

Compares:
1. Greedy single-shot generation (baseline)
2. VERITAS multi-agent search

Uses a small LLM (Qwen2.5-Coder-7B or similar) for efficiency.
"""

import sys
import os
import time
import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.veritas import (
    ProofState, ProofStrategy, StrategyPlan, TacticCandidate,
    CriticAssessment, RetrievedPremise, StrategistAgent, TacticianAgent,
    CriticAgent, RetrieverAgent, VERITASSearch, SearchConfig, create_veritas
)


# =============================================================================
# LEAN 4 Interface
# =============================================================================

class LEAN4Verifier:
    """Real LEAN 4 verifier interface."""
    
    def __init__(self, lean_path: str = None):
        self.lean_path = lean_path or os.path.expanduser("~/.elan/bin/lean")
        self.call_count = 0
        
    def validate(self, theorem: str, proof_steps: List[str], context: str = None) -> Any:
        """Validate proof with LEAN 4."""
        self.call_count += 1
        
        # Build LEAN code
        proof_body = "\n  ".join(proof_steps) if proof_steps else "sorry"
        lean_code = f"""
-- Auto-generated proof attempt
theorem test_theorem : {theorem} := by
  {proof_body}
"""
        
        if context:
            lean_code = context + "\n" + lean_code
        
        # Write to temp file and run LEAN
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(lean_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [self.lean_path, temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            success = result.returncode == 0 and "error" not in result.stderr.lower()
            has_sorry = "sorry" in proof_body.lower()
            
            # Create result object
            class Result:
                pass
            
            r = Result()
            r.signal_A_syntax = 1.0 if "unknown identifier" not in result.stderr else 0.0
            r.signal_B_typecheck = 1.0 if "type mismatch" not in result.stderr else 0.0
            r.signal_C_progress = 0.5 if not has_sorry else 0.0
            r.signal_D_complete = 1.0 if success and not has_sorry else 0.0
            r.remaining_goal = "" if success else theorem
            r.hypotheses = []
            r.subgoals = [] if success else [theorem]
            r.stderr = result.stderr
            r.success = success and not has_sorry
            
            return r
            
        except subprocess.TimeoutExpired:
            class Result:
                signal_A_syntax = 0.0
                signal_B_typecheck = 0.0
                signal_C_progress = 0.0
                signal_D_complete = 0.0
                remaining_goal = theorem
                hypotheses = []
                subgoals = [theorem]
                stderr = "Timeout"
                success = False
            return Result()
        finally:
            os.unlink(temp_path)


# =============================================================================
# LLM-Enhanced Agents
# =============================================================================

class LLMTacticianAgent(TacticianAgent):
    """Tactician agent enhanced with LLM generation."""
    
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_tactics(self, state: ProofState, num_candidates: int = 5) -> List[str]:
        """Generate tactics using LLM."""
        prompt = f"""You are a LEAN 4 theorem prover. Generate tactics to prove:

Theorem: {state.theorem}
Current goal: {state.goal}
Hypotheses: {', '.join(state.hypotheses) if state.hypotheses else 'None'}
Proof so far: {' '.join(state.tactics_applied) if state.tactics_applied else 'None'}

Generate {num_candidates} different LEAN 4 tactics (one per line, just the tactic):
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse tactics from response
        tactics = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                # Clean up common prefixes
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line:
                    tactics.append(line)
        
        return tactics[:num_candidates] if tactics else ['sorry']
    
    def __call__(self, state: ProofState, strategy: StrategyPlan, 
                 premises: List[RetrievedPremise] = None,
                 critic_guidance: CriticAssessment = None,
                 num_candidates: int = 5, **kwargs) -> List[TacticCandidate]:
        """Generate tactic candidates using LLM + heuristics."""
        
        # Get LLM-generated tactics
        llm_tactics = self.generate_tactics(state, num_candidates=3)
        
        # Get heuristic tactics from parent class
        heuristic_candidates = super().__call__(
            state, strategy, premises, critic_guidance, num_candidates=3
        )
        
        # Combine
        candidates = []
        seen = set()
        
        # Add LLM tactics with high prior
        for tactic in llm_tactics:
            if tactic not in seen:
                seen.add(tactic)
                candidates.append(TacticCandidate(
                    tactic=tactic,
                    reasoning="LLM-generated",
                    prior=0.8,
                    strategy_alignment=0.7,
                ))
        
        # Add heuristic tactics
        for c in heuristic_candidates:
            if c.tactic not in seen:
                seen.add(c.tactic)
                candidates.append(c)
        
        return candidates[:num_candidates]


# =============================================================================
# Baseline: Greedy Single-Shot
# =============================================================================

def baseline_greedy(theorem: str, model, tokenizer, verifier: LEAN4Verifier, 
                    num_attempts: int = 10, device="cuda") -> Dict[str, Any]:
    """Baseline: Generate proof attempts greedily."""
    
    start_time = time.time()
    
    prompt = f"""You are a LEAN 4 theorem prover. Prove this theorem with a single tactic or short proof:

Theorem: {theorem}

Provide ONLY the proof tactics (one per line, no explanation):
"""
    
    successes = []
    all_attempts = []
    
    for attempt in range(num_attempts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.8 + 0.1 * attempt,  # Increase temperature for diversity
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse tactics
        tactics = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line:
                    tactics.append(line)
        
        if not tactics:
            tactics = ['sorry']
        
        # Verify
        result = verifier.validate(theorem, tactics)
        all_attempts.append({
            'tactics': tactics,
            'success': result.success,
        })
        
        if result.success:
            successes.append(tactics)
            break
    
    return {
        'success': len(successes) > 0,
        'proof': successes[0] if successes else None,
        'attempts': len(all_attempts),
        'time': time.time() - start_time,
        'lean_calls': verifier.call_count,
    }


# =============================================================================
# VERITAS Search
# =============================================================================

def veritas_search(theorem: str, model, tokenizer, verifier: LEAN4Verifier,
                   max_iterations: int = 20, device="cuda") -> Dict[str, Any]:
    """VERITAS multi-agent search."""
    
    # Create LLM-enhanced tactician
    llm_tactician = LLMTacticianAgent(model, tokenizer, device)
    
    # Create VERITAS search
    search = VERITASSearch(
        strategist=StrategistAgent(),
        tactician=llm_tactician,
        critic=CriticAgent(),
        retriever=RetrieverAgent(),
        verifier=verifier,
        config=SearchConfig(
            max_iterations=max_iterations,
            max_depth=10,
            num_tactic_candidates=5,
            exploration_constant=1.5,
        ),
    )
    
    # Reset verifier counter
    verifier.call_count = 0
    
    # Run search
    result = search.search(theorem)
    
    return {
        'success': result['success'],
        'proof': result['proof'],
        'iterations': result['statistics']['iterations'],
        'time': result['statistics']['time_seconds'],
        'lean_calls': result['statistics']['lean_calls'],
        'nodes_expanded': result['statistics']['nodes_expanded'],
    }


# =============================================================================
# Test Problems
# =============================================================================

TEST_PROBLEMS = [
    # Easy: Direct computation
    ("0 + 0 = 0", "direct"),
    ("1 + 1 = 2", "direct"),
    ("2 * 2 = 4", "direct"),
    ("True", "trivial"),
    
    # Medium: Need correct tactic
    ("∀ n : Nat, n + 0 = n", "induction"),
    ("∀ n : Nat, 0 + n = n", "induction"),
    ("∀ n m : Nat, n + m = m + n", "apply_lemma"),
    
    # Harder: Need multiple tactics
    ("∀ n : Nat, n = n", "reflexivity"),
    ("∀ a b : Nat, a + b = b + a", "commutativity"),
]


# =============================================================================
# Main Comparison
# =============================================================================

def main():
    print("=" * 60)
    print("VERITAS vs Baseline Comparison")
    print("=" * 60)
    
    # Check for model
    print("\nLoading model...")
    
    # Try smaller models first for speed
    model_candidates = [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "microsoft/phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    
    model = None
    tokenizer = None
    model_name = None
    
    for candidate in model_candidates:
        try:
            print(f"  Trying {candidate}...")
            tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model_name = candidate
            print(f"  ✓ Loaded {candidate}")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if model is None:
        print("\nNo model available. Running with heuristic-only agents.")
        # Fall back to testing without LLM
        run_heuristic_only_test()
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize verifier
    print("\nInitializing LEAN 4 verifier...")
    verifier = LEAN4Verifier()
    
    # Test on problems
    print(f"\nTesting on {len(TEST_PROBLEMS)} problems...\n")
    
    baseline_results = []
    veritas_results = []
    
    for theorem, category in TEST_PROBLEMS:
        print(f"Problem: {theorem[:50]}... [{category}]")
        
        # Baseline
        verifier.call_count = 0
        baseline_result = baseline_greedy(theorem, model, tokenizer, verifier, 
                                          num_attempts=10, device=device)
        baseline_results.append(baseline_result)
        
        # VERITAS
        verifier.call_count = 0
        veritas_result = veritas_search(theorem, model, tokenizer, verifier,
                                        max_iterations=15, device=device)
        veritas_results.append(veritas_result)
        
        # Print comparison
        b_status = "✓" if baseline_result['success'] else "✗"
        v_status = "✓" if veritas_result['success'] else "✗"
        print(f"  Baseline: {b_status} ({baseline_result['attempts']} attempts, {baseline_result['time']:.2f}s)")
        print(f"  VERITAS:  {v_status} ({veritas_result['iterations']} iters, {veritas_result['time']:.2f}s)")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    baseline_solved = sum(1 for r in baseline_results if r['success'])
    veritas_solved = sum(1 for r in veritas_results if r['success'])
    
    print(f"\nProblems solved:")
    print(f"  Baseline: {baseline_solved}/{len(TEST_PROBLEMS)} ({100*baseline_solved/len(TEST_PROBLEMS):.1f}%)")
    print(f"  VERITAS:  {veritas_solved}/{len(TEST_PROBLEMS)} ({100*veritas_solved/len(TEST_PROBLEMS):.1f}%)")
    
    baseline_time = sum(r['time'] for r in baseline_results)
    veritas_time = sum(r['time'] for r in veritas_results)
    
    print(f"\nTotal time:")
    print(f"  Baseline: {baseline_time:.2f}s")
    print(f"  VERITAS:  {veritas_time:.2f}s")
    
    baseline_calls = sum(r['lean_calls'] for r in baseline_results)
    veritas_calls = sum(r['lean_calls'] for r in veritas_results)
    
    print(f"\nLEAN calls:")
    print(f"  Baseline: {baseline_calls}")
    print(f"  VERITAS:  {veritas_calls}")
    
    if veritas_solved > baseline_solved:
        print(f"\n🎉 VERITAS solved {veritas_solved - baseline_solved} more problems!")
    elif veritas_solved == baseline_solved:
        print(f"\n📊 Both methods solved the same number of problems.")
        # Check efficiency
        if veritas_calls < baseline_calls:
            print(f"   But VERITAS was more efficient ({baseline_calls - veritas_calls} fewer LEAN calls)")
    else:
        print(f"\n📉 Baseline performed better on this test set.")
    
    # Save results
    results = {
        'model': model_name,
        'baseline': baseline_results,
        'veritas': veritas_results,
        'summary': {
            'baseline_solved': baseline_solved,
            'veritas_solved': veritas_solved,
            'total_problems': len(TEST_PROBLEMS),
        }
    }
    
    results_path = Path(__file__).parent / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def run_heuristic_only_test():
    """Run comparison with heuristic agents only (no LLM)."""
    print("\n" + "=" * 60)
    print("Heuristic-Only Comparison (No LLM)")
    print("=" * 60)
    
    verifier = LEAN4Verifier()
    
    # Simple test theorems
    simple_problems = [
        "0 + 0 = 0",
        "1 + 1 = 2", 
        "True",
    ]
    
    print(f"\nTesting {len(simple_problems)} simple problems...\n")
    
    for theorem in simple_problems:
        print(f"Problem: {theorem}")
        
        # Baseline: just try common tactics
        common_tactics = ['rfl', 'decide', 'trivial', 'simp', 'omega', 'native_decide']
        baseline_success = False
        for tactic in common_tactics:
            result = verifier.validate(theorem, [tactic])
            if result.success:
                baseline_success = True
                print(f"  Baseline: ✓ (tactic: {tactic})")
                break
        if not baseline_success:
            print(f"  Baseline: ✗")
        
        # VERITAS with heuristic agents
        verifier.call_count = 0
        search = create_veritas(verifier, config=SearchConfig(max_iterations=10))
        result = search.search(theorem)
        
        v_status = "✓" if result['success'] else "✗"
        print(f"  VERITAS:  {v_status} (iters: {result['statistics']['iterations']}, calls: {result['statistics']['lean_calls']})")
        if result['success']:
            print(f"            Proof: {result['proof']}")
        print()


if __name__ == "__main__":
    main()

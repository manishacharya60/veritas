"""
miniF2F Benchmark Runner
========================
Evaluates VERITAS and baselines on the miniF2F test set (245 theorems).

Usage:
    # Fast portfolio baseline only (no LLM needed):
    python experiments/run_minif2f.py --mode portfolio

    # VERITAS with heuristic agents (no LLM):
    python experiments/run_minif2f.py --mode veritas

    # VERITAS with LLM-backed Tactician:
    python experiments/run_minif2f.py --mode veritas --model Qwen/Qwen2.5-Coder-7B-Instruct

    # Quick smoke test (5 problems):
    python experiments/run_minif2f.py --mode portfolio --max-problems 5

    # Full ablation (all variants):
    python experiments/run_minif2f.py --mode ablation
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lean.lake_validator import LakeValidator, ValidationResult
from src.lean.minif2f_parser import load_minif2f, MiniF2FTheorem, get_category_breakdown
from src.veritas import (
    ProofState, SearchConfig, VERITASSearch,
    StrategistAgent, TacticianAgent, CriticAgent, RetrieverAgent, create_veritas
)
from experiments.benchmark_utils import (
    TheoremResult, BenchmarkResult,
    PortfolioBaseline, BestOfNClaude, VERITASVerifierAdapter,
    run_benchmark, run_veritas, run_veritas_two_phase, run_veritas_ablation,
    PORTFOLIO_FAST, PORTFOLIO_EXTENDED,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
LEAN_PROJECT = PROJECT_ROOT / "lean_project"
DATA_DIR = PROJECT_ROOT / "data" / "minif2f"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="VERITAS miniF2F Benchmark")
    parser.add_argument(
        "--mode",
        choices=["portfolio", "veritas", "ablation", "both", "best_of_n", "best_of_n_300", "veritas_two_phase"],
        default="portfolio",
        help="What to run. 'best_of_n' = Claude N=32 (1 API call). "
             "'best_of_n_300' = Claude N=300 (6 API calls). "
             "'veritas_two_phase' = Phase1 Best-of-N sweep + Phase2 VERITAS MCTS (N set by --sweep-n).",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Limit number of problems (useful for quick tests)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Filter to specific categories (algebra, aime, amc, imo, ...)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for evaluation",
    )
    parser.add_argument(
        "--veritas-iterations", type=int, default=50,
        help="MCTS iterations per theorem for VERITAS",
    )
    parser.add_argument(
        "--theorem-timeout", type=int, default=300,
        help="Max seconds per theorem for VERITAS search (default: 300s = 5 min)",
    )
    parser.add_argument(
        "--sweep-n", type=int, default=5,
        help="N for Best-of-N sweep (default: 5). Used by --mode best_of_n and veritas_two_phase.",
    )
    parser.add_argument(
        "--split", choices=["test", "valid"], default="test",
        help="miniF2F split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model for LLM-backed Tactician. Use 'claude-api' for Claude (Sonnet+Haiku), "
             "or a HuggingFace model name for local models.",
    )
    parser.add_argument(
        "--lean-project", type=str, default=str(LEAN_PROJECT),
        help="Path to the Lean Lake project directory",
    )
    parser.add_argument(
        "--portfolio", choices=["fast", "extended"], default="extended",
        help="Which tactic portfolio to use for baseline",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: results/minif2f_<timestamp>.json)",
    )
    parser.add_argument(
        "--all-problems", action="store_true", default=False,
        help="Include pre-solved theorems (unsolved_only=False). Use this for standard 244-problem evaluation.",
    )
    parser.add_argument(
        "--only-theorem-names", nargs="+", default=None,
        help="Run only on these specific theorem names (useful for extending existing results).",
    )
    parser.add_argument(
        "--names-file", type=str, default=None,
        help="Path to file with one theorem name per line. Equivalent to --only-theorem-names.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load theorems ----
    print(f"Loading miniF2F theorems from {DATA_DIR}...")
    theorems = load_minif2f(
        data_dir=DATA_DIR,
        split=args.split,
        categories=args.categories,
        max_problems=args.max_problems,
        unsolved_only=not args.all_problems,
    )
    name_filter = args.only_theorem_names or []
    if args.names_file:
        with open(args.names_file) as f:
            name_filter = name_filter + [l.strip() for l in f if l.strip()]
    if name_filter:
        name_set = set(name_filter)
        theorems = [t for t in theorems if t.name in name_set]
        print(f"Filtered to {len(theorems)} specified theorems")
    print(f"Loaded {len(theorems)} theorems (all_problems={args.all_problems})")
    breakdown = get_category_breakdown(theorems)
    print(f"Category breakdown: {breakdown}")

    # ---- Init validator ----
    print(f"\nInitializing LEAN validator (project: {args.lean_project})...")
    try:
        validator = LakeValidator(project_dir=args.lean_project)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nRun this first:")
        print("  bash scripts/setup_lean_project.sh")
        sys.exit(1)

    # ---- Warm up (first Lean call loads Mathlib oleans) ----
    print("Warming up Lean (first call loads Mathlib, may take ~60s)...")
    warm_start = time.time()
    warm = validator.validate(
        theorem_statement="theorem _warmup : 1 + 1 = 2",
        tactics=["norm_num"],
    )
    warm_time = time.time() - warm_start
    if warm.success:
        print(f"  Warmup OK ({warm_time:.1f}s). Subsequent calls will be fast.")
    else:
        print(f"  Warmup failed ({warm_time:.1f}s): {warm.errors[:1]}")
        print("  This may indicate a Mathlib build issue.")
        print("  Try: cd lean_project && lake build VeritasVerifier")

    # ---- Run benchmarks ----
    all_results = []
    portfolio_choice = PORTFOLIO_FAST if args.portfolio == "fast" else PORTFOLIO_EXTENDED

    if args.mode == "best_of_n":
        bon = BestOfNClaude(validator, n=args.sweep_n)
        result = run_benchmark(
            theorems=theorems,
            method_fn=bon.prove,
            method_name=f"best_of_{args.sweep_n}_claude",
            workers=1,
            config={"n": args.sweep_n, "model": bon.model},
        )
        result.print_summary()
        all_results.append(result)

    if args.mode == "best_of_n_300":
        # Equal Lean-call budget to VERITAS MCTS (300 calls per theorem)
        # Uses 6 Claude API calls of 50 tactics each, evaluated in one batch
        bon300 = BestOfNClaude(validator, n=300)
        result = run_benchmark(
            theorems=theorems,
            method_fn=bon300.prove,
            method_name="best_of_300_claude",
            workers=1,
            config={"n": 300, "model": bon300.model,
                    "note": "Equal Lean-call budget to VERITAS (300 calls/theorem)"},
        )
        result.print_summary()
        all_results.append(result)

    if args.mode == "veritas_two_phase":
        print(f"\nUsing Two-Phase VERITAS: Best-of-{args.sweep_n} sweep → MCTS with Claude agents.")
        def two_phase_fn(theorem):
            return run_veritas_two_phase(
                theorem, validator,
                max_iterations=args.veritas_iterations,
                sweep_n=args.sweep_n,
                theorem_timeout=args.theorem_timeout,
            )
        result = run_benchmark(
            theorems=theorems,
            method_fn=two_phase_fn,
            method_name="veritas_two_phase",
            workers=1,
            config={
                "sweep_n": args.sweep_n,
                "max_iterations": args.veritas_iterations,
                "theorem_timeout": args.theorem_timeout,
                "note": "Phase1=Best-of-32 sweep, Phase2=VERITAS MCTS with failed-tactic context",
            },
        )
        result.print_summary()
        all_results.append(result)

    if args.mode in ("portfolio", "both", "ablation"):
        portfolio = PortfolioBaseline(validator, portfolio=portfolio_choice)
        result = run_benchmark(
            theorems=theorems,
            method_fn=portfolio.prove,
            method_name=f"portfolio_{args.portfolio}",
            workers=args.workers,
            config={"portfolio": args.portfolio, "tactics": len(portfolio_choice)},
        )
        result.print_summary()
        all_results.append(result)

    if args.mode in ("veritas", "both", "ablation"):
        # Determine Tactician backend
        use_claude_api = args.model == "claude-api"
        model, tokenizer = None, None
        device = "cpu"

        if use_claude_api:
            print("\nUsing Claude API-backed Tactician (Sonnet) + Critic (Haiku).")
        elif args.model:
            print(f"\nLoading model: {args.model}...")
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"  Loaded {args.model} on {device}")
            except Exception as e:
                print(f"  Failed to load model: {e}. Running heuristic-only.")

        def veritas_fn(theorem):
            return run_veritas(
                theorem, validator,
                max_iterations=args.veritas_iterations,
                model=model, tokenizer=tokenizer, device=device,
                use_claude_api=use_claude_api,
                theorem_timeout=args.theorem_timeout,
            )

        method_label = "veritas_claude" if use_claude_api else "veritas"
        result = run_benchmark(
            theorems=theorems,
            method_fn=veritas_fn,
            method_name=method_label,
            workers=1 if (model or use_claude_api) else args.workers,
            config={
                "max_iterations": args.veritas_iterations,
                "model": args.model or "heuristic",
            },
        )
        result.print_summary()
        all_results.append(result)

    if args.mode == "ablation":
        # Run ablation variants
        ablation_variants = [
            ("veritas_no_strategist", {"disable_strategist": True}),
            ("veritas_no_critic", {"disable_critic": True}),
            ("veritas_no_retriever", {"disable_retriever": True}),
            ("veritas_no_intrinsic", {"intrinsic_bonus": 0.0}),
        ]
        for variant_name, variant_config in ablation_variants:
            def ablation_fn(theorem, vc=variant_config):
                return run_veritas_ablation(theorem, validator, variant_config=vc,
                                            max_iterations=args.veritas_iterations,
                                            theorem_timeout=args.theorem_timeout)
            result = run_benchmark(
                theorems=theorems,
                method_fn=ablation_fn,
                method_name=variant_name,
                workers=args.workers,
                config={"ablation": variant_name, **variant_config},
            )
            result.print_summary()
            all_results.append(result)

    # ---- Save results ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(RESULTS_DIR / f"minif2f_{timestamp}.json")

    output = {
        "timestamp": timestamp,
        "num_theorems": len(theorems),
        "categories": breakdown,
        "results": [
            {
                "method": r.method,
                "solved": r.solved,
                "total": r.total,
                "solve_rate": r.solve_rate,
                "total_lean_calls": r.total_lean_calls,
                "total_time_seconds": r.total_time_seconds,
                "per_category": r.per_category,
                "config": r.config,
                "per_theorem": r.per_theorem,
            }
            for r in all_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # ---- Print comparison table ----
    if len(all_results) > 1:
        print(f"\n{'Method':<30} {'Solved':>8} {'Rate':>8} {'LEAN calls':>12}")
        print("-" * 62)
        for r in all_results:
            print(f"{r.method:<30} {r.solved:>4}/{r.total:<3} {r.solve_rate:>7.1%} "
                  f"{r.total_lean_calls:>12,}")


if __name__ == "__main__":
    main()

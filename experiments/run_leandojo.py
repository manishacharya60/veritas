"""
LeanDojo Benchmark Runner
=========================
Evaluates VERITAS and baselines on a stratified LeanDojo subset (Mathlib4).

Usage:
    # Download data first (one-time):
    python scripts/download_leandojo.py

    # Smoke test (5 problems):
    python experiments/run_leandojo.py --mode portfolio --max-problems 5

    # Portfolio baseline (no LLM):
    python experiments/run_leandojo.py --mode portfolio --workers 4

    # Best-of-1 Claude (pass@1):
    python experiments/run_leandojo.py --mode best_of_1 --theorem-timeout 600

    # Best-of-5 Claude (pass@5):
    python experiments/run_leandojo.py --mode best_of_5 --theorem-timeout 600

    # Main result — VERITAS Two-Phase:
    python experiments/run_leandojo.py --mode veritas_two_phase --sweep-n 5 --theorem-timeout 600 --workers 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lean.lake_validator import LakeValidator
from src.lean.leandojo_loader import load_leandojo, get_category_breakdown
from experiments.benchmark_utils import (
    TheoremResult, BenchmarkResult,
    PortfolioBaseline, BestOfNClaude,
    run_benchmark, run_veritas, run_veritas_two_phase,
    PORTFOLIO_FAST, PORTFOLIO_EXTENDED,
)

PROJECT_ROOT = Path(__file__).parent.parent
LEAN_PROJECT = PROJECT_ROOT / "lean_project"
DATA_DIR = PROJECT_ROOT / "data" / "leandojo"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="VERITAS LeanDojo Benchmark")
    parser.add_argument(
        "--mode",
        choices=["portfolio", "best_of_1", "best_of_5", "veritas_two_phase"],
        default="portfolio",
    )
    parser.add_argument("--subset-file", type=str, default="subset_176.json",
                        help="Subset JSON file in data/leandojo/ (default: subset_110.json)")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Filter to specific categories")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--veritas-iterations", type=int, default=50)
    parser.add_argument("--theorem-timeout", type=int, default=600,
                        help="Max seconds per theorem (default: 600 — Mathlib theorems are harder)")
    parser.add_argument("--sweep-n", type=int, default=5)
    parser.add_argument("--portfolio", choices=["fast", "extended"], default="extended")
    parser.add_argument("--lean-project", type=str, default=str(LEAN_PROJECT))
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    lean_project = Path(args.lean_project)

    # ---- Load theorems ----
    print(f"Loading LeanDojo theorems from {DATA_DIR} ...")
    theorems = load_leandojo(
        data_dir=DATA_DIR,
        lean_project=lean_project,
        subset_file=args.subset_file,
        categories=args.categories,
    )

    if args.max_problems is not None:
        theorems = theorems[:args.max_problems]

    print(f"Loaded {len(theorems)} theorems.")
    breakdown = get_category_breakdown(theorems)
    print(f"Category breakdown: {breakdown}")

    # ---- Init validator ----
    print(f"\nInitializing LEAN validator (project: {lean_project})...")
    try:
        validator = LakeValidator(project_dir=lean_project)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run:  bash scripts/setup_lean_project.sh")
        sys.exit(1)

    # ---- Warm up ----
    print("Warming up Lean (first call loads Mathlib, may take ~60s)...")
    warm_start = time.time()
    warm = validator.validate(
        theorem_statement="theorem _warmup : 1 + 1 = 2",
        tactics=["norm_num"],
    )
    warm_time = time.time() - warm_start
    status = "OK" if warm.success else "FAILED"
    print(f"  Warmup {status} ({warm_time:.1f}s).")

    # ---- Run ----
    all_results = []
    portfolio_choice = PORTFOLIO_FAST if args.portfolio == "fast" else PORTFOLIO_EXTENDED

    if args.mode == "portfolio":
        baseline = PortfolioBaseline(validator, portfolio=portfolio_choice)
        result = run_benchmark(
            theorems=theorems,
            method_fn=baseline.prove,
            method_name="portfolio",
            workers=args.workers,
            config={"portfolio": args.portfolio, "tactics": len(portfolio_choice)},
        )
        result.print_summary()
        all_results.append(result)

    elif args.mode == "best_of_1":
        bon1 = BestOfNClaude(validator, n=1)
        result = run_benchmark(
            theorems=theorems,
            method_fn=bon1.prove,
            method_name="best_of_1_claude",
            workers=1,
            config={"n": 1, "model": bon1.model},
        )
        result.print_summary()
        all_results.append(result)

    elif args.mode == "best_of_5":
        bon5 = BestOfNClaude(validator, n=5)
        result = run_benchmark(
            theorems=theorems,
            method_fn=bon5.prove,
            method_name="best_of_5_claude",
            workers=1,
            config={"n": 5, "model": bon5.model},
        )
        result.print_summary()
        all_results.append(result)

    elif args.mode == "veritas_two_phase":
        print(f"\nTwo-Phase VERITAS: Best-of-{args.sweep_n} sweep → MCTS.")

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
            },
        )
        result.print_summary()
        all_results.append(result)

    # ---- Save ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(RESULTS_DIR / f"leandojo_{timestamp}.json")

    output = {
        "timestamp": timestamp,
        "benchmark": "leandojo",
        "subset_file": args.subset_file,
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


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
VERITAS Evaluation Script

Run evaluation on benchmark datasets.
"""

import argparse
import json
from pathlib import Path
import logging

from src.veritas import VERITAS
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate VERITAS on benchmarks")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=["math500", "minif2f", "proofnet"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples (None = all)",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="deepseek-prover-1.3b",
        help="Generator model name",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="MCTS simulations per problem",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize VERITAS
    logger.info("Initializing VERITAS...")
    
    if args.config:
        veritas = VERITAS(config_path=args.config)
    else:
        veritas = VERITAS(
            generator_model=args.generator_model,
            num_simulations=args.num_simulations,
        )
    
    # Run evaluation
    logger.info(f"Evaluating on {args.dataset} ({args.split})")
    
    metrics = veritas.evaluate(
        dataset=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
    )
    
    # Save results
    result_file = output_dir / f"{args.dataset}_{args.split}_results.json"
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()

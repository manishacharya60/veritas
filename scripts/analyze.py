#!/usr/bin/env python
"""
VERITAS Analysis Script

Analyze evaluation results and generate reports.
"""

import argparse
import json
from pathlib import Path
import logging
from collections import defaultdict

import pandas as pd


def load_results(results_dir: Path) -> dict:
    """Load all result files from directory."""
    results = {}
    for file in results_dir.glob("*_results.json"):
        with open(file) as f:
            results[file.stem] = json.load(f)
    return results


def analyze_by_model(results_dir: Path):
    """Analyze results grouped by model."""
    results = load_results(results_dir)
    
    model_stats = defaultdict(list)
    
    for name, metrics in results.items():
        model = metrics.get("model", "unknown")
        model_stats[model].append({
            "dataset": name,
            **metrics,
        })
    
    return model_stats


def analyze_by_dataset(results_dir: Path):
    """Analyze results grouped by dataset."""
    results = load_results(results_dir)
    
    dataset_stats = defaultdict(list)
    
    for name, metrics in results.items():
        # Extract dataset name from filename
        parts = name.split("_")
        dataset = parts[0] if parts else "unknown"
        
        dataset_stats[dataset].append({
            "name": name,
            **metrics,
        })
    
    return dataset_stats


def generate_latex_table(results_dir: Path, output_file: Path):
    """Generate LaTeX table from results."""
    results = load_results(results_dir)
    
    # Build table data
    rows = []
    for name, metrics in sorted(results.items()):
        rows.append({
            "Experiment": name,
            "Success Rate": f"{metrics.get('success_rate', 0):.1%}",
            "Proved": metrics.get("proved", 0),
            "Total": metrics.get("total", 0),
            "Avg Steps": f"{metrics.get('avg_proof_length', 0):.1f}",
        })
    
    df = pd.DataFrame(rows)
    
    # Generate LaTeX
    latex = df.to_latex(index=False, escape=False)
    
    with open(output_file, 'w') as f:
        f.write(latex)
    
    return latex


def generate_summary_report(results_dir: Path, output_file: Path):
    """Generate markdown summary report."""
    results = load_results(results_dir)
    
    lines = [
        "# VERITAS Evaluation Summary",
        "",
        "## Overall Results",
        "",
        "| Dataset | Model | Success Rate | Proved/Total |",
        "|---------|-------|--------------|--------------|",
    ]
    
    for name, metrics in sorted(results.items()):
        success_rate = metrics.get("success_rate", 0)
        proved = metrics.get("proved", 0)
        total = metrics.get("total", 0)
        model = metrics.get("model", "default")
        
        lines.append(
            f"| {name} | {model} | {success_rate:.1%} | {proved}/{total} |"
        )
    
    lines.extend([
        "",
        "## Detailed Statistics",
        "",
    ])
    
    for name, metrics in sorted(results.items()):
        lines.extend([
            f"### {name}",
            "",
            f"- Success Rate: {metrics.get('success_rate', 0):.2%}",
            f"- Proved: {metrics.get('proved', 0)}",
            f"- Total: {metrics.get('total', 0)}",
            f"- Average Proof Length: {metrics.get('avg_proof_length', 0):.2f}",
            f"- Average Simulations: {metrics.get('avg_simulations', 0):.1f}",
            "",
        ])
    
    report = "\n".join(lines)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze VERITAS results")
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Output directory for analysis",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["markdown", "latex", "json", "all"],
        help="Output format",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.format in ["markdown", "all"]:
        report = generate_summary_report(
            results_dir,
            output_dir / "summary.md",
        )
        logger.info(f"Generated markdown report: {output_dir / 'summary.md'}")
    
    if args.format in ["latex", "all"]:
        latex = generate_latex_table(
            results_dir,
            output_dir / "results_table.tex",
        )
        logger.info(f"Generated LaTeX table: {output_dir / 'results_table.tex'}")
    
    if args.format in ["json", "all"]:
        # Aggregate analysis
        model_stats = analyze_by_model(results_dir)
        dataset_stats = analyze_by_dataset(results_dir)
        
        analysis = {
            "by_model": dict(model_stats),
            "by_dataset": dict(dataset_stats),
        }
        
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Generated JSON analysis: {output_dir / 'analysis.json'}")
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()

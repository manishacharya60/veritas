"""
Download and subset the LeanDojo benchmark (random split, test set).

Downloads leandojo_benchmark_4.tar.gz from Zenodo, extracts random/test.json,
and creates a stratified subset of 110 theorems (10 per Mathlib top-level module).

Usage:
    python scripts/download_leandojo.py
    python scripts/download_leandojo.py --per-category 10 --seed 42
    python scripts/download_leandojo.py --local path/to/test.json   # skip download
"""

import argparse
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lean.leandojo_loader import build_stratified_subset, _categorize

# Official Zenodo release: Yang et al., "LeanDojo: Theorem Proving with
# Retrieval-Augmented Language Models", NeurIPS 2023.
LEANDOJO_TARBALL_URL = (
    "https://zenodo.org/records/12740403/files/leandojo_benchmark_4.tar.gz?download=1"
)

DATA_DIR = Path(__file__).parent.parent / "data" / "leandojo"


def download_and_extract(url: str, dest_json: Path) -> None:
    print(f"Downloading LeanDojo benchmark from Zenodo (~68 MB)...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        urllib.request.urlretrieve(url, tmp_path)
        print(f"  Downloaded ({tmp_path.stat().st_size // (1024*1024)} MB). Extracting...")

        with tarfile.open(tmp_path, "r:gz") as tar:
            # Find random/test.json inside the archive
            target = None
            for member in tar.getmembers():
                if member.name.endswith("random/test.json"):
                    target = member
                    break
            if target is None:
                print("ERROR: random/test.json not found in tarball.")
                print("Archive contents:", [m.name for m in tar.getmembers()[:20]])
                sys.exit(1)
            f = tar.extractfile(target)
            dest_json.write_bytes(f.read())

        tmp_path.unlink()
        print(f"  Extracted to {dest_json}  ({dest_json.stat().st_size // 1024} KB)")

    except Exception as e:
        print(f"\nERROR: Download/extract failed: {e}")
        print(
            "\nManual download instructions:\n"
            "  1. Visit https://zenodo.org/records/12740403\n"
            "  2. Download leandojo_benchmark_4.tar.gz\n"
            "  3. Extract random/test.json from the archive\n"
            f"  4. Save it to {dest_json}\n"
            "  5. Re-run this script with --local path/to/test.json"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download and subset LeanDojo benchmark")
    parser.add_argument(
        "--local", type=str, default=None,
        help="Path to an already-downloaded test.json (skips download)",
    )
    parser.add_argument(
        "--per-category", type=int, default=10,
        help="Theorems to sample per Mathlib module category (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output subset JSON path (default: data/leandojo/subset_<N>.json)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Get test.json ---
    if args.local:
        test_json = Path(args.local)
        if not test_json.exists():
            print(f"ERROR: {test_json} not found")
            sys.exit(1)
    else:
        test_json = DATA_DIR / "test.json"
        if test_json.exists():
            print(f"test.json already exists at {test_json}, skipping download.")
        else:
            download_and_extract(LEANDOJO_TARBALL_URL, test_json)

    # --- Load and inspect ---
    with open(test_json) as f:
        entries = json.load(f)
    print(f"\nLoaded {len(entries)} theorems from test set.")

    # Category breakdown of full set
    from collections import Counter
    cats = Counter(_categorize(e["file_path"]) for e in entries)
    print("Category breakdown (full test set):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {count}")

    # --- Build stratified subset ---
    subset = build_stratified_subset(
        entries, per_category=args.per_category, seed=args.seed
    )
    total = len(subset)
    print(f"\nStratified subset: {total} theorems ({args.per_category} per category, seed={args.seed})")

    subset_cats = Counter(_categorize(e["file_path"]) for e in subset)
    print("Subset category breakdown:")
    for cat, count in sorted(subset_cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {count}")

    # --- Save subset ---
    output_path = args.output or str(DATA_DIR / f"subset_{total}.json")
    with open(output_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"\nSubset saved to: {output_path}")
    print(f"\nNext step: python experiments/run_leandojo.py --mode portfolio --max-problems 5")


if __name__ == "__main__":
    main()

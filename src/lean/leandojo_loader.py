"""
LeanDojo Benchmark Loader
=========================
Loads a stratified subset of LeanDojo theorems (Mathlib4 test split) into
structured objects compatible with the VERITAS experiment pipeline.

Expected data layout:
    data/leandojo/
        test.json          # LeanDojo random split, downloaded by scripts/download_leandojo.py
        subset_110.json    # Stratified 110-problem subset (created by same script)

Theorem statements are extracted from the Mathlib4 source bundled with the
lean_project's Lake packages (.lake/packages/mathlib/).
"""

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Top-level Mathlib modules mapped to short category labels.
# Ordered by expected frequency in the test set.
_MODULE_TO_CATEGORY: Dict[str, str] = {
    "Algebra":           "algebra",
    "Analysis":          "analysis",
    "Topology":          "topology",
    "NumberTheory":      "number_theory",
    "LinearAlgebra":     "linear_algebra",
    "GroupTheory":       "group_theory",
    "RingTheory":        "ring_theory",
    "FieldTheory":       "field_theory",
    "Combinatorics":     "combinatorics",
    "MeasureTheory":     "measure_theory",
    "CategoryTheory":    "category_theory",
    "Data":              "data",
    "Order":             "order",
    "Logic":             "logic",
    "Geometry":          "geometry",
    "Dynamics":          "dynamics",
    "AlgebraicGeometry": "algebraic_geometry",
    "AlgebraicTopology": "algebraic_topology",
}


@dataclass
class LeanDojoTheorem:
    """A single LeanDojo theorem, compatible with the VERITAS experiment pipeline."""
    name: str           # Short local name (last segment of full_name)
    full_name: str      # Fully qualified Mathlib name
    statement: str      # Full theorem declaration for the Lean validator
    category: str       # Derived from Mathlib module path
    file_path: str      # Relative path within Mathlib4 repo
    start_line: int     # 0-indexed line number in the source file


def _categorize(file_path: str) -> str:
    """Map a Mathlib4 file path to a short category label."""
    parts = Path(file_path).parts
    # parts[0] == "Mathlib", parts[1] == top-level module
    if len(parts) >= 2:
        return _MODULE_TO_CATEGORY.get(parts[1], "other")
    return "other"


def _find_mathlib_root(lean_project: Path) -> Optional[Path]:
    """Locate the Mathlib4 source bundled with the Lake project."""
    candidate = lean_project / ".lake" / "packages" / "mathlib"
    if candidate.is_dir():
        return candidate
    # Fallback: search one level deeper
    packages = lean_project / ".lake" / "packages"
    if packages.is_dir():
        for d in packages.iterdir():
            if d.is_dir() and (d / "Mathlib").is_dir():
                return d
    return None


def _extract_statement(
    mathlib_root: Path,
    file_path: str,
    start_line: int,
    full_name: str,
) -> Optional[str]:
    """
    Extract the theorem declaration from the Mathlib source file.

    Strategy:
    1. Try reading from the recorded start_line.
    2. If the theorem name is not found there (version mismatch), fall back
       to searching the whole file for `theorem/lemma <short_name>`.
    Returns None if extraction fails.
    """
    source_file = mathlib_root / file_path
    if not source_file.exists():
        return None

    try:
        lines = source_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    short_name = full_name.split(".")[-1]

    # Try exact line first (works when our Mathlib version matches the JSON commit)
    candidates = [start_line]
    # Also search nearby lines to handle minor version drift
    search_start = max(0, start_line - 5)
    for i in range(search_start, min(len(lines), start_line + 20)):
        if i != start_line:
            candidates.append(i)

    for line_idx in candidates:
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        if re.search(
            rf'(?:theorem|lemma|noncomputable\s+theorem|noncomputable\s+lemma)\s+{re.escape(short_name)}\b',
            line
        ):
            return _read_declaration(lines, line_idx)

    # Last resort: full-file search (slow but handles large version gaps)
    pattern = re.compile(
        rf'(?:theorem|lemma|noncomputable\s+theorem|noncomputable\s+lemma)\s+{re.escape(short_name)}\b'
    )
    for i, line in enumerate(lines):
        if pattern.search(line):
            return _read_declaration(lines, i)

    return None


def _read_declaration(lines: List[str], start: int) -> str:
    """
    Read a Lean 4 theorem/lemma declaration starting at `start`,
    stopping just before `:= by`, `:= {`, or `:=\n`.
    Returns the declaration as a single string.
    """
    # Collect lines until we hit the `:=` assignment
    collected = []
    for i in range(start, min(len(lines), start + 80)):
        line = lines[i]
        collected.append(line)
        joined = "\n".join(collected)
        # Check for the proof assignment (avoid matching `:=` inside types)
        assign = re.search(r':=\s*(?:by\b|\{|$)', joined)
        if assign:
            declaration = joined[: assign.start()].rstrip()
            return declaration

    # If we never found `:=`, return what we have (truncated fallback)
    return "\n".join(collected).strip()


def load_leandojo(
    data_dir: str | Path,
    lean_project: str | Path,
    subset_file: str = "subset_110.json",
    categories: Optional[List[str]] = None,
) -> List[LeanDojoTheorem]:
    """
    Load the LeanDojo subset from `data_dir/<subset_file>`.

    Args:
        data_dir:     Path to data/leandojo/.
        lean_project: Path to the lean_project directory (for Mathlib source lookup).
        subset_file:  JSON file produced by scripts/download_leandojo.py.
        categories:   If set, restrict to these category labels.

    Returns:
        List of LeanDojoTheorem with statements populated from Mathlib source.
    """
    data_dir = Path(data_dir)
    lean_project = Path(lean_project)

    subset_path = data_dir / subset_file
    if not subset_path.exists():
        raise FileNotFoundError(
            f"LeanDojo subset not found at {subset_path}.\n"
            f"Run:  python scripts/download_leandojo.py"
        )

    with open(subset_path) as f:
        entries = json.load(f)

    mathlib_root = _find_mathlib_root(lean_project)
    if mathlib_root is None:
        raise FileNotFoundError(
            f"Mathlib4 source not found under {lean_project}/.lake/packages/.\n"
            f"Run:  cd {lean_project} && lake update"
        )

    theorems = []
    skipped = 0
    for entry in entries:
        file_path = entry["file_path"]
        full_name = entry["full_name"]
        start_line = entry["start"][0]  # 0-indexed
        category = _categorize(file_path)

        if categories and category not in categories:
            continue

        statement = _extract_statement(mathlib_root, file_path, start_line, full_name)
        if statement is None:
            skipped += 1
            continue

        short_name = full_name.split(".")[-1]
        theorems.append(LeanDojoTheorem(
            name=short_name,
            full_name=full_name,
            statement=statement,
            category=category,
            file_path=file_path,
            start_line=start_line,
        ))

    if skipped:
        print(f"  [leandojo_loader] Skipped {skipped} theorems (statement not extractable).")

    return theorems


def build_stratified_subset(
    entries: List[dict],
    per_category: int = 10,
    seed: int = 42,
) -> List[dict]:
    """
    Build a stratified subset with `per_category` theorems from each Mathlib module.
    Takes all available if a category has fewer than `per_category` theorems.
    """
    rng = random.Random(seed)
    by_category: Dict[str, List[dict]] = defaultdict(list)
    for entry in entries:
        cat = _categorize(entry["file_path"])
        by_category[cat].append(entry)

    subset = []
    for cat, items in sorted(by_category.items()):
        rng.shuffle(items)
        subset.extend(items[:per_category])

    rng.shuffle(subset)
    return subset


def get_category_breakdown(theorems: List[LeanDojoTheorem]) -> dict:
    counts: dict = {}
    for t in theorems:
        counts[t.category] = counts.get(t.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

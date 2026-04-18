"""
CombiBench Loader
==================
Loads CombiBench theorems (Lean 4, MoonshotAI/CombiBench) into structured
objects compatible with the VERITAS experiment pipeline.

CombiBench has 100 combinatorics problems from:
  - IMO (2000–2024), 36 problems
  - Brualdi textbook, 42 problems
  - Hackmath (school-level), 10 problems
  - Other competitions (APMO, USAMO, EGMO, etc.), 12 problems

Format: each theorem in its own .lean file. Some have solution abbrevs
(like PutnamBench), some have custom definitions, some are pure proofs.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CombiTheorem:
    """A single CombiBench theorem."""
    name: str                       # e.g. "imo_2024_p3"
    source: str                     # "imo" | "brualdi" | "hackmath" | "other"
    preamble: str                   # Everything before the theorem declaration
                                    # (imports, opens, custom defs, abbrevs)
    statement: str                  # Theorem declaration without := sorry
    docstring: Optional[str]        # /-- ... -/ description
    source_file: str

    @property
    def category(self) -> str:
        return self.source

    @property
    def extra_imports(self) -> str:
        """Preamble to prepend before theorem when validating.

        Strips top-level `import ...` lines because the validator header
        already imports Mathlib. Lean 4 requires all imports before any
        declarations, so duplicating them after `open` statements fails.
        """
        lines = self.preamble.splitlines()
        filtered = [l for l in lines if not l.startswith("import ")]
        return "\n".join(filtered).strip()


def parse_combibench_file(filepath: str | Path) -> Optional["CombiTheorem"]:
    """Parse a single CombiBench .lean file."""
    filepath = Path(filepath)
    content = filepath.read_text(encoding="utf-8")
    name = filepath.stem

    # Find theorem declaration (handles noncomputable, private, etc.)
    thm_match = re.search(
        rf'(?:noncomputable\s+)?theorem\s+{re.escape(name)}\b',
        content
    )
    if not thm_match:
        return None

    thm_start = thm_match.start()

    # Everything before the theorem is the preamble
    preamble = content[:thm_start].rstrip()

    # Extract theorem signature (everything up to the proof assignment := ...)
    # CombiBench files end with `:= by sorry`, `:= by\n  tactic\n  sorry`, or `:= sorry`.
    # The first `:=` in thm_text is always the proof assignment (theorem types use `=` not `:=`).
    thm_text = content[thm_start:]
    assign_match = re.search(r'\s*:=', thm_text)
    if assign_match:
        statement = thm_text[:assign_match.start()].strip()
    else:
        statement = thm_text.strip()

    # Extract docstring from preamble (last /-- ... -/ block)
    doc_match = re.search(r'/--\s*(.*?)\s*-/', preamble, re.DOTALL)
    docstring = doc_match.group(1).strip() if doc_match else None

    source = _categorize(name)

    return CombiTheorem(
        name=name,
        source=source,
        preamble=preamble,
        statement=statement,
        docstring=docstring,
        source_file=str(filepath),
    )


def _categorize(name: str) -> str:
    if name.startswith("imo_") or name.startswith("imosl_"):
        return "imo"
    if name.startswith("brualdi_"):
        return "brualdi"
    if name.startswith("hackmath_"):
        return "hackmath"
    return "other"


def load_combibench(
    data_dir: str | Path,
    max_problems: Optional[int] = None,
) -> List[CombiTheorem]:
    """
    Load all CombiBench problems from the data directory.

    Args:
        data_dir: Path to directory containing *.lean files.
        max_problems: Limit number of problems returned.

    Returns:
        List of CombiTheorem objects, sorted alphabetically (deterministic).
    """
    data_dir = Path(data_dir)
    lean_files = sorted(data_dir.glob("*.lean"))

    if not lean_files:
        raise FileNotFoundError(f"No .lean files found in {data_dir}")

    theorems = []
    for f in lean_files:
        thm = parse_combibench_file(f)
        if thm is not None:
            theorems.append(thm)

    if max_problems is not None:
        theorems = theorems[:max_problems]

    return theorems


def get_category_breakdown(theorems: List[CombiTheorem]) -> dict:
    counts: dict = {}
    for t in theorems:
        counts[t.category] = counts.get(t.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

"""
miniF2F Theorem Parser
======================
Parses miniF2F .lean files (Lean 4 format) into structured theorem objects.

The miniF2F Lean 4 format looks like:

    theorem mathd_algebra_478 (b h v : ℝ) (h₀ : ...) (h₁ : ...) : v = 65 :=
      by sorry
    #align mathd_algebra_478 mathd_algebra_478

or for already-proved theorems:

    theorem mathd_algebra_33 (x y z : ℝ) (h₀ : x ≠ 0) ... : z / x = 7 / 25 := by
      field_simp
      nlinarith
    #align mathd_algebra_33 mathd_algebra_33
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MiniF2FTheorem:
    """A single miniF2F theorem."""
    name: str
    statement: str          # Full declaration, ending at `:=`  (no `by ...`)
    category: str           # aime | amc | imo | algebra | numbertheory | geometry | other
    proof: Optional[str]    # Existing proof tactics if already proved; None = unsolved
    source_line: int        # Line number in the source file (for debugging)

    @property
    def is_solved(self) -> bool:
        """True if the problem already has a non-sorry proof."""
        return self.proof is not None

    @property
    def lean_statement(self) -> str:
        """Statement string without trailing whitespace or := suffix."""
        return self.statement.rstrip().rstrip(":=").rstrip()

    def difficulty_estimate(self) -> int:
        """Rough difficulty 1-5 based on category."""
        return {
            "algebra": 2,
            "numbertheory": 2,
            "geometry": 3,
            "amc": 3,
            "aime": 4,
            "imo": 5,
        }.get(self.category, 2)


def parse_minif2f_file(filepath: str | Path) -> List[MiniF2FTheorem]:
    """
    Parse a miniF2F .lean file into a list of theorems.

    Handles:
    - Multi-line theorem declarations
    - Already-proved theorems (extracts proof)
    - Sorry-placeholder theorems (marks as unsolved)
    - `#align` annotations (stripped)
    """
    filepath = Path(filepath)
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()

    theorems = []

    # Find all theorem start positions
    theorem_starts = []
    for i, line in enumerate(lines):
        if re.match(r'^(theorem|lemma)\s+\w', line):
            theorem_starts.append(i)

    for idx, start_line in enumerate(theorem_starts):
        # Determine end of this theorem block
        # End = next theorem start OR next #align OR EOF
        if idx + 1 < len(theorem_starts):
            end_line = theorem_starts[idx + 1]
        else:
            end_line = len(lines)

        block_lines = lines[start_line:end_line]
        theorem = _parse_theorem_block(block_lines, start_line)
        if theorem is not None:
            theorems.append(theorem)

    return theorems


def _parse_theorem_block(
    block_lines: List[str], source_line: int
) -> Optional[MiniF2FTheorem]:
    """Parse a single theorem block (lines from `theorem` to next `theorem`)."""
    # Join lines into one string, but keep track of where := by appears
    block = "\n".join(block_lines)

    # Remove #align annotations
    block = re.sub(r'\n#align\s+\S+\s+\S+.*', '', block)
    block = block.strip()

    if not block:
        return None

    # Extract theorem name
    name_match = re.match(r'^(?:theorem|lemma)\s+(\w+)', block)
    if not name_match:
        return None
    name = name_match.group(1)

    # Find the `:= by` split point
    # Patterns to handle:
    #   (a) `... : Type :=\n  by tactic`
    #   (b) `... : Type := by tactic`
    #   (c) `... : Type :=\n  by\n  tactic1\n  tactic2`

    # Normalize: find the last occurrence of `:= by` or `:=\n  by` or `:=\n    by`
    # Strategy: find `:=` and then find `by` after it
    assign_match = re.search(r':=\s*\n?\s*by\b', block)

    if not assign_match:
        # Some theorems end with `:= by sorry` on one line
        assign_match = re.search(r':=\s+by\b', block)

    if not assign_match:
        return None

    statement = block[: assign_match.start()].rstrip()
    proof_text = block[assign_match.end():].strip()

    # Determine if actually proved (not just sorry)
    # A theorem is "solved" if the proof doesn't consist only of `sorry`
    is_only_sorry = _is_only_sorry(proof_text)
    proof = None if is_only_sorry else proof_text

    category = _categorize(name)

    return MiniF2FTheorem(
        name=name,
        statement=statement,
        category=category,
        proof=proof,
        source_line=source_line,
    )


def _is_only_sorry(proof_text: str) -> bool:
    """Return True if proof body is just `sorry` (possibly with whitespace)."""
    stripped = proof_text.strip()
    # Handle cases like "sorry", "  sorry  ", "sorry\n", "sorry -- comment"
    without_comments = re.sub(r'--[^\n]*', '', stripped).strip()
    return without_comments == "sorry"


def _categorize(name: str) -> str:
    """Categorize theorem by its name prefix."""
    if name.startswith("aime_"):
        return "aime"
    if name.startswith("amc"):
        return "amc"
    if name.startswith("imo_"):
        return "imo"
    if name.startswith("mathd_algebra"):
        return "algebra"
    if name.startswith("mathd_numbertheory"):
        return "numbertheory"
    if name.startswith("mathd_geometry"):
        return "geometry"
    if name.startswith("mathd_"):
        return "mathd"
    if name.startswith("algebra_"):
        return "algebra"
    if name.startswith("numbertheory_"):
        return "numbertheory"
    return "other"


def load_minif2f(
    data_dir: str | Path,
    split: str = "test",
    categories: Optional[List[str]] = None,
    max_problems: Optional[int] = None,
    unsolved_only: bool = True,
) -> List[MiniF2FTheorem]:
    """
    Load miniF2F problems with optional filtering.

    Args:
        data_dir: Path to the data/minif2f directory.
        split: "test" or "valid" (maps to test_problems.lean / valid_problems.lean).
        categories: If set, only return theorems in these categories.
        max_problems: Limit number of problems returned.
        unsolved_only: If True, exclude theorems that already have proofs
                       (saves time — they're already solved).
    """
    data_dir = Path(data_dir)
    filename = f"{split}_problems.lean"
    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"miniF2F data not found at {filepath}")

    theorems = parse_minif2f_file(filepath)

    if unsolved_only:
        theorems = [t for t in theorems if not t.is_solved]

    if categories:
        theorems = [t for t in theorems if t.category in categories]

    if max_problems is not None:
        theorems = theorems[:max_problems]

    return theorems


def get_category_breakdown(theorems: List[MiniF2FTheorem]) -> dict:
    """Return count per category."""
    counts = {}
    for t in theorems:
        counts[t.category] = counts.get(t.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

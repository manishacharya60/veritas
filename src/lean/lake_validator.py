"""
Lake-Project-Aware LEAN 4 Validator
=====================================
Validates proof attempts by running `lean --json` inside a Lake project
directory that has Mathlib already compiled. This gives:
  - Correct Mathlib imports (essential for miniF2F theorems)
  - Fast validation (~1-5s) after the initial `lake build` warms the cache
  - Structured JSON output for signal extraction

Usage:
    validator = LakeValidator(project_dir="lean_project")
    result = validator.validate(
        theorem_statement="theorem foo (n : Nat) : n + 0 = n",
        tactics=["simp"]
    )
"""

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


MATHLIB_HEADER = """\
import Mathlib
import Aesop

open BigOperators Real Nat Rat Finset
"""


@dataclass
class ValidationResult:
    """Result of a single proof validation."""
    success: bool               # Proof compiles with no errors and no sorry
    has_sorry: bool             # Proof compiles but uses sorry
    errors: List[str]           # Error messages from lean
    warnings: List[str]         # Warning messages
    elapsed_seconds: float

    # MCTS signals (compatible with existing veritas.py interface)
    signal_A_syntax: float      # 1.0 if no syntax errors
    signal_B_typecheck: float   # 1.0 if no type errors
    signal_C_progress: float    # Estimated goal progress (0-1)
    signal_D_complete: float    # 1.0 if proof is complete and valid

    @property
    def total_signal(self) -> float:
        return (0.25 * self.signal_A_syntax +
                0.25 * self.signal_B_typecheck +
                0.25 * self.signal_C_progress +
                0.25 * self.signal_D_complete)

    # Aliases for legacy code compatibility
    @property
    def signal_A_syntax_compat(self): return self.signal_A_syntax
    @property
    def signal_B_typecheck_compat(self): return self.signal_B_typecheck
    @property
    def signal_C_progress_compat(self): return self.signal_C_progress
    @property
    def signal_D_complete_compat(self): return self.signal_D_complete

    # Legacy interface compatibility
    remaining_goal: str = ""
    hypotheses: List[str] = None
    subgoals: List[str] = None
    stderr: str = ""

    def __post_init__(self):
        if self.hypotheses is None:
            self.hypotheses = []
        if self.subgoals is None:
            self.subgoals = []


class LakeValidator:
    """
    Validates LEAN 4 proof attempts using a pre-built Lake/Mathlib project.

    Thread-safe: each call writes to a unique temp file so concurrent
    validations don't interfere.
    """

    def __init__(
        self,
        project_dir: str,
        timeout: int = 30,
        first_call_timeout: int = 120,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.timeout = timeout
        self.first_call_timeout = first_call_timeout

        self._call_count = 0
        self._call_lock = threading.Lock()
        self._env = None    # set by _find_lean
        self._lake_bin = None  # set by _find_lean

        # Verify project exists
        if not (self.project_dir / "lakefile.toml").exists():
            raise FileNotFoundError(
                f"No lakefile.toml found in {self.project_dir}. "
                "Run scripts/setup_lean_project.sh first."
            )

        self.lean_bin = self._find_lean()

    def _find_lean_and_lake(self) -> tuple[str, str]:
        """Find lake and lean binaries. Returns (lake_bin, lean_bin)."""
        elan_bin = str(Path.home() / ".elan" / "bin")
        env = os.environ.copy()
        if elan_bin not in env.get("PATH", ""):
            env["PATH"] = elan_bin + os.pathsep + env.get("PATH", "")

        candidates = [
            str(Path.home() / ".elan" / "bin" / "lake"),
            "lake",
        ]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True, text=True, timeout=30,
                    env=env, cwd=str(self.project_dir),
                )
                if result.returncode == 0:
                    self._env = env
                    # lake env lean gives us the lean binary with proper LEAN_PATH
                    return candidate, "lean"
            except Exception:
                continue
        raise RuntimeError(
            "lake not found. Install Lean/elan:\n"
            "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh\n"
            "  source ~/.elan/env"
        )

    def _find_lean(self) -> str:
        """Kept for compatibility; delegates to _find_lean_and_lake."""
        lake, lean = self._find_lean_and_lake()
        self._lake_bin = lake
        return lean

    def validate(
        self,
        theorem_statement: str,
        tactics: List[str],
        extra_imports: str = "",
        lean_header: str = None,
    ) -> ValidationResult:
        """
        Validate a proof attempt.

        Args:
            theorem_statement: Full theorem declaration, e.g.:
                "theorem foo (n : Nat) : n + 0 = n"
                (Do NOT include `:= by` — we add that.)
            tactics: List of tactic strings to try as the proof body.
            extra_imports: Additional Lean code before the theorem.

        Returns:
            ValidationResult with success/failure and MCTS signals.
        """
        with self._call_lock:
            self._call_count += 1
            call_num = self._call_count
            # First call gets extra time for Mathlib import warmup
            timeout = self.first_call_timeout if call_num == 1 else self.timeout

        lean_code = self._build_lean_code(theorem_statement, tactics, extra_imports, lean_header)
        return self._run_lean(lean_code, timeout)

    def validate_raw(self, lean_code: str, timeout: Optional[int] = None) -> ValidationResult:
        """Validate arbitrary Lean code directly."""
        return self._run_lean(lean_code, timeout or self.timeout)

    def _build_lean_code(
        self,
        theorem_statement: str,
        tactics: List[str],
        extra_imports: str,
        lean_header: str = None,
    ) -> str:
        """Build a complete .lean file for validation."""
        parts = [lean_header if lean_header is not None else MATHLIB_HEADER]

        if extra_imports:
            parts.append(extra_imports)
            parts.append("")

        # Build proof body
        if not tactics:
            proof_body = "  sorry"
        else:
            # Indent each tactic
            indented = "\n".join(f"  {t.strip()}" for t in tactics if t.strip())
            proof_body = indented if indented else "  sorry"

        # Strip trailing `:= by` or `:=` from statement if user included it
        stmt = theorem_statement.rstrip()
        for suffix in [":= by", ":=", "by"]:
            if stmt.endswith(suffix):
                stmt = stmt[: -len(suffix)].rstrip()
                break

        parts.append(f"{stmt} := by")
        parts.append(proof_body)

        return "\n".join(parts)

    def _run_lean(self, code: str, timeout: int) -> ValidationResult:
        """Write temp file and run lean --json on it."""
        # Unique filename per call to allow concurrent validations
        unique_id = uuid.uuid4().hex[:8]
        temp_filename = f"_temp_{unique_id}.lean"
        temp_path = self.project_dir / temp_filename

        try:
            temp_path.write_text(code, encoding="utf-8")

            start = time.time()
            # `lake env lean --json <file>` sets LEAN_PATH so Mathlib oleans are found.
            # Must run from inside the project directory.
            result = subprocess.run(
                [self._lake_bin, "env", self.lean_bin, "--json", temp_filename],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._env,
            )
            elapsed = time.time() - start

            return self._parse_json_output(result, elapsed)

        except subprocess.TimeoutExpired:
            return ValidationResult(
                success=False,
                has_sorry=False,
                errors=[f"Lean timed out after {timeout}s"],
                warnings=[],
                elapsed_seconds=float(timeout),
                signal_A_syntax=0.0,
                signal_B_typecheck=0.0,
                signal_C_progress=0.0,
                signal_D_complete=0.0,
                stderr="timeout",
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                has_sorry=False,
                errors=[str(e)],
                warnings=[],
                elapsed_seconds=0.0,
                signal_A_syntax=0.0,
                signal_B_typecheck=0.0,
                signal_C_progress=0.0,
                signal_D_complete=0.0,
                stderr=str(e),
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _parse_json_output(
        self, result: subprocess.CompletedProcess, elapsed: float
    ) -> ValidationResult:
        """Parse lean --json output into a ValidationResult."""
        errors = []
        warnings = []
        has_sorry = False
        remaining_goal = ""

        # lean --json emits one JSON object per line
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                severity = msg.get("severity", "")
                # Lean 4 JSON uses "data" or "msg" depending on version
                text = msg.get("data", "") or msg.get("msg", "")
                kind = msg.get("kind", "")

                if severity == "error":
                    errors.append(text)
                    # Extract remaining goal from "unsolved goals" error
                    if not remaining_goal and "unsolved goals" in text:
                        remaining_goal = self._extract_remaining_goal(text)
                elif severity == "warning":
                    warnings.append(text)
                    if "sorry" in text.lower() or kind == "hasSorry":
                        has_sorry = True
            except json.JSONDecodeError:
                # Non-JSON output (shouldn't happen with --json, but be safe)
                if "error" in line.lower():
                    errors.append(line)

        # Also check stderr for crash-level errors
        stderr = result.stderr or ""
        if result.returncode != 0 and not errors:
            errors.append(stderr.strip() or "Lean exited with non-zero status")

        success = len(errors) == 0 and not has_sorry

        # Compute MCTS signals from error types
        signals = self._compute_signals(errors, warnings, has_sorry, success)

        return ValidationResult(
            success=success,
            has_sorry=has_sorry,
            errors=errors,
            warnings=warnings,
            elapsed_seconds=elapsed,
            signal_A_syntax=signals["A"],
            signal_B_typecheck=signals["B"],
            signal_C_progress=signals["C"],
            signal_D_complete=signals["D"],
            stderr=stderr,
            remaining_goal=remaining_goal,
            subgoals=[] if success else (["<subgoal>"] if remaining_goal else ["<unsolved>"]),
        )

    def _extract_remaining_goal(self, text: str) -> str:
        """Extract remaining goal text from a Lean 'unsolved goals' error message."""
        marker = "unsolved goals\n"
        idx = text.find(marker)
        if idx >= 0:
            goal_text = text[idx + len(marker):].strip()
            return goal_text[:500]
        # Fallback: grab everything from the first ⊢
        idx = text.find("⊢")
        if idx >= 0:
            return text[idx:idx + 500].strip()
        return ""

    def _compute_signals(
        self,
        errors: List[str],
        warnings: List[str],
        has_sorry: bool,
        success: bool,
    ) -> Dict[str, float]:
        """Compute A-B-C-D signals for MCTS from lean error messages."""
        if success:
            return {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}

        # Start pessimistic
        sig_a = 1.0  # syntax
        sig_b = 1.0  # typecheck
        sig_c = 0.0  # progress
        sig_d = 0.0  # complete

        error_text = " ".join(errors).lower()

        # Signal A: syntax errors
        if any(kw in error_text for kw in [
            "expected", "unknown tactic", "unknown identifier",
            "unexpected token", "invalid syntax",
        ]):
            sig_a = 0.0
            sig_b = 0.0

        # Signal B: type errors (syntax OK but types wrong)
        elif any(kw in error_text for kw in [
            "type mismatch", "application type mismatch",
            "failed to synthesize", "cannot unify",
        ]):
            sig_a = 1.0
            sig_b = 0.0

        # Signal C: tactic failed but code is valid
        elif any(kw in error_text for kw in [
            "unsolved goals", "tactic 'exact' failed",
            "linarith failed", "ring_nf failed", "omega could not",
        ]):
            sig_a = 1.0
            sig_b = 1.0
            sig_c = 0.3  # Progress: valid tactic, just not enough

        # Has sorry: proof accepted but incomplete
        if has_sorry:
            sig_a = 1.0
            sig_b = 1.0
            sig_c = 0.5  # Partial progress

        return {"A": sig_a, "B": sig_b, "C": sig_c, "D": sig_d}

    def validate_portfolio(
        self,
        theorem_statement: str,
        tactics: List[str],
        timeout: int = 60,
        extra_imports: str = "",
        lean_header: str = None,
    ) -> Optional[int]:
        """
        Try all tactics for one theorem in a SINGLE lean call.

        Returns the index of the first successful tactic, or None.
        Amortizes Mathlib startup: N tactics → 1 lean call (~10s).

        Strategy: rename the theorem `_p0`, `_p1`, ... for each attempt.
        Parse JSON errors by line number to find which attempt compiled cleanly.
        """
        import re as _re

        # Strip trailing := by / := / by from statement
        stmt = theorem_statement.rstrip()
        for suffix in [":= by", ":=", "by"]:
            if stmt.endswith(suffix):
                stmt = stmt[: -len(suffix)].rstrip()
                break

        # Extract keyword + parameters: "theorem <name> <params> : <type>"
        # We'll replace <name> with _p{i}
        m = _re.match(r'^(?:noncomputable\s+)?(?:theorem|lemma)\s+\S+\s*(.*)', stmt, _re.DOTALL)
        if not m:
            return None
        # Determine keyword
        kw_m = _re.match(r'^(noncomputable\s+)?(theorem|lemma)\b', stmt)
        kw = kw_m.group(2) if kw_m else "theorem"
        params_and_type = m.group(1).strip()  # "(params...) : type"

        # Build the batch file, tracking line numbers precisely
        header = lean_header if lean_header is not None else MATHLIB_HEADER
        header_lines: List[str] = header.splitlines()
        if extra_imports:
            header_lines.extend(extra_imports.splitlines())
            header_lines.append("")
        file_lines: List[str] = header_lines
        attempt_start_lines: List[int] = []  # 1-indexed line of each `theorem _pN` header

        for i, tactic_str in enumerate(tactics):
            tactic_parts = [t.strip() for t in tactic_str.replace(";", "\n").split("\n") if t.strip()]
            indented = "\n".join(f"  {t}" for t in tactic_parts) if tactic_parts else "  sorry"

            # Record the line number BEFORE appending (1-indexed)
            attempt_start_lines.append(len(file_lines) + 1)

            block = f"private {kw} _p{i} {params_and_type} := by\n{indented}"
            file_lines.extend(block.splitlines())
            file_lines.append("")  # blank line between attempts

        code = "\n".join(file_lines)

        with self._call_lock:
            self._call_count += 1

        unique_id = uuid.uuid4().hex[:8]
        temp_filename = f"_portfolio_{unique_id}.lean"
        temp_path = self.project_dir / temp_filename

        try:
            temp_path.write_text(code, encoding="utf-8")
            result = subprocess.run(
                [self._lake_bin, "env", self.lean_bin, "--json", temp_filename],
                cwd=str(self.project_dir),
                capture_output=True, text=True,
                timeout=timeout, env=self._env,
            )
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        finally:
            if temp_path.exists():
                temp_path.unlink()

        # Collect all error/sorry line numbers from lean's JSON output
        # sorry emits a warning (not error), so we must track those too
        bad_line_set: set = set()
        for raw_line in result.stdout.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                msg = json.loads(raw_line)
                severity = msg.get("severity", "")
                text = (msg.get("data", "") or msg.get("msg", "") or msg.get("message", "")).lower()
                kind = msg.get("kind", "")
                is_error = severity == "error"
                is_sorry = severity == "warning" and ("sorry" in text or kind == "hasSorry")
                if is_error or is_sorry:
                    pos = msg.get("pos")
                    if isinstance(pos, dict):
                        bad_line_set.add(pos.get("line", -1))
                    end_pos = msg.get("endPos")
                    if isinstance(end_pos, dict):
                        bad_line_set.add(end_pos.get("line", -1))
            except json.JSONDecodeError:
                pass

        # Return first attempt whose line range contains no errors or sorry.
        # Also apply a string-level sorry guard: Lean's sorry warning line number
        # is sometimes attributed to the declaration header rather than the sorry
        # expression itself, causing line-range mismatches. Belt-and-suspenders:
        # reject any tactic whose string contains 'sorry' regardless of Lean output.
        total_lines = len(file_lines)
        for i in range(len(tactics)):
            # String-level guard first (fast, catches all sorry variants)
            if "sorry" in tactics[i].lower():
                continue
            start = attempt_start_lines[i]
            end = attempt_start_lines[i + 1] if i + 1 < len(attempt_start_lines) else total_lines + 1
            if not any(start <= el < end for el in bad_line_set):
                return i

        return None

    def validate_batch_expansion(
        self,
        theorem_statement: str,
        prefix_tactics: List[str],
        candidate_tactics: List[str],
        extra_imports: str = "",
        lean_header: str = None,
        timeout: int = 60,
    ) -> List["ValidationResult"]:
        """
        Validate N candidate next-step tactics on top of a proof prefix in a
        single Lean call (6x faster than calling validate() per candidate).

        Each attempt in the batch is: prefix_tactics + candidate_tactics[i].
        Returns one ValidationResult per candidate, with remaining_goal populated
        from "unsolved goals" errors so MCTS can track proof state.
        """
        import re as _re

        if not candidate_tactics:
            return []

        # Strip trailing := by / := / by from statement
        stmt = theorem_statement.rstrip()
        for suffix in [":= by", ":=", "by"]:
            if stmt.endswith(suffix):
                stmt = stmt[: -len(suffix)].rstrip()
                break

        m = _re.match(r'^(?:noncomputable\s+)?(?:theorem|lemma)\s+\S+\s*(.*)', stmt, _re.DOTALL)
        if not m:
            err = ValidationResult(
                success=False, has_sorry=False,
                errors=["Failed to parse theorem statement"], warnings=[],
                elapsed_seconds=0.0,
                signal_A_syntax=0.0, signal_B_typecheck=0.0,
                signal_C_progress=0.0, signal_D_complete=0.0,
            )
            return [err] * len(candidate_tactics)

        kw_m = _re.match(r'^(noncomputable\s+)?(theorem|lemma)\b', stmt)
        kw = kw_m.group(2) if kw_m else "theorem"
        params_and_type = m.group(1).strip()

        # Build batch Lean file
        header = lean_header if lean_header is not None else MATHLIB_HEADER
        header_lines: List[str] = header.splitlines()
        if extra_imports:
            header_lines.extend(extra_imports.splitlines())
            header_lines.append("")

        file_lines: List[str] = list(header_lines)
        attempt_start_lines: List[int] = []

        prefix_indented = "\n".join(f"  {t.strip()}" for t in prefix_tactics if t.strip())

        for i, candidate in enumerate(candidate_tactics):
            cand_parts = [t.strip() for t in candidate.replace(";", "\n").split("\n") if t.strip()]
            cand_indented = "\n".join(f"  {t}" for t in cand_parts) if cand_parts else "  sorry"

            attempt_start_lines.append(len(file_lines) + 1)
            if prefix_indented:
                block = f"private {kw} _p{i} {params_and_type} := by\n{prefix_indented}\n{cand_indented}"
            else:
                block = f"private {kw} _p{i} {params_and_type} := by\n{cand_indented}"
            file_lines.extend(block.splitlines())
            file_lines.append("")

        code = "\n".join(file_lines)
        total_lines = len(file_lines)

        with self._call_lock:
            self._call_count += 1

        unique_id = uuid.uuid4().hex[:8]
        temp_filename = f"_batch_{unique_id}.lean"
        temp_path = self.project_dir / temp_filename

        start = time.time()
        try:
            temp_path.write_text(code, encoding="utf-8")
            result = subprocess.run(
                [self._lake_bin, "env", self.lean_bin, "--json", temp_filename],
                cwd=str(self.project_dir),
                capture_output=True, text=True,
                timeout=timeout, env=self._env,
            )
        except subprocess.TimeoutExpired:
            timeout_r = ValidationResult(
                success=False, has_sorry=False, errors=["timeout"], warnings=[],
                elapsed_seconds=float(timeout),
                signal_A_syntax=0.0, signal_B_typecheck=0.0,
                signal_C_progress=0.0, signal_D_complete=0.0,
            )
            return [timeout_r] * len(candidate_tactics)
        except Exception as e:
            err_r = ValidationResult(
                success=False, has_sorry=False, errors=[str(e)], warnings=[],
                elapsed_seconds=0.0,
                signal_A_syntax=0.0, signal_B_typecheck=0.0,
                signal_C_progress=0.0, signal_D_complete=0.0,
            )
            return [err_r] * len(candidate_tactics)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        elapsed = time.time() - start

        # Collect all Lean messages keyed by line number
        msgs_by_line: Dict[int, List[Dict]] = {}
        for raw_line in result.stdout.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                msg = json.loads(raw_line)
                severity = msg.get("severity", "")
                text = msg.get("data", "") or msg.get("msg", "") or ""
                kind = msg.get("kind", "")
                is_error = severity == "error"
                is_sorry = severity == "warning" and ("sorry" in text.lower() or kind == "hasSorry")
                if is_error or is_sorry:
                    pos = msg.get("pos")
                    if isinstance(pos, dict):
                        line_num = pos.get("line", -1)
                        msgs_by_line.setdefault(line_num, []).append(
                            {"is_error": is_error, "is_sorry": is_sorry, "text": text}
                        )
            except json.JSONDecodeError:
                pass

        # Build one ValidationResult per candidate
        results: List[ValidationResult] = []
        for i, candidate in enumerate(candidate_tactics):
            if "sorry" in candidate.lower():
                results.append(ValidationResult(
                    success=False, has_sorry=True, errors=[], warnings=["sorry in tactic"],
                    elapsed_seconds=elapsed,
                    signal_A_syntax=1.0, signal_B_typecheck=1.0,
                    signal_C_progress=0.5, signal_D_complete=0.0,
                ))
                continue

            start_line = attempt_start_lines[i]
            end_line = attempt_start_lines[i + 1] if i + 1 < len(attempt_start_lines) else total_lines + 1

            errs: List[str] = []
            warns: List[str] = []
            has_sorry_i = False
            remaining_goal = ""

            for line_num, line_msgs in msgs_by_line.items():
                if start_line <= line_num < end_line:
                    for m in line_msgs:
                        if m["is_sorry"]:
                            has_sorry_i = True
                            warns.append(m["text"])
                        else:
                            errs.append(m["text"])
                            if not remaining_goal and "unsolved goals" in m["text"]:
                                remaining_goal = self._extract_remaining_goal(m["text"])

            success = len(errs) == 0 and not has_sorry_i
            signals = self._compute_signals(errs, warns, has_sorry_i, success)
            results.append(ValidationResult(
                success=success,
                has_sorry=has_sorry_i,
                errors=errs,
                warnings=warns,
                elapsed_seconds=elapsed,
                signal_A_syntax=signals["A"],
                signal_B_typecheck=signals["B"],
                signal_C_progress=signals["C"],
                signal_D_complete=signals["D"],
                remaining_goal=remaining_goal,
                subgoals=[] if success else (["<subgoal>"] if remaining_goal else ["<unsolved>"]),
            ))

        return results

    @property
    def call_count(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# Adapter: legacy interface used by simple_comparison.py / VERITASSearch
# ---------------------------------------------------------------------------

class LakeValidatorAdapter:
    """
    Wraps LakeValidator to match the interface expected by VERITASSearch
    (from experiments/simple_comparison.py).
    """

    def __init__(self, project_dir: str, **kwargs):
        self._validator = LakeValidator(project_dir, **kwargs)
        self.call_count = 0

    def validate(
        self,
        theorem: str,
        proof_steps: List[str],
        context: str = None,
    ) -> Any:
        """Interface matching LEAN4Verifier in simple_comparison.py."""
        self.call_count += 1
        result = self._validator.validate(theorem, proof_steps)

        # Return a duck-typed object matching the expected interface
        return _LegacyResult(result)


class _LegacyResult:
    """Duck-typed result matching the legacy LEAN4Verifier.validate() output."""

    def __init__(self, r: ValidationResult):
        self.signal_A_syntax = r.signal_A_syntax
        self.signal_B_typecheck = r.signal_B_typecheck
        self.signal_C_progress = r.signal_C_progress
        self.signal_D_complete = r.signal_D_complete
        self.remaining_goal = r.remaining_goal
        self.hypotheses = r.hypotheses
        self.subgoals = r.subgoals
        self.stderr = r.stderr
        self.success = r.success

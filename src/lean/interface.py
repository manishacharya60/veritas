"""
LEAN Interface

High-level interface for LEAN theorem prover operations.
"""

from typing import Dict, Any, Optional, List, Tuple
import subprocess
import tempfile
import os
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LEANResult:
    """Result from LEAN execution."""
    success: bool
    output: str
    errors: List[str]
    goals: List[str]
    proof_state: Optional[Dict[str, Any]]


class LEANInterface:
    """
    Interface for interacting with LEAN 4 theorem prover.
    
    Provides methods for:
    - Running LEAN code
    - Checking proofs
    - Extracting proof states
    - Managing LEAN projects
    """
    
    def __init__(
        self,
        lean_path: str = "lean",
        lake_path: str = "lake",
        project_path: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize LEAN interface.
        
        Args:
            lean_path: Path to LEAN executable
            lake_path: Path to Lake build tool
            project_path: Path to LEAN project (with lakefile)
            timeout: Execution timeout in seconds
        """
        self.lean_path = lean_path
        self.lake_path = lake_path
        self.project_path = project_path
        self.timeout = timeout
        
        self._verify_installation()
    
    def _verify_installation(self):
        """Verify LEAN is installed and accessible."""
        try:
            result = subprocess.run(
                [self.lean_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"LEAN version: {version}")
            else:
                raise RuntimeError(f"LEAN check failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"LEAN not found at {self.lean_path}. "
                "Please install LEAN 4: https://leanprover.github.io/lean4/doc/setup.html"
            )
    
    def run(
        self,
        code: str,
        project: bool = True,
    ) -> LEANResult:
        """
        Run LEAN code and get results.
        
        Args:
            code: LEAN code to execute
            project: Whether to run within project context
            
        Returns:
            LEANResult with output and errors
        """
        if project and self.project_path:
            return self._run_in_project(code)
        else:
            return self._run_standalone(code)
    
    def _run_standalone(self, code: str) -> LEANResult:
        """Run LEAN code as standalone file."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.lean',
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [self.lean_path, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            return self._parse_result(result)
            
        except subprocess.TimeoutExpired:
            return LEANResult(
                success=False,
                output="",
                errors=["Execution timed out"],
                goals=[],
                proof_state=None,
            )
        finally:
            os.unlink(temp_path)
    
    def _run_in_project(self, code: str) -> LEANResult:
        """Run LEAN code within project context."""
        if not self.project_path:
            return self._run_standalone(code)
        
        # Create temporary file in project
        temp_path = Path(self.project_path) / "Temp.lean"
        
        try:
            with open(temp_path, 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                [self.lake_path, "build", "Temp"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            return self._parse_result(result)
            
        except subprocess.TimeoutExpired:
            return LEANResult(
                success=False,
                output="",
                errors=["Execution timed out"],
                goals=[],
                proof_state=None,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def _parse_result(
        self,
        result: subprocess.CompletedProcess,
    ) -> LEANResult:
        """Parse subprocess result into LEANResult."""
        errors = []
        goals = []
        
        # Parse stderr for errors
        if result.stderr:
            for line in result.stderr.split('\n'):
                if 'error' in line.lower():
                    errors.append(line)
        
        # Parse stdout for goals/state
        if result.stdout:
            goals = self._extract_goals(result.stdout)
        
        return LEANResult(
            success=result.returncode == 0,
            output=result.stdout,
            errors=errors,
            goals=goals,
            proof_state=self._extract_proof_state(result.stdout),
        )
    
    def _extract_goals(self, output: str) -> List[str]:
        """Extract goals from LEAN output."""
        goals = []
        for line in output.split('\n'):
            if '⊢' in line:
                # Extract the goal after ⊢
                goal = line.split('⊢')[-1].strip()
                goals.append(goal)
        return goals
    
    def _extract_proof_state(
        self,
        output: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract structured proof state from LEAN output."""
        if not output:
            return None
        
        # Basic parsing - would need enhancement for full state
        state = {
            "hypotheses": [],
            "goals": self._extract_goals(output),
        }
        
        # Extract hypotheses (lines before ⊢)
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if '⊢' in line:
                # Previous lines might be hypotheses
                for j in range(max(0, i-10), i):
                    if ':' in lines[j] and '⊢' not in lines[j]:
                        state["hypotheses"].append(lines[j].strip())
        
        return state
    
    def check_proof(
        self,
        theorem: str,
        proof: str,
        context: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a proof is valid.
        
        Args:
            theorem: Theorem statement
            proof: Proof tactics
            context: Additional context (imports, defs)
            
        Returns:
            Tuple of (is_valid, message)
        """
        code = self._build_proof_code(theorem, proof, context)
        result = self.run(code)
        
        if result.success:
            return True, "Proof verified"
        else:
            return False, "\n".join(result.errors)
    
    def _build_proof_code(
        self,
        theorem: str,
        proof: str,
        context: Optional[str],
    ) -> str:
        """Build complete LEAN code for proof checking."""
        parts = []
        
        # Standard imports
        parts.append("import Mathlib")
        parts.append("")
        
        if context:
            parts.append(context)
            parts.append("")
        
        # Add theorem with proof
        if "by" not in theorem:
            parts.append(f"{theorem} := by")
        else:
            parts.append(theorem)
        
        # Add proof steps
        for line in proof.split('\n'):
            parts.append(f"  {line.strip()}")
        
        return '\n'.join(parts)
    
    def get_tactics(self) -> List[str]:
        """Get list of common LEAN 4 tactics."""
        return [
            # Basic
            "intro", "intros", "exact", "apply", "refine",
            "rfl", "rw", "simp", "ring", "linarith",
            # Structural
            "cases", "induction", "rcases", "obtain",
            "constructor", "ext", "funext",
            # Automation
            "aesop", "decide", "trivial", "norm_num",
            # Advanced
            "calc", "have", "suffices", "show",
            "by_contra", "by_cases", "push_neg",
            # Library
            "omega", "polyrith", "nlinarith",
        ]
    
    def suggest_tactics(
        self,
        goal: str,
        hypotheses: List[str],
    ) -> List[str]:
        """
        Suggest tactics based on goal structure.
        
        Args:
            goal: Current proof goal
            hypotheses: Available hypotheses
            
        Returns:
            List of suggested tactics
        """
        suggestions = []
        
        # Universal quantifier
        if "∀" in goal or "forall" in goal.lower():
            suggestions.extend(["intro", "intros"])
        
        # Existential quantifier
        if "∃" in goal or "exists" in goal.lower():
            suggestions.extend(["use", "refine ⟨_, _⟩"])
        
        # Conjunction
        if "∧" in goal:
            suggestions.append("constructor")
        
        # Disjunction
        if "∨" in goal:
            suggestions.extend(["left", "right"])
        
        # Equality
        if "=" in goal:
            suggestions.extend(["rfl", "ring", "simp", "linarith"])
        
        # Inequality
        if any(op in goal for op in ["≤", "≥", "<", ">"]):
            suggestions.extend(["linarith", "omega", "norm_num"])
        
        # Default suggestions
        if not suggestions:
            suggestions = ["simp", "aesop", "decide"]
        
        return suggestions

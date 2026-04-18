"""
Proof Validator Agent

Validates generated proof steps using LEAN's type checker.
Provides binary success/failure signals with detailed error information.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
import os
import json


class ValidationStatus(Enum):
    """Status of proof validation."""
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    TACTIC_FAILED = "tactic_failed"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ValidationResult:
    """Result of proof validation."""
    status: ValidationStatus
    is_valid: bool
    remaining_goals: List[str]
    error_message: Optional[str]
    proof_state: Optional[Dict[str, Any]]
    
    # Numerical signals for MCTS (A to D)
    signal_A_syntax: float  # Syntax validity
    signal_B_typecheck: float  # Type checking passed
    signal_C_progress: float  # Goal progress made
    signal_D_complete: float  # Proof complete
    
    @property
    def total_signal(self) -> float:
        """Combined signal for MCTS backpropagation."""
        return (
            0.25 * self.signal_A_syntax +
            0.25 * self.signal_B_typecheck +
            0.25 * self.signal_C_progress +
            0.25 * self.signal_D_complete
        )


class ProofValidator:
    """
    LEAN-based proof validator for MCTS simulation.
    
    Responsibilities:
    - Validate generated proof steps against LEAN type checker
    - Track remaining goals after applying tactics
    - Compute numerical signals (A-D) for MCTS value updates
    """
    
    def __init__(
        self,
        lean_path: str = "lean",
        timeout: int = 30,
        memory_limit: int = 4096,
    ):
        """
        Initialize the Proof Validator.
        
        Args:
            lean_path: Path to LEAN executable
            timeout: Maximum time for proof checking (seconds)
            memory_limit: Memory limit for LEAN (MB)
        """
        self.lean_path = lean_path
        self.timeout = timeout
        self.memory_limit = memory_limit
        
        self._verify_lean_installation()
    
    def _verify_lean_installation(self):
        """Verify LEAN is properly installed."""
        try:
            result = subprocess.run(
                [self.lean_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"LEAN check failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(f"LEAN not found at {self.lean_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("LEAN version check timed out")
    
    def validate(
        self,
        proof_step: str,
        theorem: str,
        prior_steps: List[str],
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a proof step using LEAN.
        
        Args:
            proof_step: The proof step/tactic to validate
            theorem: The theorem being proved
            prior_steps: Previously validated proof steps
            context: Additional LEAN context (imports, definitions)
            
        Returns:
            ValidationResult with status and signals
        """
        # Build complete LEAN file
        lean_code = self._build_lean_file(
            theorem, prior_steps + [proof_step], context
        )
        
        # Run LEAN type checker
        status, output, error = self._run_lean(lean_code)
        
        # Parse results and compute signals
        return self._parse_result(status, output, error, prior_steps)
    
    def validate_complete_proof(
        self,
        theorem: str,
        proof: str,
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a complete proof.
        
        Args:
            theorem: The theorem statement
            proof: The complete proof
            context: Additional LEAN context
            
        Returns:
            ValidationResult indicating if proof is complete and valid
        """
        lean_code = self._build_lean_file(theorem, [proof], context)
        status, output, error = self._run_lean(lean_code)
        return self._parse_result(status, output, error, [])
    
    def get_proof_state(
        self,
        theorem: str,
        steps: List[str],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get the current proof state after applying steps.
        
        Args:
            theorem: The theorem being proved
            steps: Proof steps applied so far
            context: Additional LEAN context
            
        Returns:
            Dictionary with hypotheses, goals, and other state info
        """
        # Build proof with #check_state marker
        lean_code = self._build_lean_file_with_state(theorem, steps, context)
        _, output, _ = self._run_lean(lean_code)
        
        return self._parse_proof_state(output)
    
    def _build_lean_file(
        self,
        theorem: str,
        steps: List[str],
        context: Optional[str],
    ) -> str:
        """Build a complete LEAN file for validation."""
        parts = []
        
        # Standard imports
        parts.append("import Mathlib")
        parts.append("import Aesop")
        parts.append("")
        
        # Additional context
        if context:
            parts.append(context)
            parts.append("")
        
        # Theorem with proof
        parts.append(theorem)
        parts.append("  by")
        
        for step in steps:
            parts.append(f"    {step}")
        
        return "\n".join(parts)
    
    def _build_lean_file_with_state(
        self,
        theorem: str,
        steps: List[str],
        context: Optional[str],
    ) -> str:
        """Build LEAN file with state inspection."""
        parts = []
        
        parts.append("import Mathlib")
        parts.append("import Aesop")
        parts.append("")
        
        if context:
            parts.append(context)
            parts.append("")
        
        parts.append(theorem)
        parts.append("  by")
        
        for step in steps:
            parts.append(f"    {step}")
        
        # Add state inspection
        parts.append("    trace_state")
        
        return "\n".join(parts)
    
    def _run_lean(
        self,
        code: str,
    ) -> Tuple[ValidationStatus, str, str]:
        """
        Run LEAN on the given code.
        
        Args:
            code: LEAN code to validate
            
        Returns:
            Tuple of (status, stdout, stderr)
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.lean',
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [self.lean_path, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return ValidationStatus.SUCCESS, result.stdout, result.stderr
            else:
                # Parse error type
                error = result.stderr.lower()
                if "syntax error" in error:
                    return ValidationStatus.SYNTAX_ERROR, result.stdout, result.stderr
                elif "type mismatch" in error:
                    return ValidationStatus.TYPE_ERROR, result.stdout, result.stderr
                elif "tactic" in error and "failed" in error:
                    return ValidationStatus.TACTIC_FAILED, result.stdout, result.stderr
                else:
                    return ValidationStatus.UNKNOWN_ERROR, result.stdout, result.stderr
                    
        except subprocess.TimeoutExpired:
            return ValidationStatus.TIMEOUT, "", "Validation timed out"
        finally:
            os.unlink(temp_path)
    
    def _parse_result(
        self,
        status: ValidationStatus,
        output: str,
        error: str,
        prior_steps: List[str],
    ) -> ValidationResult:
        """Parse LEAN output into ValidationResult with signals."""
        
        # Compute signals based on status
        if status == ValidationStatus.SUCCESS:
            return ValidationResult(
                status=status,
                is_valid=True,
                remaining_goals=[],
                error_message=None,
                proof_state=None,
                signal_A_syntax=1.0,
                signal_B_typecheck=1.0,
                signal_C_progress=1.0,
                signal_D_complete=1.0,
            )
        
        elif status == ValidationStatus.SYNTAX_ERROR:
            return ValidationResult(
                status=status,
                is_valid=False,
                remaining_goals=self._extract_goals(output),
                error_message=error,
                proof_state=None,
                signal_A_syntax=0.0,
                signal_B_typecheck=0.0,
                signal_C_progress=0.0,
                signal_D_complete=0.0,
            )
        
        elif status == ValidationStatus.TYPE_ERROR:
            return ValidationResult(
                status=status,
                is_valid=False,
                remaining_goals=self._extract_goals(output),
                error_message=error,
                proof_state=None,
                signal_A_syntax=1.0,
                signal_B_typecheck=0.0,
                signal_C_progress=0.0,
                signal_D_complete=0.0,
            )
        
        elif status == ValidationStatus.TACTIC_FAILED:
            # Tactic was valid but didn't work
            return ValidationResult(
                status=status,
                is_valid=False,
                remaining_goals=self._extract_goals(output),
                error_message=error,
                proof_state=None,
                signal_A_syntax=1.0,
                signal_B_typecheck=1.0,
                signal_C_progress=0.2,  # Some credit for valid attempt
                signal_D_complete=0.0,
            )
        
        else:  # TIMEOUT or UNKNOWN
            return ValidationResult(
                status=status,
                is_valid=False,
                remaining_goals=[],
                error_message=error,
                proof_state=None,
                signal_A_syntax=0.5,
                signal_B_typecheck=0.0,
                signal_C_progress=0.0,
                signal_D_complete=0.0,
            )
    
    def _extract_goals(self, output: str) -> List[str]:
        """Extract remaining goals from LEAN output."""
        goals = []
        # Parse LEAN goal output format
        # This is simplified - actual implementation needs proper parsing
        if "⊢" in output:
            for line in output.split("\n"):
                if "⊢" in line:
                    goals.append(line.split("⊢")[-1].strip())
        return goals
    
    def _parse_proof_state(self, output: str) -> Dict[str, Any]:
        """Parse proof state from trace_state output."""
        return {
            "hypotheses": [],
            "goals": self._extract_goals(output),
            "raw_output": output
        }

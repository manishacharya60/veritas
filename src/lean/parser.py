"""
LEAN Parser

Parsing utilities for LEAN code and output.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class ParsedTheorem:
    """Parsed theorem structure."""
    name: str
    statement: str
    type_signature: str
    namespace: Optional[str]
    attributes: List[str]


@dataclass
class ParsedTactic:
    """Parsed tactic information."""
    name: str
    arguments: List[str]
    modifiers: List[str]


@dataclass
class ParsedGoal:
    """Parsed proof goal."""
    goal_type: str
    hypotheses: List[Tuple[str, str]]  # (name, type)
    target: str


class LEANParser:
    """
    Parser for LEAN 4 code and output.
    
    Provides utilities for:
    - Parsing theorem statements
    - Extracting proof states
    - Parsing error messages
    - Analyzing tactics
    """
    
    # Regex patterns
    THEOREM_PATTERN = re.compile(
        r'(theorem|lemma|def|example)\s+(\w+)\s*(?:\{[^}]*\})?\s*(?:\([^)]*\))?\s*:\s*([^:=]+)(?::=|where)',
        re.MULTILINE | re.DOTALL
    )
    
    GOAL_PATTERN = re.compile(
        r'⊢\s*(.+)',
        re.MULTILINE
    )
    
    HYPOTHESIS_PATTERN = re.compile(
        r'(\w+)\s*:\s*([^,\n]+)',
    )
    
    ERROR_PATTERN = re.compile(
        r'error:\s*(.+)',
        re.IGNORECASE
    )
    
    TACTIC_PATTERN = re.compile(
        r'^(\w+)(?:\s+(.+))?$'
    )
    
    @classmethod
    def parse_theorem(cls, code: str) -> Optional[ParsedTheorem]:
        """
        Parse a theorem statement from LEAN code.
        
        Args:
            code: LEAN code containing a theorem
            
        Returns:
            ParsedTheorem or None if not found
        """
        match = cls.THEOREM_PATTERN.search(code)
        if not match:
            return None
        
        keyword, name, type_sig = match.groups()
        
        # Extract namespace if present
        namespace = None
        if '.' in name:
            parts = name.rsplit('.', 1)
            namespace = parts[0]
            name = parts[1]
        
        # Extract attributes
        attributes = cls._extract_attributes(code, match.start())
        
        return ParsedTheorem(
            name=name,
            statement=match.group(0),
            type_signature=type_sig.strip(),
            namespace=namespace,
            attributes=attributes,
        )
    
    @classmethod
    def _extract_attributes(cls, code: str, position: int) -> List[str]:
        """Extract attributes before a theorem."""
        attributes = []
        
        # Look for @[...] before the theorem
        before = code[:position]
        attr_matches = re.findall(r'@\[([^\]]+)\]', before[-200:])
        
        for match in attr_matches:
            attributes.extend(match.split(','))
        
        return [a.strip() for a in attributes]
    
    @classmethod
    def parse_goals(cls, output: str) -> List[ParsedGoal]:
        """
        Parse proof goals from LEAN output.
        
        Args:
            output: LEAN output containing goals
            
        Returns:
            List of ParsedGoal objects
        """
        goals = []
        
        # Split by goal markers
        sections = output.split('⊢')
        
        for i, section in enumerate(sections[1:], 1):  # Skip first (before any ⊢)
            # Target is the first line after ⊢
            lines = section.strip().split('\n')
            target = lines[0].strip() if lines else ""
            
            # Look for hypotheses in previous section
            hypotheses = []
            if i > 0:
                prev_section = sections[i-1] if i < len(sections) else ""
                hyp_matches = cls.HYPOTHESIS_PATTERN.findall(prev_section)
                hypotheses = [(name, typ.strip()) for name, typ in hyp_matches]
            
            goals.append(ParsedGoal(
                goal_type="main" if i == 1 else "subgoal",
                hypotheses=hypotheses,
                target=target,
            ))
        
        return goals
    
    @classmethod
    def parse_error(cls, error_text: str) -> Dict[str, Any]:
        """
        Parse a LEAN error message.
        
        Args:
            error_text: Error message text
            
        Returns:
            Dictionary with error details
        """
        error_info = {
            "type": "unknown",
            "message": error_text,
            "line": None,
            "column": None,
            "expected": None,
            "found": None,
        }
        
        # Extract line/column
        loc_match = re.search(r':(\d+):(\d+):', error_text)
        if loc_match:
            error_info["line"] = int(loc_match.group(1))
            error_info["column"] = int(loc_match.group(2))
        
        # Classify error type
        text_lower = error_text.lower()
        if "syntax" in text_lower:
            error_info["type"] = "syntax"
        elif "type mismatch" in text_lower:
            error_info["type"] = "type_mismatch"
        elif "unknown identifier" in text_lower:
            error_info["type"] = "unknown_identifier"
        elif "tactic" in text_lower and "failed" in text_lower:
            error_info["type"] = "tactic_failed"
        elif "unsolved goals" in text_lower:
            error_info["type"] = "unsolved_goals"
        
        # Extract expected/found for type errors
        if error_info["type"] == "type_mismatch":
            expected_match = re.search(r'expected\s+(.+)', error_text)
            found_match = re.search(r'found\s+(.+)', error_text)
            if expected_match:
                error_info["expected"] = expected_match.group(1).strip()
            if found_match:
                error_info["found"] = found_match.group(1).strip()
        
        return error_info
    
    @classmethod
    def parse_tactic(cls, tactic_code: str) -> ParsedTactic:
        """
        Parse a tactic invocation.
        
        Args:
            tactic_code: Tactic code string
            
        Returns:
            ParsedTactic object
        """
        tactic_code = tactic_code.strip()
        
        # Handle modifiers (like 'simp only' or 'rw [...]')
        modifiers = []
        if tactic_code.startswith("simp only"):
            modifiers.append("only")
            tactic_code = tactic_code[10:]
        
        match = cls.TACTIC_PATTERN.match(tactic_code)
        if match:
            name = match.group(1)
            args_str = match.group(2) or ""
            
            # Parse arguments
            arguments = cls._parse_tactic_arguments(args_str)
            
            return ParsedTactic(
                name=name,
                arguments=arguments,
                modifiers=modifiers,
            )
        
        return ParsedTactic(
            name=tactic_code,
            arguments=[],
            modifiers=[],
        )
    
    @classmethod
    def _parse_tactic_arguments(cls, args_str: str) -> List[str]:
        """Parse tactic arguments."""
        if not args_str:
            return []
        
        args = []
        
        # Handle bracketed lists [a, b, c]
        if args_str.startswith('['):
            end = args_str.find(']')
            if end > 0:
                inner = args_str[1:end]
                args.extend([a.strip() for a in inner.split(',')])
                args_str = args_str[end+1:].strip()
        
        # Handle remaining arguments
        if args_str:
            args.extend(args_str.split())
        
        return args
    
    @classmethod
    def extract_proof_steps(cls, proof_code: str) -> List[str]:
        """
        Extract individual proof steps from proof code.
        
        Args:
            proof_code: Complete proof code
            
        Returns:
            List of individual tactic calls
        """
        steps = []
        
        # Remove 'by' keyword if present
        proof_code = re.sub(r'^\s*by\s*', '', proof_code)
        
        # Split by common tactic separators
        # Handle both ';' combinator and newlines
        lines = proof_code.replace(';', '\n').split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--'):  # Skip comments
                steps.append(line)
        
        return steps
    
    @classmethod
    def format_proof(
        cls,
        theorem: str,
        steps: List[str],
        indent: int = 2,
    ) -> str:
        """
        Format proof steps into complete LEAN code.
        
        Args:
            theorem: Theorem statement
            steps: List of proof steps
            indent: Indentation spaces
            
        Returns:
            Formatted LEAN code
        """
        lines = [theorem]
        
        if "by" not in theorem:
            lines.append(f"{' ' * indent}by")
        
        for step in steps:
            lines.append(f"{' ' * (indent * 2)}{step}")
        
        return '\n'.join(lines)
    
    @classmethod
    def normalize_lean_code(cls, code: str) -> str:
        """
        Normalize LEAN code for comparison.
        
        Args:
            code: LEAN code
            
        Returns:
            Normalized code
        """
        # Remove comments
        code = re.sub(r'--.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\-.*?\-/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        code = code.strip()
        
        return code

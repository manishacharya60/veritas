"""
Claude API-Backed Tactician Agent
===================================
Replaces the heuristic TacticianAgent with Claude calls.

This is the key improvement over the heuristic baseline:
- Claude reads the actual theorem, goal, and hypotheses
- Generates theorem-specific tactics, not generic ones
- Understands Lean 4 syntax and Mathlib lemma names
- Conditions on strategy, premises, and Critic guidance

For the paper: this demonstrates that VERITAS's value comes from the
multi-agent coordination + MCTS structure, not just the LLM alone
(the ablation with Best-of-N will show that).
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import anthropic

from src.veritas import (
    BaseAgent, ProofState, StrategyPlan, TacticCandidate,
    CriticAssessment, RetrievedPremise, ProofStrategy, TacticianAgent
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Lean 4 theorem prover with deep knowledge of Mathlib.
Your task is to suggest the next tactic(s) to apply in a proof.

Rules:
- Output ONLY valid Lean 4 (not Lean 3, not Coq, not Isabelle)
- Each tactic must be syntactically complete and correct
- Prefer simpler tactics when possible
- Common Mathlib tactics: rfl, norm_num, ring, omega, simp, linarith, nlinarith,
  aesop, decide, native_decide, positivity, norm_cast, field_simp, push_neg,
  constructor, exact, apply, intro, induction, cases, rcases, obtain, use, refine
- For universal statements: start with `intro`
- For equalities: try ring, norm_num, simp, linarith
- For nat/int arithmetic: omega is very powerful
- For induction: `induction n with | zero => ... | succ n ih => ...`
- Multi-step: separate tactics with newlines (each on its own line)
- Do NOT output sorry
"""

TACTIC_REQUEST = """\
Prove this Lean 4 theorem step by step.

## Theorem
```lean
{theorem}
```

## Current Goal
```
{goal}
```

## Available Hypotheses
{hypotheses}

## Proof Steps Applied So Far
{history}

## Suggested Strategy
{strategy}

## Relevant Lemmas (from Mathlib)
{premises}

## Your Task
Suggest {n} different proof attempts. Each attempt is either:
- A single tactic (e.g., `ring`)
- Multiple tactics on separate lines (e.g., `intro n\nsimp`)

Format your response as JSON:
```json
[
  {{"tactic": "ring", "confidence": 0.9, "reasoning": "goal is an algebraic identity"}},
  {{"tactic": "norm_num", "confidence": 0.8, "reasoning": "goal is numeric computation"}},
  {{"tactic": "intro n\\nsimp", "confidence": 0.6, "reasoning": "universally quantified, simp might close"}}
]
```

Only include tactics you believe have a real chance of working.
Do NOT include sorry.
"""

# ---------------------------------------------------------------------------
# Claude Tactician
# ---------------------------------------------------------------------------

class ClaudeTacticianAgent(TacticianAgent):
    """
    Tactician agent backed by Claude API.

    Falls back to heuristic parent class if:
    - API key is not set
    - API call fails
    - Response cannot be parsed

    Thread-safe: uses a lock around API calls to respect rate limits.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        fallback_to_heuristic: bool = True,
        rate_limit_delay: float = 0.1,
        cache_responses: bool = True,
        phase1_failed_tactics: List[str] = None,
    ):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback_to_heuristic = fallback_to_heuristic
        self.rate_limit_delay = rate_limit_delay
        self.cache_responses = cache_responses
        self.phase1_failed_tactics: List[str] = phase1_failed_tactics or []

        self._client = anthropic.Anthropic()
        self._lock = threading.Lock()
        self._cache: Dict[str, List[TacticCandidate]] = {}
        self._api_calls = 0
        self._cache_hits = 0
        self._last_call_time = 0.0

    def __call__(
        self,
        state: ProofState,
        strategy: StrategyPlan,
        premises: List[RetrievedPremise] = None,
        critic_guidance: CriticAssessment = None,
        num_candidates: int = 6,
        **kwargs,
    ) -> List[TacticCandidate]:
        """Generate tactic candidates using Claude."""

        cache_key = self._cache_key(state, strategy, num_candidates)
        if self.cache_responses and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        try:
            candidates = self._call_claude(
                state, strategy, premises, critic_guidance, num_candidates
            )
        except Exception as e:
            logger.warning(f"Claude API failed: {e}. Falling back to heuristics.")
            if self.fallback_to_heuristic:
                candidates = super().__call__(
                    state, strategy, premises, critic_guidance, num_candidates
                )
            else:
                candidates = []

        # Always augment with a few heuristic tactics to ensure coverage
        heuristic = super().__call__(
            state, strategy, premises, critic_guidance, num_candidates=3
        )
        combined = self._merge(candidates, heuristic, num_candidates)

        if self.cache_responses:
            self._cache[cache_key] = combined

        return combined

    def _call_claude(
        self,
        state: ProofState,
        strategy: StrategyPlan,
        premises: Optional[List[RetrievedPremise]],
        critic_guidance: Optional[CriticAssessment],
        num_candidates: int,
    ) -> List[TacticCandidate]:
        """Make the API call and parse response."""

        prompt = self._build_prompt(state, strategy, premises, critic_guidance, num_candidates)

        # Rate limiting
        with self._lock:
            now = time.time()
            wait = self.rate_limit_delay - (now - self._last_call_time)
            if wait > 0:
                time.sleep(wait)
            self._api_calls += 1
            self._last_call_time = time.time()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text
        return self._parse_response(raw, num_candidates)

    def _build_prompt(
        self,
        state: ProofState,
        strategy: StrategyPlan,
        premises: Optional[List[RetrievedPremise]],
        critic_guidance: Optional[CriticAssessment],
        n: int,
    ) -> str:
        """Build the prompt for Claude."""

        # Hypotheses
        if state.hypotheses:
            hyp_str = "\n".join(f"  {h}" for h in state.hypotheses)
        else:
            hyp_str = "  (none)"

        # History
        if state.tactics_applied:
            hist_str = "\n".join(f"  {t}" for t in state.tactics_applied)
        else:
            hist_str = "  (proof not started yet)"

        # Strategy
        strat_str = f"{strategy.strategy.value}"
        if strategy.target_lemmas:
            strat_str += f" (suggested lemmas: {', '.join(strategy.target_lemmas[:3])})"
        if strategy.subgoal_decomposition:
            strat_str += f"\n  Decomposition: {'; '.join(strategy.subgoal_decomposition[:3])}"

        # Premises
        if premises:
            prem_str = "\n".join(
                f"  - {p.lemma_name}: {p.lemma_statement[:80]} ({p.usage_hint})"
                for p in premises[:5]
            )
        else:
            prem_str = "  (none retrieved)"

        # Add Critic guidance to the end if available
        extra = ""
        if critic_guidance and critic_guidance.suggested_tactics:
            extra = f"\n## Critic Suggests\n{', '.join(critic_guidance.suggested_tactics[:3])}"
        if critic_guidance and critic_guidance.reasoning:
            extra += f"\nCritic reasoning: {critic_guidance.reasoning[:200]}"

        # Phase 1 failed tactics — tell Claude what was already tried
        if self.phase1_failed_tactics:
            failed_preview = self.phase1_failed_tactics[:20]
            extra += f"\n## Already Tried (Phase 1 — all failed, do NOT repeat these)\n"
            extra += "\n".join(f"  - {t}" for t in failed_preview)

        return TACTIC_REQUEST.format(
            theorem=state.theorem,
            goal=state.goal or state.theorem,
            hypotheses=hyp_str,
            history=hist_str,
            strategy=strat_str,
            premises=prem_str,
            n=n,
        ) + extra

    def _parse_response(self, raw: str, n: int) -> List[TacticCandidate]:
        """Parse Claude's JSON response into TacticCandidate objects."""
        candidates = []

        # Extract JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else None

        if json_str:
            try:
                items = json.loads(json_str)
                for item in items[:n]:
                    tactic = item.get("tactic", "").strip()
                    if not tactic or "sorry" in tactic.lower():
                        continue
                    confidence = float(item.get("confidence", 0.5))
                    reasoning = item.get("reasoning", "Claude suggestion")
                    candidates.append(TacticCandidate(
                        tactic=tactic,
                        reasoning=reasoning,
                        prior=min(1.0, confidence),
                        strategy_alignment=confidence,
                    ))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"JSON parse error: {e}. Trying line-by-line parse.")
                candidates = self._parse_fallback(raw, n)
        else:
            candidates = self._parse_fallback(raw, n)

        return candidates

    def _parse_fallback(self, raw: str, n: int) -> List[TacticCandidate]:
        """Fallback: extract tactics from plain text if JSON parsing fails."""
        candidates = []
        for line in raw.splitlines():
            line = line.strip()
            # Skip explanatory lines
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # Skip lines that look like prose
            if len(line.split()) > 8 and not any(
                kw in line for kw in ['intro', 'simp', 'ring', 'norm_num', 'omega', 'linarith']
            ):
                continue
            # Clean up markdown
            line = re.sub(r'^[-*•`]\s*', '', line).strip('`').strip()
            if line and 'sorry' not in line.lower():
                candidates.append(TacticCandidate(
                    tactic=line,
                    reasoning="Claude suggestion (fallback parse)",
                    prior=0.5,
                    strategy_alignment=0.5,
                ))
            if len(candidates) >= n:
                break
        return candidates

    def _merge(
        self,
        claude: List[TacticCandidate],
        heuristic: List[TacticCandidate],
        n: int,
    ) -> List[TacticCandidate]:
        """Merge Claude and heuristic candidates, deduplicating."""
        seen = set()
        merged = []
        for c in claude + heuristic:
            tactic_key = c.tactic.strip()
            if tactic_key not in seen:
                seen.add(tactic_key)
                merged.append(c)
        return sorted(merged, key=lambda x: -x.prior)[:n]

    def _cache_key(self, state: ProofState, strategy: StrategyPlan, n: int) -> str:
        return f"{state.hash()}:{strategy.strategy.value}:{n}"

    @property
    def stats(self) -> Dict[str, int]:
        return {"api_calls": self._api_calls, "cache_hits": self._cache_hits}


# ---------------------------------------------------------------------------
# Claude Critic Agent
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """\
You are an expert Lean 4 proof evaluator. Given a proof state, estimate:
1. The probability (0-1) that a proof exists from this state
2. Which tactics are most likely to make progress
3. Whether this branch should be abandoned

Be concise and numerical. Output JSON only.
"""

CRITIC_PROMPT = """\
## Theorem
{theorem}

## Current Goal
{goal}

## Proof History
{history}

## Depth
{depth} steps taken

Evaluate this proof state. Output JSON:
```json
{{
  "value": 0.7,
  "reasoning": "One sentence explanation",
  "suggested_tactics": ["ring", "norm_num", "linarith"],
  "should_prune": false
}}
```

value: probability [0,1] of finding a proof from here
should_prune: true if this branch looks hopeless
"""


class ClaudeCriticAgent(BaseAgent):
    """
    Critic agent backed by Claude API.

    Provides richer value estimates than the heuristic CriticAgent:
    - Actually reads the goal and understands what's needed
    - Suggests tactics informed by the theorem structure
    - Pruning recommendations based on proof state analysis

    For the ablation study: replacing this with the heuristic Critic
    should show a measurable drop in solve rate.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",  # Use Haiku for Critic (fast+cheap)
        max_tokens: int = 256,
        temperature: float = 0.3,  # Low temp for consistent value estimates
        cache_responses: bool = True,
        rate_limit_delay: float = 0.3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_responses = cache_responses
        self.rate_limit_delay = rate_limit_delay

        self._client = anthropic.Anthropic()
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._api_calls = 0
        self._last_call_time = 0.0

        # Keep heuristic critic as fallback
        from src.veritas import CriticAgent
        self._heuristic = CriticAgent()

    def __call__(self, state: ProofState, tactics=None, **kwargs):
        """Assess proof state value and suggest tactics."""
        from src.veritas import CriticAssessment

        cache_key = state.hash()
        if self.cache_responses and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            assessment = self._call_claude(state)
        except Exception as e:
            logger.debug(f"Claude Critic API failed: {e}. Using heuristic.")
            assessment = self._heuristic(state, tactics)

        if self.cache_responses:
            self._cache[cache_key] = assessment

        return assessment

    def _call_claude(self, state: ProofState):
        from src.veritas import CriticAssessment

        history_str = "\n".join(state.tactics_applied) if state.tactics_applied else "(none)"
        prompt = CRITIC_PROMPT.format(
            theorem=state.theorem,
            goal=state.goal or state.theorem,
            history=history_str,
            depth=state.depth,
        )

        with self._lock:
            now = time.time()
            wait = self.rate_limit_delay - (now - self._last_call_time)
            if wait > 0:
                time.sleep(wait)
            self._api_calls += 1
            self._last_call_time = time.time()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=CRITIC_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text
        return self._parse_assessment(raw, state)

    def _parse_assessment(self, raw: str, state: ProofState):
        from src.veritas import CriticAssessment

        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else None

        if json_str:
            try:
                data = json.loads(json_str)
                return CriticAssessment(
                    value=float(data.get("value", 0.5)),
                    policy_prior={},
                    reasoning=data.get("reasoning", ""),
                    suggested_tactics=data.get("suggested_tactics", [])[:5],
                    pruning_recommendation=bool(data.get("should_prune", False)),
                )
            except Exception:
                pass

        # Fallback: extract a number from the response
        nums = re.findall(r'\b0\.\d+\b', raw)
        value = float(nums[0]) if nums else 0.5
        return CriticAssessment(
            value=value,
            policy_prior={},
            reasoning=raw[:100],
            suggested_tactics=[],
            pruning_recommendation=value < 0.1,
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {"api_calls": self._api_calls}

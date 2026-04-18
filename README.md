# VERITAS: Verification-Enhanced Reasoning with Integrated Tactic Agents and Search

A multi-agent MCTS framework for automated theorem proving in Lean 4, evaluated on competition mathematics benchmarks.

## Results

### miniF2F (244 problems, Lean 4)

| Method                | Solved       | Rate   | Lean Calls |
|-----------------------|--------------|--------|------------|
| Portfolio (heuristic) | 66 / 244     | 27.0%  | 244        |
| Best-of-1 Claude      | 72 / 244     | 29.5%  | 244        |
| Best-of-5 Claude      | 90 / 244     | 36.9%  | ~1,220     |
| **VERITAS Two-Phase** | **99 / 244** | **40.6%** | ~6,000  |

### combiBench (55 solvable problems, Lean 4)

Competition combinatorics problems (IMO 2000–2024, Brualdi textbook, Hackmath).
Evaluated on the 55-problem solvable subset (excludes 43 answer-finding problems with `sorry`).

| Method                | Solved      | Rate    |
|-----------------------|-------------|---------|
| Portfolio (heuristic) | 2 / 55      | 3.6%    |
| Best-of-1 Claude      | 1 / 55      | 1.8%    |
| Best-of-5 Claude      | 1 / 55      | 1.8%    |
| **VERITAS Two-Phase** | **4 / 55**  | **7.3%** |

## Architecture

VERITAS coordinates four specialized agents through Critic-Guided MCTS:

```
┌─────────────────────────────────────────────────────────────┐
│  Strategist → Tactician → Lean 4 Oracle                     │
│       ↑            ↑           ↓                            │
│    Retriever ←── Critic ←── ABCD Signals                    │
│                    ↓                                        │
│             MCTS Node Selection                             │
└─────────────────────────────────────────────────────────────┘
```

- **Strategist** (Claude Sonnet): high-level proof strategy (induction, cases, rewriting)
- **Tactician** (Claude Sonnet): tactic generation conditioned on strategy + failed history
- **Critic** (Claude Haiku): value estimation and policy priors for node selection
- **Retriever**: premise retrieval from Mathlib hint library
- **Lean 4 Oracle**: validates each tactic attempt, emits ABCD signals (syntax → typecheck → progress → complete)

**Two-Phase search**: Phase 1 runs a Best-of-N sweep to quickly solve easy theorems and collect failed tactics. Phase 2 runs MCTS with those failed tactics injected into the Tactician prompt, avoiding repeated dead ends.

## Setup

### 1. Prerequisites

- Python 3.10+
- Lean 4 via [elan](https://github.com/leanprover/elan)
- `ANTHROPIC_API_KEY` environment variable

```bash
# Install elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.elan/env
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the Lean project (downloads and compiles Mathlib ~20 min first time)

```bash
bash scripts/setup_lean_project.sh
```

This creates `lean_project/` with a pre-configured `lakefile.toml` that imports Mathlib.

### 4. Verify setup

```bash
python experiments/run_minif2f.py --mode portfolio --max-problems 5
```

Expected output: warmup OK (~60s first time), then 5 theorems evaluated.

## Running Benchmarks

### miniF2F

```bash
# Portfolio baseline (no LLM, parallelizable)
python experiments/run_minif2f.py --mode portfolio --workers 4

# Best-of-1 Claude (pass@1)
python experiments/run_minif2f.py --mode best_of_n --sweep-n 1

# Best-of-5 Claude (pass@5)
python experiments/run_minif2f.py --mode best_of_n --sweep-n 5

# VERITAS Two-Phase (main result)
python experiments/run_minif2f.py --mode veritas_two_phase --sweep-n 5 --theorem-timeout 300

# Full 244-problem evaluation (includes pre-solved "sorry" theorems)
python experiments/run_minif2f.py --mode veritas_two_phase --all-problems
```

### combiBench

```bash
python experiments/run_combibench.py --mode portfolio --workers 4
python experiments/run_combibench.py --mode best_of_1
python experiments/run_combibench.py --mode best_of_5
python experiments/run_combibench.py --mode veritas_two_phase --sweep-n 5
```

### LeanDojo (Mathlib library theorems — optional)

```bash
# Download data first (one-time, ~68 MB from Zenodo)
python scripts/download_leandojo.py

# Smoke test
python experiments/run_leandojo.py --mode portfolio --max-problems 5

# Full run
python experiments/run_leandojo.py --mode portfolio --workers 4
python experiments/run_leandojo.py --mode best_of_1
python experiments/run_leandojo.py --mode veritas_two_phase --sweep-n 5 --theorem-timeout 600
```

Note: VERITAS Two-Phase performs poorly on LeanDojo (Mathlib library automation theorems) because Claude lacks knowledge of specific Mathlib lemma names. Portfolio (31.5%) strongly outperforms Claude-based methods on this benchmark.

## Project Structure

```
VERITAS/
├── src/
│   ├── veritas.py                  # Core 4-agent MCTS framework
│   ├── agents/
│   │   └── claude_tactician.py     # Claude Sonnet/Haiku agent implementations
│   └── lean/
│       ├── lake_validator.py       # Lean 4 proof validator (Lake/Mathlib)
│       ├── minif2f_parser.py       # miniF2F dataset loader
│       ├── combibench_loader.py    # combiBench dataset loader
│       └── leandojo_loader.py      # LeanDojo dataset loader
├── experiments/
│   ├── benchmark_utils.py          # Shared infrastructure (baselines, runners)
│   ├── run_minif2f.py              # miniF2F benchmark entry point
│   ├── run_combibench.py           # combiBench benchmark entry point
│   └── run_leandojo.py             # LeanDojo benchmark entry point
├── scripts/
│   ├── setup_lean_project.sh       # Builds lean_project/ with Mathlib
│   └── download_leandojo.py        # Downloads LeanDojo benchmark from Zenodo
├── data/
│   ├── minif2f/                    # miniF2F Lean 4 files
│   └── combibench/                 # combiBench JSON
├── results/
│   └── result4.md                  # Detailed results and ablation analysis
├── lean_project/                   # Lake project (generated by setup script)
│   └── lakefile.toml
└── requirements.txt
```

## Programmatic Usage

```python
from src.veritas import VERITASSearch, SearchConfig, create_veritas
from src.lean.lake_validator import LakeValidator

validator = LakeValidator(project_dir="lean_project")

# Quick: use the factory function
search = create_veritas(validator)

# Or: full control over agents
from src.agents.claude_tactician import ClaudeTacticianAgent, ClaudeCriticAgent
from src.veritas import StrategistAgent, RetrieverAgent

search = VERITASSearch(
    strategist=StrategistAgent(),
    tactician=ClaudeTacticianAgent(),
    critic=ClaudeCriticAgent(),
    retriever=RetrieverAgent(),
    verifier=validator,
    config=SearchConfig(max_iterations=50, timeout_seconds=300),
)

result = search.search("theorem foo (n : Nat) : n + 0 = n")
if result["success"]:
    print("Proof:", result["proof"])
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (for Claude modes) | Anthropic API key |

## Citation

```bibtex
@article{veritas2025,
  title={VERITAS: Verification-Enhanced Reasoning with Integrated Tactic Agents and Search},
  year={2025},
}
```

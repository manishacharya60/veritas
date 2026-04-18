#!/usr/bin/env bash
# =============================================================================
# VERITAS Lean Project Setup
# =============================================================================
# Sets up a Lake/Mathlib project for theorem verification.
# Run this ONCE before running any benchmarks.
#
# What this does:
#   1. Fetches Mathlib4 via `lake update`
#   2. Downloads pre-built .olean files via `lake exe cache get`
#      (avoids recompiling Mathlib from scratch, saves ~2 hours)
#   3. Builds the minimal VeritasVerifier module to warm the olean cache
#
# After this completes, each LEAN validation call takes ~1-5s (not 60s+).
#
# Usage:
#   bash scripts/setup_lean_project.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LEAN_PROJECT="$PROJECT_ROOT/lean_project"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- Check prerequisites ----
info "Checking prerequisites..."

if ! command -v lean &>/dev/null; then
    error "lean not found. Install via elan:"
    echo "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
    echo "  source \$HOME/.elan/env"
    exit 1
fi

if ! command -v lake &>/dev/null; then
    error "lake not found (should come with lean/elan)"
    exit 1
fi

LEAN_VER=$(lean --version 2>&1 | head -1)
LAKE_VER=$(lake --version 2>&1 | head -1)
info "Lean: $LEAN_VER"
info "Lake: $LAKE_VER"

# ---- Enter project directory ----
cd "$LEAN_PROJECT"
info "Working in: $LEAN_PROJECT"

# ---- lake update (fetch Mathlib) ----
info "Fetching Mathlib (lake update)..."
info "This resolves the correct Mathlib version for your Lean installation."
lake update

# ---- Download pre-built oleans ----
info "Downloading pre-built Mathlib oleans (lake exe cache get)..."
info "This avoids recompiling Mathlib (~2h). Download size: ~1-3 GB."
if lake exe cache get; then
    info "Olean cache downloaded successfully."
else
    warn "cache get failed. Falling back to local build."
    warn "This may take 1-3 hours on first run. Subsequent runs will be fast."
fi

# ---- Build VeritasVerifier module ----
info "Building VeritasVerifier module (warms olean cache)..."
if lake build VeritasVerifier; then
    info "Build successful."
else
    error "Build failed. Check output above for errors."
    echo ""
    echo "Common fixes:"
    echo "  - Lean version mismatch: ensure lean-toolchain matches installed lean"
    echo "  - Network issues: retry 'lake exe cache get' manually"
    exit 1
fi

# ---- Smoke test ----
info "Running smoke test..."
SMOKE_FILE="$LEAN_PROJECT/_smoke_test.lean"
cat > "$SMOKE_FILE" << 'EOF'
import Mathlib
open BigOperators Real Nat

theorem smoke_1 : 1 + 1 = 2 := by norm_num
theorem smoke_2 (n : ℕ) : n + 0 = n := by simp
theorem smoke_3 (a b : ℝ) (h : a = b) : b = a := by linarith
EOF

if lean --json "$SMOKE_FILE" 2>&1 | grep -q '"severity":"error"'; then
    warn "Smoke test had errors — check Lean/Mathlib version compatibility."
else
    info "Smoke test passed."
fi
rm -f "$SMOKE_FILE"

echo ""
info "Setup complete! You can now run benchmarks:"
echo ""
echo "  # Quick test (5 problems, portfolio only):"
echo "  python experiments/run_minif2f.py --mode portfolio --max-problems 5"
echo ""
echo "  # Full miniF2F portfolio baseline:"
echo "  python experiments/run_minif2f.py --mode portfolio"
echo ""
echo "  # Full VERITAS vs portfolio comparison:"
echo "  python experiments/run_minif2f.py --mode both"
echo ""
echo "  # Full ablation study:"
echo "  python experiments/run_minif2f.py --mode ablation"

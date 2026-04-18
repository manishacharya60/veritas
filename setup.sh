#!/bin/bash
# VERITAS Setup Script
# Run this script to set up the development environment

set -e

echo "=========================================="
echo "VERITAS Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install LEAN 4 via elan (if not installed)
if ! command -v lean &> /dev/null; then
    echo "Installing LEAN 4 via elan..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    source ~/.profile
else
    echo "LEAN already installed: $(lean --version)"
fi

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/test_veritas_integration.py -v --tb=short

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run experiments:"
echo "  python experiments/simple_comparison.py"
echo ""

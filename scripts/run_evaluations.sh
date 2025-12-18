#!/bin/bash
# Run all promptfoo evaluations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config/promptfoo/evaluations"

echo "========================================"
echo "Financial RAG Evaluation Suite"
echo "========================================"
echo ""

# Check if promptfoo is installed
if ! command -v promptfoo &> /dev/null; then
    echo "Error: promptfoo is not installed"
    echo "Install it with: npm install -g promptfoo"
    exit 1
fi

# Function to run a single evaluation
run_eval() {
    local config_file=$1
    local name=$(basename "$config_file" .yaml)

    echo "----------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------"

    promptfoo eval -c "$config_file" --output "$PROJECT_ROOT/results/${name}_$(date +%Y%m%d_%H%M%S).json"

    echo ""
}

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [all|1|2|3|4|5|6|redteam]"
    echo ""
    echo "Options:"
    echo "  all     - Run all evaluations"
    echo "  1       - Model comparison"
    echo "  2       - RAG retriever comparison"
    echo "  3       - RAG context quality"
    echo "  4       - RAG generation quality"
    echo "  5       - Prompt evaluation"
    echo "  6       - Red team testing"
    echo "  redteam - Red team testing (same as 6)"
    echo ""
    exit 0
fi

case "$1" in
    all)
        for config in "$CONFIG_DIR"/*.yaml; do
            run_eval "$config"
        done
        ;;
    1)
        run_eval "$CONFIG_DIR/01_model_comparison.yaml"
        ;;
    2)
        run_eval "$CONFIG_DIR/02_rag_retriever.yaml"
        ;;
    3)
        run_eval "$CONFIG_DIR/03_rag_context.yaml"
        ;;
    4)
        run_eval "$CONFIG_DIR/04_rag_generation.yaml"
        ;;
    5)
        run_eval "$CONFIG_DIR/05_prompt_evaluation.yaml"
        ;;
    6|redteam)
        echo "Running red team evaluation..."
        promptfoo redteam run -c "$CONFIG_DIR/06_redteam.yaml"
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
esac

echo "========================================"
echo "Evaluations complete!"
echo "View results with: promptfoo view"
echo "========================================"

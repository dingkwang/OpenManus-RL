#!/bin/bash
#
# Run AlfWorld trajectory collection with LLM integration
# Usage: ./run_alfworld.sh [steps] [batch_size]
#

# Configuration
API_ENDPOINT="${OAI_ENDPOINT:-}"
API_KEY="${OAI_KEY:-}"
MAX_STEPS="${1:-10}"
BATCH_SIZE="${2:-1}"

# Set environment
export OAI_ENDPOINT="$API_ENDPOINT"
export OAI_KEY="$API_KEY"

# Display configuration
echo "========================================"
echo "AlfWorld Trajectory Collection"
echo "========================================"
echo "API Endpoint: ${API_ENDPOINT:0:30}..."
echo "Steps: $MAX_STEPS"
echo "Batch: $BATCH_SIZE"
echo ""

# Run trajectory collection
cd "$(dirname "$0")/.." || exit 1
python test/alfworld_rollout.py --steps "$MAX_STEPS" --batch "$BATCH_SIZE"

# Check for generated trajectories
if [ -d "trajectories" ]; then
    echo ""
    echo "Generated trajectories:"
    ls -lh trajectories/*.json 2>/dev/null | tail -5
fi
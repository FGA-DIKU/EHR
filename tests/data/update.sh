#!/bin/bash

# Ensure we're in the project root directory
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
if [ -z "$PROJECT_ROOT" ]; then
    echo "Error: Not in a git repository. Please run from project directory."
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Updating test data from project root: $PROJECT_ROOT"

# 1. Remove existing test data
echo "Removing existing test data..."
rm -rf ./tests/data/features/
rm -rf ./tests/data/outcomes/

# 2. Regenerate features test data
echo "Regenerating features test data..."
python -m corebehrt.main.create_data \
    --config ./tests/pipeline_configs/create_data.yaml

# 3. Regenerate outcomes test data
echo "Regenerating outcomes test data..."
python -m corebehrt.main.create_outcomes \
    --config ./tests/pipeline_configs/create_outcomes.yaml

echo "Test data update complete!"
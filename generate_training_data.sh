#!/bin/bash

# Ensure we are in the right virtual environment
if [[ "$VIRTUAL_ENV" != *"$(pwd)/venv"* ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/raw_data
mkdir -p data/mlx_format_qwen_v10

# Generate synthetic data
echo "Generating synthetic training data..."
python data/generate_expanded_dataset.py \
    --output-dir data/raw_data \
    --num-examples 2000

# Get the most recent files
LATEST_TRAIN=$(ls -t data/raw_data/train_*.json | head -n1)
LATEST_VALID=$(ls -t data/raw_data/validation_*.json | head -n1)

# Process the data for Qwen format
echo "Processing data for Qwen format..."
python data/preprocess_for_qwen.py \
    --input-dir data/raw_data \
    --output-dir data/mlx_format_qwen_v10

echo "
Data generation completed! 🎉

Generated files:
- Raw data: $LATEST_TRAIN, $LATEST_VALID
- Processed data: data/mlx_format_qwen_v10/train.jsonl, data/mlx_format_qwen_v10/valid.jsonl

Next steps:
1. Review the generated data if needed
2. Preprocess the data:
   ./preprocess.sh
3. Start training:
   ./train.sh
"
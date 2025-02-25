#!/bin/bash

# Ensure we're in the virtual environment
if [[ "$VIRTUAL_ENV" != *"$(pwd)/venv"* ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/raw_data
mkdir -p data/qwen_format_v2

# Generate synthetic data
echo "Generating synthetic training data..."
python data/generate_expanded_dataset.py \
    --output-dir data/raw_data \
    --num-examples 2000

# Process the data with new format
echo "Processing data with improved format..."
python data/preprocess_for_qwen_v2.py \
    --input-dir data/raw_data \
    --output-dir data/qwen_format_v2

echo "
Data generation completed! 🎉

Next steps:
1. Review the generated data in data/qwen_format_v2/
2. Start training with the new data:
   python scripts/train_lora.py --config config/lora_qwen_config.json
" 
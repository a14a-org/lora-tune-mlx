#!/bin/bash

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create necessary directories
mkdir -p data/multi_step
mkdir -p data/qwen_format_v3

echo "Generating multi-step conversation dataset..."
python3 data/generate_expanded_dataset.py --num_train 190 --num_valid 20 --output_dir data/multi_step

echo "Preprocessing training data..."
python3 data/preprocess_for_qwen_v3.py data/multi_step/train.json data/qwen_format_v3/train.jsonl

echo "Preprocessing validation data..."
python3 data/preprocess_for_qwen_v3.py data/multi_step/valid.json data/qwen_format_v3/valid.jsonl

echo "Data generation and preprocessing complete!" 
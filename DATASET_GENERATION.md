# Dataset Generation Instructions

This document outlines the steps to generate and preprocess the training dataset.

## Manual Steps

1. First, ensure all necessary directories exist:
```bash
mkdir -p data/multi_step
mkdir -p data/qwen_format_v3
```

2. Generate the dataset using `data/generate_expanded_dataset.py`:
```bash
# This will generate:
# - data/multi_step/train.json (190 examples)
# - data/multi_step/valid.json (20 examples)
python3 data/generate_expanded_dataset.py --num_train 190 --num_valid 20 --output_dir data/multi_step
```

3. Preprocess the generated data using `data/preprocess_for_qwen_v3.py`:
```bash
# This will create:
# - data/qwen_format_v3/train.jsonl
# - data/qwen_format_v3/valid.jsonl
# - data/qwen_format_v3/processing_stats_train.json
# - data/qwen_format_v3/processing_stats_valid.json
python3 data/preprocess_for_qwen_v3.py data/multi_step/train.json data/qwen_format_v3/train.jsonl
python3 data/preprocess_for_qwen_v3.py data/multi_step/valid.json data/qwen_format_v3/valid.jsonl
```

## Automated Script

All of the above steps are combined in the `regenerate_data_v3.sh` script. You can run everything at once with:

```bash
./regenerate_data_v3.sh
```

## Output Files

After running either the manual steps or the automated script, you should have:

1. Raw data:
   - `data/multi_step/train.json` (~303KB)
   - `data/multi_step/valid.json` (~31KB)

2. Preprocessed data:
   - `data/qwen_format_v3/train.jsonl` (~262KB)
   - `data/qwen_format_v3/valid.jsonl` (~27KB)
   - `data/qwen_format_v3/processing_stats_train.json` (~131B)
   - `data/qwen_format_v3/processing_stats_valid.json` (~128B)

The preprocessed `.jsonl` files are the ones that should be used for training. 
# Qwen2.5-7B LoRA Fine-tuning for Home Automation

This repository contains the code and data for fine-tuning the Qwen2.5-7B model using LoRA (Low-Rank Adaptation) for home automation tasks.

## Requirements

- Python 3.13
- macOS or Linux operating system
- At least 32GB of RAM (48GB recommended)
- MLX-compatible device (Apple Silicon for macOS)

## Project Structure

```
.
├── README.md                    # Project documentation
├── setup.sh                     # Setup script for environment
├── checkpoints/                 # Directory for trained model checkpoints
├── config/
│   └── lora_qwen_config.json   # LoRA and training configuration
├── data/
│   └── v1.1/
│       ├── mlx_format_qwen_v10/  # Training data
│       │   ├── train.jsonl
│       │   └── valid.jsonl
│       ├── generate_expanded_dataset.py  # Data generation script
│       └── preprocess_for_qwen.py       # Data preprocessing script
├── docs/
│   ├── training_plan.md        # Training methodology
│   └── training_progress.md    # Training progress and results
├── scripts/
│   ├── train_lora.py          # Training script
│   └── test_simple.py         # Testing script
├── models/
│   └── lora_linear.py         # LoRA model components
├── utils/
│   └── __init__.py            # Utility functions
└── requirements.txt           # Python dependencies
```

## Setup

### Quick Setup

Run the setup script to automatically create a virtual environment and install dependencies:

```bash
./setup.sh
```

This script will:
1. Check for Python 3.13
2. Create and activate a virtual environment
3. Install all required dependencies
4. Provide next steps for training

### Manual Setup

If you prefer to set up manually:

1. Create and activate a virtual environment:
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Generate synthetic training data:
   ```bash
   python data/v1.1/generate_expanded_dataset.py \
     --output-dir data/v1.1/raw_data
   ```

2. Preprocess data for Qwen model:
   ```bash
   python data/v1.1/preprocess_for_qwen.py \
     --input-dir data/v1.1/raw_data \
     --output-dir data/v1.1/mlx_format_qwen_v10
   ```

## Training

1. Configure training parameters in `config/lora_qwen_config.json`

2. Run training:
   ```bash
   python scripts/train_lora.py --config config/lora_qwen_config.json
   ```

## Testing

Test the fine-tuned model:
```bash
python scripts/test_simple.py
```

## Model Architecture

The project uses LoRA to fine-tune the Qwen2.5-7B model by adding low-rank adaptation layers to the attention components. Key configurations:

- LoRA rank (r): 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-4

## Training Data

The training data consists of home automation scenarios including:
- Light control
- Temperature management
- Weather queries
- Multi-turn conversations

Each example follows a specific format with system prompts, user queries, and tool responses.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
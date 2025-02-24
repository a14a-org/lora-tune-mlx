# Qwen2.5-7B LoRA Fine-tuning for Home Automation

This repository contains the code and data for fine-tuning the Qwen2.5-7B model using LoRA (Low-Rank Adaptation) for home automation tasks. The model is specifically trained to handle home automation commands and queries using a structured tool-calling format.

## Overview

The project fine-tunes the [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model using MLX, Apple's machine learning framework. The fine-tuned model can:

- Control smart home devices (lights, thermostats)
- Provide weather information
- Handle multi-turn conversations
- Process structured tool calls in XML format
- Manage error cases and provide appropriate feedback

Example interaction:
```
User: Can you turn on the living room lights?
Assistant: I'll help you with that.
<tool name='control_lights'>room="living room" action="on"</tool>
System: <tool_response name='control_lights'>{"status":"success","message":"Lights turned on"}</tool_response>
Assistant: The living room lights have been turned on successfully.
```

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
├── preprocess.sh               # Script to preprocess training data
├── train.sh                    # Script to run model training
├── generate_training_data.sh    # Script to generate training data
├── checkpoints/                 # Directory for trained model checkpoints
├── config/
│   └── lora_qwen_config.json   # LoRA and training configuration
├── data/
│   ├── raw_data/               # Generated synthetic training data
│   ├── mlx_format_qwen/        # Processed data for Qwen model
│   ├── generate_expanded_dataset.py  # Data generation script
│   └── preprocess_for_qwen.py       # Data preprocessing script
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

## Tool Format

The model uses a structured XML-style format for tool calls:

```xml
<tool_definition name='control_lights'>
  description: Control smart lights in a room
  parameters:
    - room (string): The room where the lights are located
    - action (string, values: ['on', 'off', 'dim']): The action to perform
    - brightness (integer, optional): Brightness level (0-100)
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
</tool_definition>
```

Available tools:
1. `control_lights`: Control smart lights in rooms
2. `get_weather`: Get weather information for locations
3. `set_thermostat`: Set temperature for the thermostat
4. `set_thermostat_delta`: Adjust temperature relative to current setting

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

- Base model: Qwen2.5-7B-Instruct (BF16 version)
- LoRA rank (r): 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-4
- Training steps: 500
- Batch size: 8
- Sequence length: 2048

## Training Data

The training data consists of home automation scenarios including:
- Light control (on/off/dim)
- Temperature management (set/adjust)
- Weather queries
- Multi-turn conversations
- Error handling scenarios

Each example follows a specific format with:
- System prompts defining available tools
- User queries in natural language
- Assistant responses with tool calls
- System responses with tool results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) by Alibaba Cloud
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [MLX-Examples](https://github.com/ml-explore/mlx-examples) for reference implementations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research or application, please cite:

```bibtex
@misc{lora-tune-mlx,
  author = {a14a-org},
  title = {LoRA Fine-tuning for MLX},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/a14a-org/lora-tune-mlx}
}
``` 
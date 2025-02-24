#!/bin/bash

# Default values
CONFIG_FILE="config/lora_qwen_config.json"
CHECKPOINT=""
START_STEP=0
SEED=42

# Function to display usage information
show_help() {
    echo "Usage: ./train.sh [options]"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -c, --config FILE          Specify config file (default: config/lora_qwen_config.json)"
    echo "  -r, --resume CHECKPOINT    Resume training from checkpoint file"
    echo "  -s, --start-step STEP      Specify start step when resuming (required with --resume)"
    echo "  --seed SEED                Set random seed (default: 42)"
    echo
    echo "Examples:"
    echo "  ./train.sh                                  # Start new training"
    echo "  ./train.sh -r checkpoints/checkpoint-100.npz -s 100  # Resume from checkpoint"
    echo "  ./train.sh --config custom_config.json      # Use custom config"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--resume)
            CHECKPOINT="$2"
            shift 2
            ;;
        -s|--start-step)
            START_STEP="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate arguments
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ -n "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

if [ -n "$CHECKPOINT" ] && [ "$START_STEP" -eq 0 ]; then
    echo "Error: --start-step is required when resuming from checkpoint"
    exit 1
fi

# Ensure we're in the right virtual environment
if [[ "$VIRTUAL_ENV" != *"$(pwd)/venv"* ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Add current directory to Python path
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Prepare command
CMD="python scripts/train_lora.py --config $CONFIG_FILE --seed $SEED"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --resume-from $CHECKPOINT --start-step $START_STEP"
fi

# Print training configuration
echo "Starting training with configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Seed: $SEED"
if [ -n "$CHECKPOINT" ]; then
    echo "  Resuming from: $CHECKPOINT"
    echo "  Start step: $START_STEP"
fi
echo

# Run training
echo "Running: $CMD"
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully! 🎉"
    echo
    echo "To test the model, run:"
    echo "  python scripts/test_simple.py"
else
    echo "Training failed with error code $?"
fi

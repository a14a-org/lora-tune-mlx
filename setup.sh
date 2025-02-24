#!/bin/bash

# Function to check Python version
check_python_version() {
    if command -v python3.13 &>/dev/null; then
        echo "✓ Python 3.13 found"
        PYTHON_CMD=python3.13
    else
        echo "✗ Python 3.13 not found"
        echo "Please install Python 3.13 first. You can use:"
        echo "  - brew install python@3.13 (on macOS)"
        echo "  - pyenv install 3.13 (using pyenv)"
        echo "  - Or download from python.org"
        exit 1
    fi
}

# Function to create and activate virtual environment
setup_venv() {
    echo "Setting up virtual environment..."
    if [ -d "venv" ]; then
        echo "Found existing venv, removing it..."
        rm -rf venv
    fi
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Verify we're in the right venv
    if [[ "$VIRTUAL_ENV" != *"$(pwd)/venv" ]]; then
        echo "✗ Failed to activate virtual environment"
        exit 1
    fi
    echo "✓ Virtual environment activated"
}

# Function to install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install requirements
    if pip install -r requirements.txt; then
        echo "✓ Dependencies installed successfully"
    else
        echo "✗ Failed to install dependencies"
        exit 1
    fi
}

# Main setup process
echo "Starting setup process..."
check_python_version
setup_venv
install_dependencies

echo "
Setup completed successfully! 🎉

To start using the environment:
1. Activate the virtual environment:
   source venv/bin/activate

2. Generate training data:
   python data/v1.1/generate_expanded_dataset.py --output-dir data/v1.1/raw_data

3. Preprocess the data:
   python data/v1.1/preprocess_for_qwen.py --input-dir data/v1.1/raw_data --output-dir data/v1.1/mlx_format_qwen_v10

4. Start training:
   python scripts/train_lora.py --config config/lora_qwen_config.json
" 
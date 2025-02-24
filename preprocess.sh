#!/bin/bash

# Default values
INPUT_DIR="data/raw_data"
OUTPUT_DIR="data/mlx_format_qwen"

# Function to display usage information
show_help() {
    echo "Usage: ./preprocess.sh [options]"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -i, --input-dir DIR        Input directory containing raw data (default: data/raw_data)"
    echo "  -o, --output-dir DIR       Output directory for processed data (default: data/mlx_format_qwen)"
    echo "  -v, --version DATE         Specific version to process (format: YYYYMMDD_HHMMSS)"
    echo
    echo "Examples:"
    echo "  ./preprocess.sh                           # Interactive version selection"
    echo "  ./preprocess.sh -v 20250224_175409       # Process specific version"
    echo "  ./preprocess.sh -i custom/input -o custom/output  # Use custom directories"
}

# Function to list available versions
list_versions() {
    local dir="$1"
    echo "Available versions:"
    echo
    
    # Get unique timestamps from train files
    versions=$(ls "$dir"/train_*.json 2>/dev/null | sed -E 's/.*train_([0-9]{8}_[0-9]{6}).json/\1/' | sort -u)
    
    if [ -z "$versions" ]; then
        echo "No data files found in $dir"
        exit 1
    fi
    
    # Display each version with its files
    i=1
    for version in $versions; do
        echo "[$i] Version: $version"
        echo "    Files:"
        echo "    - $dir/train_$version.json"
        echo "    - $dir/validation_$version.json"
        echo
        i=$((i+1))
    done
}

# Function to convert JSON to JSONL using Python
convert_to_jsonl() {
    local input_file="$1"
    local output_file="$2"
    python3 -c "
import json
with open('$input_file', 'r') as f_in:
    data = json.load(f_in)
    with open('$output_file', 'w') as f_out:
        # Handle nested data array structure
        if isinstance(data, dict) and 'data' in data:
            items = data['data']
        elif isinstance(data, list):
            items = data
        else:
            items = [data]
        
        # Write each item as a line
        for item in items:
            f_out.write(json.dumps(item) + '\n')
"
}

# Parse command line arguments
VERSION=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate directories
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# If no version specified, show interactive selection
if [ -z "$VERSION" ]; then
    echo "No version specified. Please select a version to process:"
    echo
    list_versions "$INPUT_DIR"
    echo
    read -p "Enter the number of the version to process: " selection
    
    # Get the selected version
    VERSION=$(ls "$INPUT_DIR"/train_*.json 2>/dev/null | \
              sed -E 's/.*train_([0-9]{8}_[0-9]{6}).json/\1/' | \
              sort -u | sed -n "${selection}p")
    
    if [ -z "$VERSION" ]; then
        echo "Invalid selection"
        exit 1
    fi
fi

# Validate version files exist
if [ ! -f "$INPUT_DIR/train_$VERSION.json" ] || [ ! -f "$INPUT_DIR/validation_$VERSION.json" ]; then
    echo "Error: Could not find data files for version $VERSION"
    echo "Expected files:"
    echo "  - $INPUT_DIR/train_$VERSION.json"
    echo "  - $INPUT_DIR/validation_$VERSION.json"
    exit 1
fi

# Ensure we're in the right virtual environment
if [[ "$VIRTUAL_ENV" != *"$(pwd)/venv"* ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Print preprocessing configuration
echo "Starting preprocessing with configuration:"
echo "  Input directory:  $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Version:         $VERSION"
echo "  Input files:"
echo "    - $INPUT_DIR/train_$VERSION.json"
echo "    - $INPUT_DIR/validation_$VERSION.json"
echo

# Create temporary directory for JSONL conversion
TEMP_DIR="$INPUT_DIR/temp_$VERSION"
mkdir -p "$TEMP_DIR"

# Convert JSON to JSONL format
echo "Converting JSON to JSONL format..."
convert_to_jsonl "$INPUT_DIR/train_$VERSION.json" "$TEMP_DIR/train.jsonl"
convert_to_jsonl "$INPUT_DIR/validation_$VERSION.json" "$TEMP_DIR/valid.jsonl"

# Run preprocessing
echo "Running preprocessing..."
CMD="python data/preprocess_for_qwen.py --input-dir $TEMP_DIR --output-dir $OUTPUT_DIR"
echo "Running: $CMD"
$CMD

# Check exit status
PREPROCESS_STATUS=$?

# Clean up temporary directory
rm -rf "$TEMP_DIR"

# Check if preprocessing was successful
if [ $PREPROCESS_STATUS -eq 0 ]; then
    echo "Preprocessing completed successfully! 🎉"
    echo
    echo "Generated files:"
    echo "  - $OUTPUT_DIR/train.jsonl"
    echo "  - $OUTPUT_DIR/valid.jsonl"
    echo
    echo "You can now start training with:"
    echo "  ./train.sh"
else
    echo "Preprocessing failed with error code $PREPROCESS_STATUS"
    exit $PREPROCESS_STATUS
fi

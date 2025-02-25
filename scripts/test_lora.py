#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

import sys
sys.path.append('.')
from models import LoRALinear
from utils import *

def load_config(config_path: str):
    """Load the LoRA configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_lora_weights(model, checkpoint_path: str):
    """Load LoRA weights from checkpoint."""
    print(f"Loading LoRA weights from {checkpoint_path}")
    weights = mx.load(checkpoint_path)
    
    # Apply weights to LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"Applying weights to LoRA module: {name}")
            if f"{name}.lora_A" in weights and f"{name}.lora_B" in weights:
                module.lora_A = weights[f"{name}.lora_A"]
                module.lora_B = weights[f"{name}.lora_B"]
                print(f"  Applied weights to {name}")
            else:
                print(f"  WARNING: Could not find weights for {name}")

def main():
    parser = argparse.ArgumentParser(description="Test LoRA model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for testing")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print("Loading base model...")
    
    # Load base model and tokenizer
    model, tokenizer = load(config["model_name"])
    
    # Apply LoRA layers
    print("Applying LoRA layers...")
    trainable_params = apply_lora_to_model(model, config["lora_config"])
    
    # Load checkpoint
    load_lora_weights(model, args.checkpoint)
    
    # Prepare prompt
    prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate response
    print("\nGenerating response...")
    print(f"Prompt: {args.prompt}")
    print("\nResponse:")
    
    tokens = tokenizer.encode(prompt)
    tokens = mx.array([tokens])
    
    for token in generate(tokens, model, temp=0.7):
        text = tokenizer.decode([token])
        print(text, end="", flush=True)
        if "<|im_end|>" in text:
            break
    print("\n")

if __name__ == "__main__":
    main() 
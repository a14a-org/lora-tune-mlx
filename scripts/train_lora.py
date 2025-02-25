import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

# Fix the import path
import sys
sys.path.append('.')  # Add project root to Python path
from utils import *  # Import utils module

import utils as lora_utils
from models import LoRALinear
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention


def load_config(config_path: str):
    """Load the LoRA configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


class Dataset:
    """Light-weight wrapper to hold lines from a jsonl file"""
    def __init__(self, path: Path):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]

    def __getitem__(self, idx: int):
        # Convert chat format to text using the Qwen2.5 chat template
        messages = self._data[idx]["messages"]
        
        # Format each message according to Qwen2.5's style
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Start each message with the correct token
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Add final assistant token to guide the model
        if not text.endswith("<|im_start|>assistant\n"):
            text += "<|im_start|>assistant\n"
        
        return text

    def __len__(self):
        return 0 if self._data is None else len(self._data)


def load_datasets(config):
    """Load train and validation datasets from config."""
    train = Dataset(Path(config["data_config"]["train_file"]))
    valid = Dataset(Path(config["data_config"]["validation_file"]))
    
    if len(train) == 0:
        raise ValueError("Training set not found or empty")
    if len(valid) == 0:
        raise ValueError("Validation set not found or empty")
    
    return train, valid


def loss(model, inputs, targets, lengths):
    """Calculate loss for the model."""
    # Create attention mask
    batch_size = inputs.shape[0]
    seq_length = inputs.shape[1]
    attention_mask = mx.arange(seq_length)[None, :] < lengths[:, None]
    
    # Forward pass
    logits = model(inputs, attention_mask)
    logits = logits.astype(mx.float32)
    
    # Calculate loss only on non-padded tokens
    ce = nn.losses.cross_entropy(logits, targets, reduction="none") * attention_mask
    ntoks = attention_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, max_seq_length, train=False):
    """Iterate over batches of data."""
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [min(len(x), max_seq_length) for x in batch]
            
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
            for j in range(batch_size):
                batch_arr[j, :lengths[j]] = batch[j][:lengths[j]]
            
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(model, dataset, compute_loss, tokenizer, config):
    """Evaluate the model on the given dataset."""
    print("Starting evaluation...")
    all_losses = []
    ntokens = 0
    batch_size = config["training_config"]["batch_size"]
    max_seq_length = config["data_config"]["max_seq_length"]
    max_eval_batches = 10  # Limit number of validation batches

    batch_count = 0
    for batch in iterate_batches(dataset, tokenizer, batch_size, max_seq_length):
        print(f"Processing validation batch {batch_count + 1}...")
        losses, toks = compute_loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()
        
        batch_count += 1
        if batch_count >= max_eval_batches:
            print(f"Reached max validation batches ({max_eval_batches})")
            break

    val_loss = np.sum(all_losses) / ntokens
    print(f"Evaluation complete. Loss: {val_loss:.4f}")
    return val_loss


def train(model, train_set, val_set, tokenizer, config, output_dir, trainable_params, start_step=0):
    """Train the model using the provided configuration."""
    training_config = config["training_config"]
    
    # Create optimizer with learning rate
    optimizer = optim.Adam(learning_rate=training_config["learning_rate"])
    
    # Create loss function with weight decay
    def compute_loss(model, inputs, targets, lengths):
        # Forward pass
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        
        # Calculate loss only on non-padded tokens
        attention_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none") * attention_mask
        ntoks = attention_mask.sum()
        ce = ce.sum() / ntoks
        
        # Add weight decay
        if training_config["weight_decay"] > 0:
            # Add L2 regularization only for trainable parameters
            l2_loss = sum(mx.sum(p * p) for _, p in trainable_params)
            ce = ce + 0.5 * training_config["weight_decay"] * l2_loss
        
        return ce, ntoks
    
    # Create value_and_grad function that only computes gradients for trainable parameters
    loss_value_and_grad = nn.value_and_grad(model, compute_loss)
    
    def train_step(model, inputs, targets, lengths):
        (loss, toks), grads = loss_value_and_grad(model, inputs, targets, lengths)
        
        # Update only trainable parameters
        for name, param in trainable_params:
            # Find corresponding gradient
            grad = None
            for k, v in grads.items():
                if k.endswith(name.split('.')[-1]):
                    grad = v
                    break
            if grad is not None:
                param -= optimizer.learning_rate * grad
        
        return loss, toks
    
    num_epochs = training_config["num_epochs"]
    batch_size = training_config["batch_size"]
    max_seq_length = config["data_config"]["max_seq_length"]
    max_steps = training_config.get("max_steps", float('inf'))  # Get max_steps or set to infinity if not specified
    
    # Training loop
    step = start_step
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_losses = []
        ntokens = 0
        
        for batch in iterate_batches(train_set, tokenizer, batch_size, max_seq_length, train=True):
            loss, toks = train_step(model, *batch)
            train_losses.append((loss * toks).item())
            ntokens += toks.item()
            step += 1
            
            train_loss = np.sum(train_losses) / ntokens
            print(f"Step {step}: Train loss: {train_loss:.4f}")
            
            # Save checkpoint at step 100
            if step == 100:
                print("\nReached step 100, saving checkpoint...")
                model.eval()
                val_loss = evaluate(model, val_set, compute_loss, tokenizer, config)
                print(f"Validation loss at step 100: {val_loss:.4f}")
                save_checkpoint(model, optimizer, step, val_loss, output_dir)
                model.train()
            
            if step >= max_steps:  # Check if we've reached max_steps
                print(f"\nReached max_steps ({max_steps}), stopping training.")
                # Save final checkpoint
                model.eval()
                print("\nRunning final validation...")
                val_loss = evaluate(model, val_set, compute_loss, tokenizer, config)
                print(f"\nFinal validation loss: {val_loss:.4f}")
                print("\nSaving final checkpoint...")
                try:
                    save_checkpoint(model, optimizer, step, val_loss, output_dir)
                    print("\nCheckpoint saved successfully.")
                except Exception as e:
                    print(f"\nERROR: Failed to save checkpoint: {e}")
                print("\nTraining completed.")
                return
            
            if step > 0 and step % training_config["eval_steps"] == 0:
                # Validation
                model.eval()
                val_loss = evaluate(model, val_set, compute_loss, tokenizer, config)
                print(f"Validation loss: {val_loss:.4f}")
                
                # Save checkpoint if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, val_loss, output_dir)
                
                model.train()
                train_losses = []
                ntokens = 0
    
    # Final validation
    model.eval()
    final_val_loss = evaluate(model, val_set, compute_loss, tokenizer, config)
    print(f"\nFinal validation loss: {final_val_loss:.4f}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, step, final_val_loss, output_dir)


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer using MLX-LM."""
    model, tokenizer = load(model_name)
    return model, tokenizer


def apply_lora_to_model(model, lora_config):
    """Apply LoRA to the model."""
    print("\nApplying LoRA layers")
    
    # Print model structure for debugging
    print("\nModel structure:")
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")
    
    # Get the transformer layers
    transformer = model.model
    if hasattr(transformer, "layers"):
        layers = transformer.layers
    elif hasattr(transformer, "h"):
        layers = transformer.h
    else:
        raise ValueError("Could not find layers in model structure")
    
    # Apply LoRA to attention layers
    for layer in layers:
        # Get the original Q and V projections
        if "q_proj" in lora_config["target_modules"]:
            q_proj = layer.self_attn.q_proj
            # Check if original layer has bias
            has_bias = hasattr(q_proj, "bias") and q_proj.bias is not None
            q_lora = LoRALinear(
                q_proj.weight.shape[1],
                q_proj.weight.shape[0],
                r=lora_config["r"],
                alpha=lora_config["alpha"],
                dropout=lora_config["dropout"],
                bias=has_bias  # Set based on original layer
            )
            # Copy weights and bias
            q_lora.weight = q_proj.weight
            if has_bias:
                q_lora.bias = q_proj.bias
            layer.self_attn.q_proj = q_lora
            print(f"Applied LoRA to q_proj in layer")
        
        if "v_proj" in lora_config["target_modules"]:
            v_proj = layer.self_attn.v_proj
            has_bias = hasattr(v_proj, "bias") and v_proj.bias is not None
            v_lora = LoRALinear(
                v_proj.weight.shape[1],
                v_proj.weight.shape[0],
                r=lora_config["r"],
                alpha=lora_config["alpha"],
                dropout=lora_config["dropout"],
                bias=has_bias
            )
            v_lora.weight = v_proj.weight
            if has_bias:
                v_lora.bias = v_proj.bias
            layer.self_attn.v_proj = v_lora
            print(f"Applied LoRA to v_proj in layer")
    
    # Create trainable parameter list
    trainable_params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"Found LoRA module to train: {name}")
            trainable_params.extend([
                (f"{name}.lora_A", module.lora_A),
                (f"{name}.lora_B", module.lora_B)
            ])
    
    print(f"Total trainable LoRA parameters: {len(trainable_params)}")
    return trainable_params


def save_checkpoint(model, optimizer, step, val_loss, output_dir):
    """Save a checkpoint of the model."""
    print(f"\nAttempting to save checkpoint to {output_dir}")
    checkpoint_path = output_dir / f"checkpoint-{step}.npz"
    print(f"Full checkpoint path: {checkpoint_path}")
    
    # Save only the LoRA parameters
    lora_state = {}
    print("\nFinding LoRA modules...")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"Found LoRA module: {name}")
            lora_state[f"{name}.lora_A"] = module.lora_A
            lora_state[f"{name}.lora_B"] = module.lora_B
            print(f"  lora_A shape: {module.lora_A.shape}")
            print(f"  lora_B shape: {module.lora_B.shape}")
    
    print(f"\nFound {len(lora_state)} LoRA parameters to save")
    if len(lora_state) == 0:
        print("WARNING: No LoRA parameters found to save!")
        return
    
    try:
        mx.savez(checkpoint_path, **lora_state)
        print(f"Successfully saved checkpoint to {checkpoint_path}")
        print(f"Checkpoint saved at step {step} with validation loss: {val_loss:.4f}")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train a model with LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--adapter-path", type=str, default="adapters.npz", 
                       help="Path to save adapter weights")
    parser.add_argument("--resume-from", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--start-step", type=int, help="Step number to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    print("Loading pretrained model")
    
    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    
    # Apply LoRA and get trainable parameters
    print("Applying LoRA layers")
    trainable_params = apply_lora_to_model(model, config["lora_config"])
    
    # Load datasets
    print("Loading datasets")
    train_set, val_set = load_datasets(config)
    
    # Create output directory
    output_dir = Path("lora_checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    # Train the model
    print("Starting training")
    train(model, train_set, val_set, tokenizer, config, output_dir, trainable_params, start_step=args.start_step or 0)
    
    print("Training completed")


if __name__ == "__main__":
    main() 
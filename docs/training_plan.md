# Training and Testing Plan for 100-step LoRA Fine-tuning

## Overview
This document outlines the steps to train the Qwen2.5-7B model with LoRA for 100 steps and test its performance.

## 1. Configuration Setup

Update the LoRA configuration file at `config/lora_qwen_config.json`:

```json
{
    "model_name": "mlx-community/Qwen2.5-7B-Instruct-bf16",
    "lora_config": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    },
    "training_config": {
        "num_epochs": 3,
        "max_steps": 100,        // Changed from 1 to 100
        "learning_rate": 1e-6,
        "warmup_steps": 500,
        "batch_size": 8,
        "gradient_accumulation_steps": 8,
        "weight_decay": 0.01,
        "max_grad_norm": 0.3,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 20         // Changed to evaluate more frequently
    },
    "data_config": {
        "train_file": "data/v1.1/mlx_format_qwen_v9/train.jsonl",
        "validation_file": "data/v1.1/mlx_format_qwen_v9/valid.jsonl",
        "max_seq_length": 2048,
        "preprocessing_num_workers": 4
    }
}
```

## 2. Clean Previous Checkpoints

Before starting the new training run, clean the previous checkpoints:

```bash
rm -rf lora_checkpoints/*
```

## 3. Training Process

Run the training script with the updated configuration:

```bash
source venv/bin/activate
python train_lora.py --config config/lora_qwen_config.json
```

Expected output will show:
- Training progress every 10 steps
- Validation loss every 20 steps
- Final checkpoint saved at step 100

## 4. Testing the Model

After training completes, test the model using `test_simple.py`:

```bash
source venv/bin/activate
python test_simple.py
```

The script will:
1. Load the base model
2. Apply LoRA layers
3. Load the checkpoint from `lora_checkpoints/checkpoint-100.npz`
4. Run inference on the test prompt

## 5. Test Prompts

Current test prompt in `test_simple.py`:
```python
test_prompts = [
    "<|im_start|>system\nYou are a home automation assistant that helps control smart home devices and provides information about the home environment.\n<|im_end|>\n<|im_start|>user\nCan you turn on the lights in the living room?\n<|im_end|>\n<|im_start|>assistant\n"
]
```

Additional test prompts you can uncomment in `test_simple.py`:
```python
test_prompts = [
    # Original prompt
    "<|im_start|>system\nYou are a home automation assistant that helps control smart home devices and provides information about the home environment.\n<|im_end|>\n<|im_start|>user\nCan you turn on the lights in the living room?\n<|im_end|>\n<|im_start|>assistant\n",
    
    # Temperature check prompt
    "<|im_start|>system\nYou are a home automation assistant that helps control smart home devices and provides information about the home environment.\n<|im_end|>\n<|im_start|>user\nWhat's the current temperature in the house?\n<|im_end|>\n<|im_start|>assistant\n",
    
    # Weather check prompt
    "<|im_start|>system\nYou are a home automation assistant that helps control smart home devices and provides information about the home environment.\n<|im_end|>\n<|im_start|>user\nCan you check the weather in Amsterdam?\n<|im_end|>\n<|im_start|>assistant\n"
]
```

## 6. Expected Timeline

1. Configuration update: 1 minute
2. Cleanup: 1 minute
3. Training (100 steps): ~30-45 minutes
4. Testing: 5-10 minutes
Total estimated time: 40-60 minutes

## 7. Success Criteria

1. Training completes successfully with checkpoint saved at step 100
2. Validation loss shows improvement over the training run
3. Model generates coherent and contextually appropriate responses
4. Response quality is better or comparable to the 1-step training

## 8. Monitoring and Logging

Key metrics to monitor during training:
- Training loss per step
- Validation loss every 20 steps
- Memory usage
- Training speed (tokens/second)
- Generation speed during testing

## 9. Troubleshooting

If issues arise:

1. Memory issues:
   - Reduce batch_size in config
   - Reduce max_seq_length

2. Training too slow:
   - Adjust logging_steps and eval_steps
   - Consider reducing validation batches

3. Poor generation quality:
   - Check training loss trends
   - Verify checkpoint loading
   - Try adjusting learning_rate

## 10. File Paths Reference

```
.
├── config/
│   └── lora_qwen_config.json    # Configuration file
├── lora_checkpoints/            # Checkpoint directory
│   └── checkpoint-100.npz       # Will be created during training
├── train_lora.py               # Training script
├── test_simple.py              # Testing script
└── data/
    └── v1.1/
        └── mlx_format_qwen_v9/
            ├── train.jsonl      # Training data
            └── valid.jsonl      # Validation data
``` 
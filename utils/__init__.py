# Copyright © 2023-2024 Apple Inc.

import glob
import json
import logging
from pathlib import Path
from typing import Generator

import mlx.core as mx
import mlx.nn as nn
import models
import transformers
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoConfig
from safetensors.numpy import load_file as safe_load


def fetch_from_hub(hf_path: str):
    model_path = snapshot_download(
        repo_id=hf_path,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
    )
    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        for k, v in safe_load(wf).items():
            # Convert bfloat16 to float32 before creating MLX array
            if v.dtype == 'bfloat16':
                v = v.astype('float32')
            weights[k] = v

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return weights, config.to_dict(), tokenizer


def upload_to_hub(path: str, name: str, hf_path: str):
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    repo_id = f"mlx-community/{name}"

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = f"""
# {name}
This model was converted to MLX format from [`{hf_path}`]().
Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
## Use with mlx
```bash
pip install mlx
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/llms/hf_llm
python generate.py --model {repo_id} --prompt "My name is"
```
"""
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gibibyte=5)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        mx.save_safetensors(
            str(save_dir / shard_name), shard, metadata={"format": "mlx"}
        )
        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(save_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def load(path_or_hf_repo: str, tokenizer_config={}):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        for k, v in safe_load(wf).items():
            # Convert bfloat16 to float32 before creating MLX array
            if v.dtype == 'bfloat16':
                v = v.astype('float32')
            weights[k] = v

    # Filter out bias terms from attention layers
    filtered_weights = {}
    for k, v in weights.items():
        if not (k.endswith('.bias') and any(x in k for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])):
            filtered_weights[k] = mx.array(v)

    model_args = models.ModelArgs.from_dict(config)
    model = models.Model(model_args)
    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in filtered_weights
        )
        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    # Handle tied embeddings
    if config.get("tie_word_embeddings", False):
        filtered_weights["lm_head.weight"] = filtered_weights["model.embed_tokens.weight"]

    model.load_weights(list(filtered_weights.items()))
    mx.eval(model.parameters())

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, **tokenizer_config
    )
    return model, tokenizer, config


def generate(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    top_k: int = None,
    top_p: float = None,
    sliding_window: int = None
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.
        top_k (int, optional): The number of highest probability tokens to keep for sampling.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling.
        sliding_window (int, optional): Size of the sliding window for attention. If None, use full attention.

    Yields:
        mx.array: The generated text.
    """

    def sample(logits: mx.array) -> mx.array:
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        
        # Apply temperature
        logits = logits / temp
        
        # Apply top-k sampling if specified
        if top_k is not None:
            # Get top k values and mask out the rest
            top_logits = mx.topk(logits, min(top_k, logits.shape[-1]))
            min_value = mx.min(top_logits, axis=-1, keepdims=True)
            logits = mx.where(logits < min_value, float('-inf'), logits)
        
        # Apply softmax to get probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Sample from the distribution
        return mx.random.categorical(probs)

    y = prompt
    cache = None
    while True:
        # If using sliding window, only keep the last window_size tokens
        if sliding_window is not None and y.shape[-1] > sliding_window:
            y = y[:, -sliding_window:]
            cache = None  # Reset cache when we truncate the input
            
        # Handle models that return cache and those that don't
        output = model(y[None], cache=cache)
        if isinstance(output, tuple):
            logits, cache = output
        else:
            logits = output
            cache = None
            
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y

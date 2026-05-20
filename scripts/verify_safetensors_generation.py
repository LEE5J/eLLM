#!/usr/bin/env python3
"""Load a local safetensors model with Transformers and run a short generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="models/Qwen3-Coder-30B-A3B-Instruct-full",
        help="Directory containing config/tokenizer files and safetensors shards.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function that keeps only even numbers from a list, then explain it briefly.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def print_safetensors_summary(model_dir: Path) -> None:
    shards = sorted(model_dir.glob("model-*.safetensors"))
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    total_size = index.get("metadata", {}).get("total_size")

    print("=== safetensors ===")
    print(f"model_dir: {model_dir}")
    print(f"shards: {len(shards)}")
    if total_size is not None:
        print(f"index_total_size_bytes: {total_size}")

    first_shard = shards[0]
    with safe_open(first_shard, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        first_key = keys[0]
        tensor = handle.get_tensor(first_key)
        print(f"first_shard: {first_shard.name}")
        print(f"first_shard_tensor_count: {len(keys)}")
        print(f"first_tensor: {first_key} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def apply_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(args.threads)

    print_safetensors_summary(model_dir)

    print("=== load ===")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    first_param = next(model.parameters())
    print(f"model_class: {model.__class__.__name__}")
    print(f"first_parameter: dtype={first_param.dtype} device={first_param.device}")

    prompt_text = apply_chat_template(tokenizer, args.prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    print("=== prompt ===")
    print(args.prompt)
    print(f"input_tokens: {inputs.input_ids.shape[-1]}")

    print("=== generated ===")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs.input_ids.shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(generated_text)
    print(f"generated_tokens: {generated_ids.shape[-1]}")


if __name__ == "__main__":
    main()

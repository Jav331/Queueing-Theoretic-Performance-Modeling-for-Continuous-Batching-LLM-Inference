from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from datasets.exceptions import DataFilesNotFoundError
from huggingface_hub import hf_hub_download


DEFAULT_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DEFAULT_DATA_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT = Path("data") / "sharegpt_trace.parquet"
DEFAULT_LENGTH_DIST = Path("data") / "length_dist.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical ShareGPT replay trace for sim/vLLM runs."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--data-file",
        default=DEFAULT_DATA_FILE,
        help="Optional file inside the HF dataset repo for JSON-style ShareGPT repos.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--sample-size", "-N", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-chars-per-turn",
        type=int,
        default=None,
        help="Skip pathological turns before tokenization. Default: 16 * max_model_len.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--length-dist", type=Path, default=DEFAULT_LENGTH_DIST)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing canonical trace.",
    )
    return parser.parse_args()


def load_examples(args: argparse.Namespace):
    try:
        return load_dataset(args.dataset, split=args.split)
    except DataFilesNotFoundError:
        data_path = hf_hub_download(
            repo_id=args.dataset,
            filename=args.data_file,
            repo_type="dataset",
        )
        with open(data_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "train", "examples"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError(f"Unsupported ShareGPT JSON structure in {args.data_file}")


def normalize_turn(turn: Any) -> tuple[str, str] | None:
    if not isinstance(turn, dict):
        return None
    role = turn.get("from") or turn.get("role")
    text = turn.get("value") or turn.get("content")
    if role is None or text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    role = str(role).lower()
    if role in {"human", "user"}:
        return "user", text
    if role in {"gpt", "assistant", "chatgpt"}:
        return "assistant", text
    return role, text


def extract_prompt_response(example: dict[str, Any]) -> tuple[str, str] | None:
    conversations = example.get("conversations") or example.get("conversation")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    turns = []
    for raw_turn in conversations:
        turn = normalize_turn(raw_turn)
        if turn is None:
            return None
        turns.append(turn)

    # Use the first user -> assistant pair. This mirrors common ShareGPT
    # inference benchmarks while keeping prompt/gen lengths unambiguous.
    for idx in range(len(turns) - 1):
        role, prompt = turns[idx]
        next_role, response = turns[idx + 1]
        if role == "user" and next_role == "assistant":
            return prompt, response
    return None


def token_count(tokenizer, text: str, max_tokens: int) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=True,
        max_length=max_tokens + 1,
        verbose=False,
    )
    return len(encoded["input_ids"])


def build_length_dist(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    prompt_lengths = [int(row["prompt_len"]) for row in rows]
    gen_lengths = [int(row["gen_len"]) for row in rows]
    return {
        "source_dataset": args.dataset,
        "source_data_file": args.data_file,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "sample_size": len(rows),
        "max_model_len": args.max_model_len,
        "prompt_len": prompt_lengths,
        "gen_len": gen_lengths,
    }


def main() -> None:
    args = parse_args()
    # Keep tokenizer-only preprocessing quiet on machines without torch.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    max_chars_per_turn = args.max_chars_per_turn or (16 * args.max_model_len)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists. Pass --overwrite to regenerate intentionally."
        )

    print(f"loading dataset: {args.dataset} [{args.split}]")
    dataset = load_examples(args)
    print(f"loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    valid_rows: list[dict[str, Any]] = []
    scanned = 0
    for example in dataset:
        scanned += 1
        pair = extract_prompt_response(example)
        if pair is None:
            continue
        prompt_text, response_text = pair
        if (
            len(prompt_text) > max_chars_per_turn
            or len(response_text) > max_chars_per_turn
        ):
            continue
        prompt_len = token_count(tokenizer, prompt_text, args.max_model_len)
        gen_len = token_count(tokenizer, response_text, args.max_model_len)
        if prompt_len <= 0 or gen_len <= 0:
            continue
        if prompt_len + gen_len > args.max_model_len:
            continue
        valid_rows.append(
            {
                "prompt_text": prompt_text,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
            }
        )

    if len(valid_rows) < args.sample_size:
        raise ValueError(
            f"Only {len(valid_rows)} valid conversations after filtering; requested {args.sample_size}."
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(valid_rows, args.sample_size)
    rows = [
        {
            "request_id": request_id,
            "prompt_text": row["prompt_text"],
            "prompt_len": row["prompt_len"],
            "gen_len": row["gen_len"],
        }
        for request_id, row in enumerate(sampled)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.length_dist.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    args.length_dist.write_text(
        json.dumps(build_length_dist(rows, args), indent=2),
        encoding="utf-8",
    )

    print(f"scanned: {scanned}")
    print(f"valid after filtering: {len(valid_rows)}")
    print(f"sampled: {len(rows)}")
    print(f"wrote {args.output}")
    print(f"wrote {args.length_dist}")


if __name__ == "__main__":
    main()

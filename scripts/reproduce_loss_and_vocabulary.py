#!/usr/bin/env python3
"""Reproduce the full-vocabulary floor analysis and sampled Pile loss.

The two analyses are independent:

* ``vocabulary`` evaluates the 300-prompt IOI set at Pythia-160M step 2000.
* ``loss`` reconstructs the fixed sampling procedure from the
  ``monology/pile-uncopyrighted`` training stream. Because the source is an
  external streaming dataset, exact reproduction also depends on its record
  ordering remaining unchanged.
"""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

import numpy as np

from lib.ioi_dataset import build_prompt_records
from lib.runtime import final_logits, load_hooked_model

MODEL_ID = "EleutherAI/pythia-160m-deduped"
LOSS_STEPS = (1000, 2000, 3000, 5000, 10000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=("vocabulary", "loss", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vocabulary-output", type=Path, default=Path("results/full_vocabulary_floor.json"))
    parser.add_argument("--loss-output", type=Path, default=Path("results/pile_loss_sample.json"))
    return parser.parse_args()


def rank(logits: np.ndarray, token_id: int) -> int:
    return int(np.sum(logits > logits[token_id]) + 1)


def reproduce_vocabulary(device: str | None) -> dict:
    model = load_hooked_model(MODEL_ID, 2000, device)
    records = build_prompt_records(model.tokenizer, prompts_per_template=30, seed=42, control_seed=43)
    logits = final_logits(model, [record.prompt for record in records])
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    neither = 0
    repeated_probability = []
    correct_probability = []
    repeated_rank = []
    correct_rank = []
    for row, record in enumerate(records):
        io_id = int(model.to_single_token(" " + record.io_name))
        s_id = int(model.to_single_token(" " + record.s_name))
        top = int(np.argmax(logits[row]))
        neither += int(top not in (io_id, s_id))
        repeated_probability.append(float(probabilities[row, s_id]))
        correct_probability.append(float(probabilities[row, io_id]))
        repeated_rank.append(rank(logits[row], s_id))
        correct_rank.append(rank(logits[row], io_id))

    return {
        "description": "Full-vocabulary behavior at the Pythia-160M floor.",
        "model": MODEL_ID,
        "step": 2000,
        "n_prompts": len(records),
        "greedy_selects_neither_fraction": neither / len(records),
        "prob_repeated_name": float(np.mean(repeated_probability)),
        "prob_correct_name": float(np.mean(correct_probability)),
        "mean_rank_repeated_name": float(np.mean(repeated_rank)),
        "mean_rank_correct_name": float(np.mean(correct_rank)),
        "rank_definition": "1 + number of vocabulary logits strictly greater",
    }


def sample_pile_sequences(tokenizer) -> list[list[int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The sampled-loss reproduction requires the `datasets` package.") from exc

    stream = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )
    candidates = list(islice(stream, 100_000, 100_100))
    sequences = []
    for record in candidates:
        text = record.get("text")
        if not isinstance(text, str):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= 512:
            sequences.append([int(token) for token in ids[:512]])
        if len(sequences) == 50:
            break
    if len(sequences) < 50:
        raise RuntimeError(
            f"Only {len(sequences)} of the 100 candidate texts had at least 512 tokens"
        )
    return sequences


def load_hf_model(step: int, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=f"step{step}",
        torch_dtype=torch.float32,
    ).to(resolved)
    model.eval()
    return model, resolved


def mean_sequence_loss(model, device: str, sequences: list[list[int]]) -> float:
    import torch

    losses = []
    with torch.inference_mode():
        for ids in sequences:
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            output = model(input_ids=tokens, labels=tokens)
            losses.append(float(output.loss))
    return float(np.mean(losses))


def reproduce_loss(device: str | None) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sequences = sample_pile_sequences(tokenizer)
    by_step = {}
    for step in LOSS_STEPS:
        model, resolved = load_hf_model(step, device)
        by_step[f"step_{step}"] = {
            "released_pythia_160m_loss": mean_sequence_loss(model, resolved, sequences)
        }
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "description": (
            "Language-modeling loss on a fixed sample reconstructed from the "
            "monology/pile-uncopyrighted training stream."
        ),
        "dataset": {
            "id": "monology/pile-uncopyrighted",
            "split": "train",
            "streaming": True,
        },
        "procedure": {
            "skip_records": 100_000,
            "candidate_texts_read": 100,
            "minimum_tokens": 512,
            "tokens_per_sequence": 512,
            "maximum_sequences": 50,
            "aggregation": "mean causal language-model loss across sequences",
        },
        "external_dependency_note": (
            "Exact numerical reproduction assumes the streaming dataset retains "
            "the record ordering used for the released result."
        ),
        "by_step": by_step,
        "paper_series": "released_pythia_160m_loss",
    }


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.analysis in ("vocabulary", "all"):
        write(args.vocabulary_output, reproduce_vocabulary(args.device))
    if args.analysis in ("loss", "all"):
        write(args.loss_output, reproduce_loss(args.device))


if __name__ == "__main__":
    main()

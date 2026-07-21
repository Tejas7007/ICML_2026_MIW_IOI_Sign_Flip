#!/usr/bin/env python3
"""Reproduce the 144-head zero-ablation sweep used for the step-1000 claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

from lib.ioi_dataset import build_prompt_records

MODEL_ID = "EleutherAI/pythia-160m-deduped"
DEFAULT_STEP = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero every Pythia-160M attention head independently."
    )
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/head_zero_ablation_step1000.json"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_model(step: int, device: str) -> HookedTransformer:
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=f"step{step}",
    )
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        hf_model=hf_model,
        device=device,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
    )
    del hf_model
    model.eval()
    return model


def build_examples(
    model: HookedTransformer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records = [
        record
        for record in build_prompt_records(
            model.tokenizer,
            prompts_per_template=20,
            seed=42,
            control_seed=43,
        )
        if record.template_id < 8
    ]
    if len(records) != 160:
        raise RuntimeError(f"Expected 160 prompts, produced {len(records)}")

    tokens = model.to_tokens([record.prompt for record in records]).to(model.cfg.device)
    io_ids = torch.tensor(
        [model.to_single_token(" " + record.io_name) for record in records],
        device=model.cfg.device,
    )
    s_ids = torch.tensor(
        [model.to_single_token(" " + record.s_name) for record in records],
        device=model.cfg.device,
    )
    return tokens, io_ids, s_ids


def logit_differences(
    logits: torch.Tensor,
    io_ids: torch.Tensor,
    s_ids: torch.Tensor,
) -> torch.Tensor:
    final = logits[:, -1, :]
    row = torch.arange(final.shape[0], device=final.device)
    return final[row, io_ids] - final[row, s_ids]


def main() -> int:
    args = parse_args()
    model = load_model(args.step, args.device)
    tokens, io_ids, s_ids = build_examples(model)

    with torch.inference_mode():
        baseline_ld = logit_differences(model(tokens), io_ids, s_ids)
    baseline_mean = float(baseline_ld.mean().item())
    baseline_accuracy = float((baseline_ld > 0).float().mean().item())

    heads: dict[str, float] = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            def zero_head(
                value: torch.Tensor,
                _hook: Any,
                head_index: int = head,
            ) -> torch.Tensor:
                changed = value.clone()
                changed[:, :, head_index, :] = 0.0
                return changed

            with torch.inference_mode():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", zero_head)],
                )
            ablated_mean = float(
                logit_differences(logits, io_ids, s_ids).mean().item()
            )
            heads[f"L{layer}H{head}"] = ablated_mean - baseline_mean

    largest_positive = max(heads.items(), key=lambda item: item[1])
    largest_negative = min(heads.items(), key=lambda item: item[1])
    largest_absolute = max(heads.items(), key=lambda item: abs(item[1]))

    result = {
        "model": MODEL_ID,
        "checkpoint": args.step,
        "intervention": "zero one attention-head output at a time",
        "metric": (
            "change in mean IO-minus-S logit difference "
            "relative to the clean baseline"
        ),
        "protocol": {
            "template_selection": "ALL_TEMPLATES[:8]",
            "template_order": "the first eight BABA-style template families",
            "template_families": 8,
            "prompts_per_template": 20,
            "symmetric_name_role_swaps": True,
            "seed": 42,
            "n_examples": int(tokens.shape[0]),
            "n_layers": int(model.cfg.n_layers),
            "n_heads_per_layer": int(model.cfg.n_heads),
            "n_heads_tested": len(heads),
        },
        "baseline": {
            "accuracy": baseline_accuracy,
            "mean_logit_difference": baseline_mean,
        },
        "heads": heads,
        "extrema": {
            "largest_positive_change": {
                "head": largest_positive[0],
                "delta_logit_difference": largest_positive[1],
            },
            "largest_negative_change": {
                "head": largest_negative[0],
                "delta_logit_difference": largest_negative[1],
            },
            "largest_absolute_change": {
                "head": largest_absolute[0],
                "absolute_delta_logit_difference": abs(largest_absolute[1]),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["extrema"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

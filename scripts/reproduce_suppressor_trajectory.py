#!/usr/bin/env python3
"""Reproduce the mature-selected suppressor trajectories and characterization.

For each Pythia model, every attention head is mean-ablated at maturity. The head
whose ablation most reduces IO-minus-S logit difference is selected once, then
that fixed head is mean-ablated at earlier checkpoints without reselection.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from lib.ioi_dataset import build_prompt_records
from lib.runtime import bootstrap_mean, load_hooked_model, locate_name_occurrence

PYTHIA = {
    "pythia-160m": {
        "model_id": "EleutherAI/pythia-160m-deduped",
        "steps": [1000, 2000, 3000, 5000, 8000, 13000, 143000],
    },
    "pythia-410m": {
        "model_id": "EleutherAI/pythia-410m-deduped",
        "steps": [1000, 2000, 3000, 5000, 8000, 13000, 143000],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/suppressor_ablation_trajectory.json"))
    parser.add_argument(
        "--characterization-output",
        type=Path,
        default=Path("results/suppressor_characterization.json"),
    )
    return parser.parse_args()


def template_batches(model, prompts_per_template: int = 30):
    records = build_prompt_records(
        model.tokenizer,
        prompts_per_template=prompts_per_template,
        seed=42,
        control_seed=43,
    )
    batches = []
    for template_id in range(10):
        current = [record for record in records if record.template_id == template_id]
        tokens = model.to_tokens([record.prompt for record in current], prepend_bos=True)
        io_ids = np.asarray([model.to_single_token(" " + record.io_name) for record in current])
        s_ids = np.asarray([model.to_single_token(" " + record.s_name) for record in current])
        batches.append((current, tokens, io_ids, s_ids))
    return batches


def logit_differences(model, batches, head=None, mode="mean") -> np.ndarray:
    import torch

    values = []
    for _, tokens, io_ids_np, s_ids_np in batches:
        tokens = tokens.to(model.cfg.device)
        io_ids = torch.tensor(io_ids_np, device=model.cfg.device)
        s_ids = torch.tensor(s_ids_np, device=model.cfg.device)
        rows = torch.arange(tokens.shape[0], device=model.cfg.device)
        hooks = []
        if head is not None:
            layer, index = head

            def ablate(z, hook, index=index, mode=mode):
                out = z.clone()
                if mode == "mean":
                    replacement = z[:, :, index].mean(dim=(0, 1), keepdim=True)
                    out[:, :, index] = replacement
                elif mode == "zero":
                    out[:, :, index] = 0
                else:
                    raise ValueError(mode)
                return out

            hooks = [(f"blocks.{layer}.attn.hook_z", ablate)]
        with torch.inference_mode():
            logits = (
                model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
                if hooks else model(tokens, return_type="logits")
            )[:, -1]
        ld = logits[rows, io_ids] - logits[rows, s_ids]
        values.extend(ld.float().cpu().tolist())
    return np.asarray(values)


def select_mature_head(model, batches):
    base = logit_differences(model, batches)
    base_mean = float(base.mean())
    scores = {}
    best = None
    best_effect = 0.0
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            ablated = logit_differences(model, batches, (layer, head), "mean")
            effect = float(ablated.mean() - base_mean)
            scores[f"L{layer}H{head}"] = effect
            if effect < best_effect:
                best = (layer, head)
                best_effect = effect
    if best is None:
        raise RuntimeError("No mature suppressor head found")
    return best, scores


def characterize(model, batches, head):
    import torch

    layer, head_index = head
    model.set_use_attn_result(True)
    attention_values = []
    projection_values = []
    for records, tokens, io_ids_np, s_ids_np in batches:
        tokens = tokens.to(model.cfg.device)
        s2 = torch.tensor([
            locate_name_occurrence(model.tokenizer, record.prompt, record.s_name, 2)
            for record in records
        ], device=model.cfg.device)
        keep = {
            f"blocks.{layer}.attn.hook_pattern",
            f"blocks.{layer}.attn.hook_result",
        }
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in keep)
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head_index, -1]
        rows = torch.arange(tokens.shape[0], device=model.cfg.device)
        attention_values.extend(pattern[rows, s2].float().cpu().tolist())

        result = cache[f"blocks.{layer}.attn.hook_result"][:, -1, head_index]
        io_ids = torch.tensor(io_ids_np, device=model.cfg.device)
        s_ids = torch.tensor(s_ids_np, device=model.cfg.device)
        direction = model.W_U[:, io_ids].T - model.W_U[:, s_ids].T
        projection = (result * direction).sum(dim=-1)
        projection_values.extend(projection.float().cpu().tolist())
    return {
        "attention_to_S2": float(np.mean(attention_values)),
        "IO_minus_S_projection": float(np.mean(projection_values)),
    }


def main() -> None:
    args = parse_args()
    trajectories = {}
    characterization = {}

    for key, config in PYTHIA.items():
        mature = load_hooked_model(config["model_id"], 143000, args.device)
        mature_batches = template_batches(mature, 30)
        head, scores = select_mature_head(mature, mature_batches)
        trajectories[key] = {
            "model_id": config["model_id"],
            "suppression_head": list(head),
            "selection": "most negative mature mean-ablation effect",
            "mature_head_scores": scores,
            "by_step": {},
        }
        del mature
        gc.collect()

        for step in config["steps"]:
            model = load_hooked_model(config["model_id"], step, args.device)
            batches = template_batches(model, 30)
            base = logit_differences(model, batches)
            ablated = logit_differences(model, batches, head, "mean")
            delta = ablated - base
            trajectories[key]["by_step"][f"step_{step}"] = {
                "base_ld": float(base.mean()),
                "ablation_delta": float(delta.mean()),
                "ablation_ci": bootstrap_mean(delta, draws=10_000, seed=42),
                "ioi_acc": float((base > 0).mean()),
            }
            if key == "pythia-160m" and step in (1000, 3000, 143000):
                characterization.setdefault("pythia-160m", {
                    "head": f"L{head[0]}H{head[1]}",
                    "protocol": {
                        "n_examples": 200,
                        "template_families": 10,
                        "prompts_per_family": 20,
                    },
                    "by_step": {},
                })
                characterization_batches = template_batches(model, 20)
                characterization["pythia-160m"]["by_step"][f"step_{step}"] = characterize(
                    model, characterization_batches, head
                )
            del model
            gc.collect()

    from transformer_lens import HookedTransformer
    import torch

    stanford_id = "stanford-crfm/alias-gpt2-small-x21"
    stanford_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stanford = HookedTransformer.from_pretrained(
        stanford_id,
        revision="checkpoint-100000",
        device=stanford_device,
    )
    stanford.eval()
    stanford_batches = template_batches(stanford, 20)
    stanford_head = (10, 10)
    base = logit_differences(stanford, stanford_batches)
    zeroed = logit_differences(stanford, stanford_batches, stanford_head, "zero")
    stats = characterize(stanford, stanford_batches, stanford_head)
    characterization["stanford-gpt2-small"] = {
        "model_id": stanford_id,
        "step": 100000,
        "head": "L10H10",
        "zero_ablation_accuracy_change": float((zeroed > 0).mean() - (base > 0).mean()),
        **stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trajectories, indent=2) + "\n")
    args.characterization_output.parent.mkdir(parents=True, exist_ok=True)
    args.characterization_output.write_text(json.dumps(characterization, indent=2) + "\n")


if __name__ == "__main__":
    main()

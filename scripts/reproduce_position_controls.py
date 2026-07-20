#!/usr/bin/env python3
"""Reproduce the Pythia-160M position-control battery.

The random arms replace the residual-stream activation at one position with an
independent Gaussian direction normalized to the clean activation norm for the
same prompt, layer, and position. A new vector is drawn for every prompt and
layer. All arms use layers 3--5 at steps 2000 and 143000.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from lib.ioi_dataset import build_prompt_records
from lib.runtime import bootstrap_mean, load_hooked_model, locate_name_occurrence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--steps", default="2000,143000")
    parser.add_argument("--layers", default="3,4,5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/position_control_battery.json"))
    return parser.parse_args()


def run_stage(model, step: int, layers: list[int], batch_size: int) -> dict:
    import torch

    records = build_prompt_records(model.tokenizer, prompts_per_template=30, seed=42, control_seed=43)
    rows = [model.to_tokens(r.prompt, prepend_bos=True)[0] for r in records]
    controls = [model.to_tokens(r.control_prompt, prepend_bos=True)[0] for r in records]
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row.numel())].append(i)

    arm_names = ("real_S2", "random_S2", "random_S1", "random_IO", "random_struct")
    base_values: list[float] = []
    changed: dict[str, list[float]] = {name: [] for name in arm_names}
    generator = torch.Generator(device=model.cfg.device)
    generator.manual_seed(42 + step)

    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            clean_tokens = torch.stack([rows[i] for i in chunk]).to(model.cfg.device)
            control_tokens = torch.stack([controls[i] for i in chunk]).to(model.cfg.device)
            batch_records = [records[i] for i in chunk]

            positions = {"S2": [], "S1": [], "IO": [], "struct": []}
            for record in batch_records:
                s1 = locate_name_occurrence(model.tokenizer, record.prompt, record.s_name, 1)
                s2 = locate_name_occurrence(model.tokenizer, record.prompt, record.s_name, 2)
                io = locate_name_occurrence(model.tokenizer, record.prompt, record.io_name, 1)
                positions["S1"].append(s1)
                positions["S2"].append(s2)
                positions["IO"].append(io)
                positions["struct"].append(s1 + 1)

            with torch.inference_mode():
                base_logits, clean_cache = model.run_with_cache(
                    clean_tokens, names_filter=lambda name: name in hook_names
                )
                _, control_cache = model.run_with_cache(
                    control_tokens, names_filter=lambda name: name in hook_names
                )
            base_final = base_logits[:, -1]
            io_ids = torch.tensor(
                [model.to_single_token(" " + r.io_name) for r in batch_records],
                device=model.cfg.device,
            )
            s_ids = torch.tensor(
                [model.to_single_token(" " + r.s_name) for r in batch_records],
                device=model.cfg.device,
            )
            ridx = torch.arange(len(chunk), device=model.cfg.device)
            base_ld = base_final[ridx, io_ids] - base_final[ridx, s_ids]
            base_values.extend(base_ld.float().cpu().tolist())

            random_values: dict[tuple[str, int], torch.Tensor] = {}
            for arm, key in (
                ("random_S2", "S2"),
                ("random_S1", "S1"),
                ("random_IO", "IO"),
                ("random_struct", "struct"),
            ):
                p = torch.tensor(positions[key], device=model.cfg.device)
                for layer in layers:
                    clean = clean_cache[f"blocks.{layer}.hook_resid_post"][ridx, p]
                    noise = torch.randn(
                        clean.shape, generator=generator,
                        device=clean.device, dtype=clean.dtype,
                    )
                    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    random_values[(arm, layer)] = noise * clean.norm(dim=-1, keepdim=True)

            def hooks_for(arm: str):
                hooks = []
                for layer in layers:
                    name = f"blocks.{layer}.hook_resid_post"
                    if arm == "real_S2":
                        donor = control_cache[name]
                        p = torch.tensor(positions["S2"], device=model.cfg.device)

                        def patch(value, hook, donor=donor, p=p):
                            out = value.clone()
                            out[ridx, p] = donor[ridx, p]
                            return out
                    else:
                        key = {
                            "random_S2": "S2",
                            "random_S1": "S1",
                            "random_IO": "IO",
                            "random_struct": "struct",
                        }[arm]
                        p = torch.tensor(positions[key], device=model.cfg.device)
                        replacement = random_values[(arm, layer)]

                        def patch(value, hook, p=p, replacement=replacement):
                            out = value.clone()
                            out[ridx, p] = replacement
                            return out
                    hooks.append((name, patch))
                return hooks

            for arm in arm_names:
                with torch.inference_mode():
                    logits = model.run_with_hooks(
                        clean_tokens, fwd_hooks=hooks_for(arm), return_type="logits"
                    )[:, -1]
                ld = logits[ridx, io_ids] - logits[ridx, s_ids]
                changed[arm].extend(ld.float().cpu().tolist())

            del clean_cache, control_cache

    base = np.asarray(base_values)
    arms = {}
    for arm in arm_names:
        delta = np.asarray(changed[arm]) - base
        arms[arm] = {
            "delta_ld_mean": float(delta.mean()),
            "delta_ci": bootstrap_mean(delta, draws=10_000, seed=42),
        }
    return {
        "step": step,
        "n": len(base),
        "ioi_acc": float((base > 0).mean()),
        "base_ld_mean": float(base.mean()),
        "arms": arms,
    }


def main() -> None:
    args = parse_args()
    steps = [int(x) for x in args.steps.split(",")]
    layers = [int(x) for x in args.layers.split(",")]
    output = {
        "protocol": {
            "model": args.model_id,
            "layers": layers,
            "n_prompts": 300,
            "random_replacement": (
                "independent per-prompt, per-layer Gaussian direction normalized "
                "to the clean activation norm"
            ),
        }
    }
    for step in steps:
        model = load_hooked_model(args.model_id, step, args.device)
        label = "dip" if step == 2000 else "mature" if step == 143000 else f"step_{step}"
        output[label] = run_stage(model, step, layers, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

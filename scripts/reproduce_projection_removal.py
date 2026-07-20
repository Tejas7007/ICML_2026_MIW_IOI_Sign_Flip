#!/usr/bin/env python3
"""Reproduce removal of the linear duplication-probe direction.

The probe is trained in-distribution on layer-5 S2 activations from the
300-prompt Pythia-160M step-2000 evaluation. The normalized probe weight is
removed from the S2 residual-stream activation as

    h' = h - strength * (h dot d) * d.

The script also evaluates an orthogonal direction, a shuffled-label probe
direction, and five random unit directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from lib.ioi_dataset import build_prompt_records
from lib.runtime import load_hooked_model, locate_name_occurrence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/projection_removal.json"))
    return parser.parse_args()


def main() -> None:
    import torch
    from sklearn.linear_model import LogisticRegression

    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = load_hooked_model(args.model_id, args.step, args.device)
    records = build_prompt_records(model.tokenizer, prompts_per_template=30, seed=42, control_seed=43)

    template_data = []
    activations = []
    labels = []
    hook_name = f"blocks.{args.layer}.hook_resid_post"

    for template_id in range(10):
        current = [record for record in records if record.template_id == template_id]
        clean = model.to_tokens([record.prompt for record in current], prepend_bos=True).to(model.cfg.device)
        control = model.to_tokens([record.control_prompt for record in current], prepend_bos=True).to(model.cfg.device)
        s2 = [
            locate_name_occurrence(model.tokenizer, record.prompt, record.s_name, 2)
            for record in current
        ]
        io_ids = torch.tensor(
            [model.to_single_token(" " + record.io_name) for record in current],
            device=model.cfg.device,
        )
        s_ids = torch.tensor(
            [model.to_single_token(" " + record.s_name) for record in current],
            device=model.cfg.device,
        )
        with torch.inference_mode():
            _, clean_cache = model.run_with_cache(
                clean, names_filter=lambda name: name == hook_name
            )
            _, control_cache = model.run_with_cache(
                control, names_filter=lambda name: name == hook_name
            )
        rows = torch.arange(len(current), device=model.cfg.device)
        positions = torch.tensor(s2, device=model.cfg.device)
        activations.append(clean_cache[hook_name][rows, positions].float().cpu().numpy())
        labels.extend([1] * len(current))
        activations.append(control_cache[hook_name][rows, positions].float().cpu().numpy())
        labels.extend([0] * len(current))
        template_data.append((clean, io_ids, s_ids, s2))

    x = np.concatenate(activations)
    y = np.asarray(labels)
    classifier = LogisticRegression(max_iter=2_000, random_state=42, C=1.0)
    classifier.fit(x, y)

    direction = torch.tensor(
        classifier.coef_[0], dtype=torch.float32, device=model.cfg.device
    )
    direction = direction / direction.norm()

    shuffled_y = y.copy()
    np.random.seed(43)
    np.random.shuffle(shuffled_y)
    shuffled = LogisticRegression(max_iter=2_000, random_state=42, C=1.0)
    shuffled.fit(x, shuffled_y)
    shuffled_direction = torch.tensor(
        shuffled.coef_[0], dtype=torch.float32, device=model.cfg.device
    )
    shuffled_direction = shuffled_direction / shuffled_direction.norm()

    random_directions = []
    for _ in range(5):
        candidate = torch.randn(model.cfg.d_model, device=model.cfg.device)
        random_directions.append(candidate / candidate.norm())
    orthogonal = torch.randn(model.cfg.d_model, device=model.cfg.device)
    orthogonal = orthogonal - torch.dot(orthogonal, direction) * direction
    orthogonal = orthogonal / orthogonal.norm()

    conditions = {
        "remove_dup_direction": direction,
        "remove_ortho_direction": orthogonal,
        "remove_shuffled_direction": shuffled_direction,
        **{
            f"remove_random_{index}": value
            for index, value in enumerate(random_directions)
        },
    }
    strengths = (0.5, 1.0, 2.0, 4.0)
    results = {
        "metadata": {
            "model": args.model_id,
            "step": args.step,
            "layer": args.layer,
            "n_prompts": 300,
            "probe_training_accuracy": float(classifier.score(x, y)),
            "equation": "h' = h - strength * (h dot d) * d",
            "values_are_point_estimates": True,
        }
    }

    def evaluate(direction_tensor=None, strength=1.0):
        values = []
        for tokens, io_ids, s_ids, s2 in template_data:
            rows = torch.arange(tokens.shape[0], device=model.cfg.device)
            if direction_tensor is None:
                with torch.inference_mode():
                    logits = model(tokens, return_type="logits")[:, -1]
            else:
                positions = torch.tensor(s2, device=model.cfg.device)

                def remove_projection(value, hook):
                    out = value.clone()
                    selected = out[rows, positions]
                    projection = (
                        selected @ direction_tensor
                    ).unsqueeze(-1) * direction_tensor
                    out[rows, positions] = selected - strength * projection
                    return out

                with torch.inference_mode():
                    logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, remove_projection)],
                        return_type="logits",
                    )[:, -1]
            ld = logits[rows, io_ids] - logits[rows, s_ids]
            values.extend(ld.float().cpu().tolist())
        arr = np.asarray(values)
        return {"ld": round(float(arr.mean()), 4), "acc": round(float((arr > 0).mean()), 4)}

    results["baseline"] = evaluate()
    for name, value in conditions.items():
        for strength in strengths:
            results[f"{name}_{strength:.1f}x"] = evaluate(value, strength)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

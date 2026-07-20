#!/usr/bin/env python3
"""Reproduce the split-safe frozen-suppressor-set trajectory.

This is the narrow held-out robustness check reported in Appendix D. It does
not reproduce the broader mechanism analyses from the separate EMNLP project.

Design
------
* Model: independently trained ``teys7007/pythia-160m-seed42-dense``.
* The mature suppressor set is fixed before this trajectory is evaluated.
* The holdout excludes all names and template families used by the discovery
  partitions.
* Each of 192 held-out items supplies two matched contrasts, XX<-YX and
  YY<-XY. The token at S2 is identical in target and donor.
* For each selected head, its complete ``hook_z`` output at END is copied from
  the matched donor run into the target run.
* The fair three-way margin is

      logit(IO) - max(logit(repeated S), logit(donor alternate name)).

The two contrasts for each item are averaged before the bootstrap, so the
confidence interval resamples 192 independent item-level means rather than
pretending that the 384 contrasts are independent.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from lib.locked_ioi_benchmark import (
    IOIBenchmark,
    NAME_POOL,
    OBJECTS,
    PLACES,
    TEMPLATE_FAMILIES,
)
from lib.runtime import load_hooked_model


MODEL_ID = "teys7007/pythia-160m-seed42-dense"
DEFAULT_STEPS = (1800, 2000, 3200, 5000, 6000)
HOLDOUT_SEED = 20260725
DISCOVERY_PARTITIONS = (
    (20260703, "selection"),
    (20260703, "validation"),
    (20260704, "validation"),
)

MATURE_SUPPRESSOR_SET = (
    (6, 2), (6, 7), (7, 3), (7, 7), (8, 0),
    (8, 5), (9, 4), (9, 11), (10, 0), (10, 2),
    (10, 6), (10, 7), (10, 9), (11, 1), (11, 4),
)


@dataclass(frozen=True)
class FactorialItem:
    item_id: str
    order: str
    template_id: int
    io_name: str
    x_name: str
    y_name: str
    prompts: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_STEPS),
        help="Comma-separated checkpoint steps.",
    )
    parser.add_argument("--n-items", type=int, default=192)
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/splitsafe_suppressor_set.json"),
    )
    return parser.parse_args()


def single_token_id(tokenizer, name: str) -> int:
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{name!r} is not a single leading-space token: {ids}")
    return int(ids[0])


def build_strict_holdout(n_items: int, seed: int) -> tuple[list[FactorialItem], dict]:
    used_names: set[str] = set()
    used_templates: set[int] = set()
    excluded = []
    for benchmark_seed, split_name in DISCOVERY_PARTITIONS:
        split = IOIBenchmark(seed=benchmark_seed).split(split_name)
        used_names.update(split.names)
        used_templates.update(split.template_indices)
        excluded.append({
            "benchmark_seed": benchmark_seed,
            "split": split_name,
            "names": sorted(split.names),
            "template_indices": sorted(split.template_indices),
        })

    available_names = sorted(set(NAME_POOL) - used_names)
    available_templates = sorted(
        set(range(len(TEMPLATE_FAMILIES))) - used_templates
    )
    if len(available_names) < 3 or not available_templates:
        raise RuntimeError("The strict holdout has too few names or templates")

    rng = random.Random(seed)
    items: list[FactorialItem] = []
    for index in range(n_items):
        io_name, x_name, y_name = rng.sample(available_names, 3)
        template_id = available_templates[index % len(available_templates)]
        family = TEMPLATE_FAMILIES[template_id]
        order = "ABBA" if index % 2 == 0 else "BABA"
        template = family.abba if order == "ABBA" else family.baba
        shared = {
            "IO": io_name,
            "PLACE": rng.choice(PLACES),
            "OBJECT": rng.choice(OBJECTS),
            "ARG_ADJ": "long",
        }
        prompts = {
            "XX": template.format(S1=x_name, S2=x_name, **shared),
            "YX": template.format(S1=y_name, S2=x_name, **shared),
            "XY": template.format(S1=x_name, S2=y_name, **shared),
            "YY": template.format(S1=y_name, S2=y_name, **shared),
        }
        items.append(FactorialItem(
            item_id=f"joint{index:04d}",
            order=order,
            template_id=template_id,
            io_name=io_name,
            x_name=x_name,
            y_name=y_name,
            prompts=prompts,
        ))

    manifest = {
        "holdout_seed": seed,
        "n_requested_items": n_items,
        "excluded_partitions": excluded,
        "available_names": available_names,
        "available_template_indices": available_templates,
        "n_available_names": len(available_names),
        "n_available_templates": len(available_templates),
    }
    return items, manifest


def bootstrap_item_means(
    item_values: Iterable[float],
    *,
    draws: int,
    seed: int,
) -> list[float]:
    values = np.asarray(list(item_values), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def run_step(model, items: list[FactorialItem], step: int, draws: int) -> dict:
    import torch

    hook_names = {
        layer: f"blocks.{layer}.attn.hook_z"
        for layer, _ in MATURE_SUPPRESSOR_SET
    }
    wanted = set(hook_names.values())
    by_item: dict[str, list[float]] = {}
    raw = []

    for item in items:
        io_token = single_token_id(model.tokenizer, item.io_name)
        x_token = single_token_id(model.tokenizer, item.x_name)
        y_token = single_token_id(model.tokenizer, item.y_name)

        contrasts = (
            ("XX<-YX", "XX", "YX", x_token, y_token),
            ("YY<-XY", "YY", "XY", y_token, x_token),
        )
        for contrast, target_name, donor_name, repeated_token, alternate_token in contrasts:
            target = model.to_tokens(
                item.prompts[target_name], prepend_bos=True
            ).to(model.cfg.device)
            donor = model.to_tokens(
                item.prompts[donor_name], prepend_bos=True
            ).to(model.cfg.device)
            if target.shape != donor.shape:
                raise RuntimeError(
                    f"{item.item_id}/{contrast}: target and donor lengths differ"
                )

            with torch.inference_mode():
                base_logits = model(target, return_type="logits")[0, -1]
                _, donor_cache = model.run_with_cache(
                    donor,
                    names_filter=lambda name: name in wanted,
                )

            end_position = target.shape[1] - 1
            hooks = []
            for layer, hook_name in hook_names.items():
                donor_z = donor_cache[hook_name][0, end_position].detach()
                selected_heads = tuple(
                    head for selected_layer, head in MATURE_SUPPRESSOR_SET
                    if selected_layer == layer
                )

                def patch_z(
                    activation,
                    hook,
                    *,
                    source=donor_z,
                    heads=selected_heads,
                    position=end_position,
                ):
                    changed = activation.clone()
                    for head in heads:
                        changed[0, position, head] = source[head].to(changed.dtype)
                    return changed

                hooks.append((hook_name, patch_z))

            with torch.inference_mode():
                patched_logits = model.run_with_hooks(
                    target,
                    fwd_hooks=hooks,
                    return_type="logits",
                )[0, -1]

            base_margin = float(
                base_logits[io_token]
                - torch.maximum(
                    base_logits[repeated_token],
                    base_logits[alternate_token],
                )
            )
            patched_margin = float(
                patched_logits[io_token]
                - torch.maximum(
                    patched_logits[repeated_token],
                    patched_logits[alternate_token],
                )
            )
            delta = patched_margin - base_margin
            by_item.setdefault(item.item_id, []).append(delta)
            raw.append({
                "item_id": item.item_id,
                "order": item.order,
                "template_id": item.template_id,
                "contrast": contrast,
                "base_margin3": base_margin,
                "patched_margin3": patched_margin,
                "d_margin3": delta,
            })
            del donor_cache

    if any(len(values) != 2 for values in by_item.values()):
        raise RuntimeError("Every held-out item must contribute exactly two contrasts")
    item_means = {
        item_id: float(np.mean(values))
        for item_id, values in by_item.items()
    }

    bootstrap_seed = step + 10_577
    return {
        "step": step,
        "n_items": len(item_means),
        "n_observations": len(raw),
        "d_margin3_mean": float(np.mean(list(item_means.values()))),
        "d_margin3_ci": bootstrap_item_means(
            item_means.values(),
            draws=draws,
            seed=bootstrap_seed,
        ),
        "bootstrap": {
            "unit": "held-out factorial item",
            "draws": draws,
            "seed": bootstrap_seed,
            "two_contrasts_averaged_before_resampling": True,
        },
        "raw": raw,
    }


def main() -> None:
    args = parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    items, holdout = build_strict_holdout(args.n_items, args.seed)
    result = {
        "description": (
            "Frozen mature suppressor-set trajectory on the independently "
            "trained Pythia-160M model."
        ),
        "model": args.model_id,
        "condition": "mature_suppressors_full_z",
        "intervention": (
            "Replace each selected head's complete hook_z output at END with "
            "the output from a matched donor that keeps the S2 token fixed."
        ),
        "mature_suppressor_heads": [
            f"L{layer}H{head}" for layer, head in MATURE_SUPPRESSOR_SET
        ],
        "metric": {
            "name": "fair three-way margin",
            "formula": "logit(IO) - max(logit(repeated S), logit(donor alternate))",
        },
        "holdout": holdout,
        "n_items_per_checkpoint": args.n_items,
        "n_observations_per_checkpoint": 2 * args.n_items,
        "by_step": {},
    }
    for step in steps:
        model = load_hooked_model(args.model_id, step, args.device)
        step_result = run_step(model, items, step, args.bootstrap)
        result["by_step"][f"step_{step}"] = {
            "d_margin3_mean": step_result["d_margin3_mean"],
            "d_margin3_ci": step_result["d_margin3_ci"],
            "n_items": step_result["n_items"],
            "n_observations": step_result["n_observations"],
            "bootstrap": step_result["bootstrap"],
        }
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce the held-out and position-shuffle probe controls.

Held-out probe
---------------
For each prompt, the positive example is the S2 residual-stream activation from
the repeated-name prompt and the negative example is the activation at the same
position from its matched non-repeating control. The selection and validation
splits use disjoint names and template families. Each fold trains on 600
activation examples and evaluates on 600 examples; the folds swap the splits.

Position-shuffle probe
----------------------
This is a separate in-distribution control on the 300-prompt, ten-family set.
Within each template batch, one token-position permutation is applied to both
the repeated and control prompts. A standardized ridge classifier is trained on
one random half and evaluated on the other half.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lib.ioi_dataset import build_prompt_records
from lib.locked_ioi_benchmark import IOIBenchmark
from lib.runtime import load_hooked_model, locate_name_occurrence, residual_activations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="teys7007/pythia-160m-seed42-dense")
    parser.add_argument("--steps", default="0,2000,143000")
    parser.add_argument("--layers", default="1,5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/heldout_probe.json"))
    parser.add_argument(
        "--shuffle-output",
        type=Path,
        default=Path("results/position_shuffle_probe.json"),
    )
    return parser.parse_args()


def probe_matrix(model, examples, layers: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    clean = [example.clean_prompt() for example in examples]
    control = [example.dedup_prompt() for example in examples]
    positions = [
        locate_name_occurrence(model.tokenizer, prompt, example.base.S, 2)
        for prompt, example in zip(clean, examples)
    ]
    x_clean = residual_activations(model, clean, positions, layers)
    x_control = residual_activations(model, control, positions, layers)
    matrices = {
        layer: np.concatenate([x_clean[layer], x_control[layer]], axis=0)
        for layer in layers
    }
    labels = np.concatenate([
        np.ones(len(examples), dtype=int),
        np.zeros(len(examples), dtype=int),
    ])
    return matrices, labels


def heldout_accuracy(model, train_examples, test_examples, layers: list[int]) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = probe_matrix(model, train_examples, layers)
    x_test, y_test = probe_matrix(model, test_examples, layers)
    out = {}
    for layer in layers:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=42),
        )
        classifier.fit(x_train[layer], y_train)
        out[f"layer_{layer}"] = float(classifier.score(x_test[layer], y_test))
    return out


def run_heldout(args, steps: list[int], layers: list[int]) -> dict:
    benchmark = IOIBenchmark()
    split_a = benchmark.split("selection").examples
    split_b = benchmark.split("validation").examples
    result = {
        "model": args.model_id,
        "task": {
            "positive_class": "S2 activation from repeated-name prompt",
            "negative_class": "same position from matched non-repeating control",
            "n_train_per_fold": 600,
            "folds": 2,
            "split": "disjoint names and template families; folds swap partitions",
            "classifier": "StandardScaler + logistic regression, C=1.0",
        },
        "by_step": {},
    }
    for step in steps:
        model = load_hooked_model(args.model_id, step, args.device)
        ab = heldout_accuracy(model, split_a, split_b, layers)
        ba = heldout_accuracy(model, split_b, split_a, layers)
        result["by_step"][f"step_{step}"] = {
            key: {
                "trained_on_A": ab[key],
                "trained_on_B": ba[key],
                "mean": float((ab[key] + ba[key]) / 2),
            }
            for key in ab
        }
    return result


def ridge_accuracy(x: np.ndarray, y: np.ndarray, *, lam: float = 50.0) -> float:
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    x = (x - mu) / sd
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    train, test = order[:len(y)//2], order[len(y)//2:]
    xtr = x[train]
    ytr = y[train] * 2 - 1
    w = np.linalg.solve(xtr.T @ xtr + lam * np.eye(xtr.shape[1]), xtr.T @ ytr)
    intercept = ytr.mean() - (xtr @ w).mean()
    pred = (x[test] @ w + intercept) > 0
    return float((pred == (y[test] > 0)).mean())


def run_position_shuffle(args) -> dict:
    import torch

    model_id = "EleutherAI/pythia-160m-deduped"
    model = load_hooked_model(model_id, 0, args.device)
    records = build_prompt_records(model.tokenizer, prompts_per_template=30, seed=42, control_seed=43)
    layers = [1, 2, 3, 6]
    by_template = {
        tid: [record for record in records if record.template_id == tid]
        for tid in range(10)
    }

    def collect(layer: int, shuffled: bool) -> tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        hook = f"blocks.{layer}.hook_resid_post"
        torch.manual_seed(42)
        for template_records in by_template.values():
            clean = torch.stack([
                model.to_tokens(record.prompt, prepend_bos=True)[0]
                for record in template_records
            ]).to(model.cfg.device)
            control = torch.stack([
                model.to_tokens(record.control_prompt, prepend_bos=True)[0]
                for record in template_records
            ]).to(model.cfg.device)
            s2 = torch.tensor([
                locate_name_occurrence(model.tokenizer, record.prompt, record.s_name, 2)
                for record in template_records
            ], device=model.cfg.device)
            if shuffled:
                perm = torch.randperm(clean.shape[1], device=model.cfg.device)
                inverse = perm.argsort()
                clean = clean[:, perm]
                control = control[:, perm]
                s2 = inverse[s2]
            for tokens, label in ((clean, 1), (control, 0)):
                with torch.inference_mode():
                    _, cache = model.run_with_cache(
                        tokens, names_filter=lambda name: name == hook
                    )
                rows = torch.arange(tokens.shape[0], device=model.cfg.device)
                features.append(cache[hook][rows, s2].float().cpu().numpy())
                labels.extend([label] * tokens.shape[0])
        return np.concatenate(features), np.asarray(labels)

    per_layer = {}
    for layer in layers:
        intact_x, labels = collect(layer, False)
        shuffled_x, shuffled_labels = collect(layer, True)
        if not np.array_equal(labels, shuffled_labels):
            raise RuntimeError("Probe labels lost alignment")
        per_layer[str(layer)] = {
            "intact": ridge_accuracy(intact_x, labels),
            "shuffled": ridge_accuracy(shuffled_x, labels),
        }
    return {
        "model": model_id,
        "step": 0,
        "protocol": {
            "distribution": "in-distribution 300-prompt ten-family set",
            "permutation": "one position permutation per template batch",
            "classifier": "standardized ridge regression, lambda=50",
            "split": "random 50/50 train/test split, seed 0",
        },
        "per_layer": per_layer,
    }


def main() -> None:
    args = parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    layers = [int(value) for value in args.layers.split(",")]
    heldout = run_heldout(args, steps, layers)
    shuffled = run_position_shuffle(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(heldout, indent=2) + "\n")
    args.shuffle_output.parent.mkdir(parents=True, exist_ok=True)
    args.shuffle_output.write_text(json.dumps({"0": shuffled}, indent=2) + "\n")


if __name__ == "__main__":
    main()

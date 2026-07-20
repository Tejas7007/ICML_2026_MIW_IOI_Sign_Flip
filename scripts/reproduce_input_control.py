#!/usr/bin/env python3
"""Reproduce the locked 800-prompt input-level de-duplication control.

This is the producer for ``data/locked_input_control_160m.json``. It evaluates
the original prompt, two independently chosen S2 replacement names, and a neutral
filler-token placebo on the deterministic test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lib.cluster_metrics import crossed_cluster_bootstrap
from lib.locked_ioi_benchmark import IOIBenchmark
from lib.runtime import final_logits, load_hooked_model

EXPECTED_TEST_HASH = "34d4fd78419110f21e70f8129a84d992cc6b10d02ddaa4c5d172c6d586ad0553"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/locked_input_control_160m.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    return parser.parse_args()


def score(model, examples, prompt_kind: str, batch_size: int) -> dict[str, np.ndarray]:
    prompts = [getattr(example, f"{prompt_kind}_prompt")() for example in examples]
    logits = final_logits(model, prompts, batch_size=batch_size)
    io_ids = np.asarray([model.to_single_token(" " + ex.base.IO) for ex in examples])
    s_ids = np.asarray([model.to_single_token(" " + ex.base.S) for ex in examples])
    rows = np.arange(len(examples))
    logit_io = logits[rows, io_ids]
    logit_s = logits[rows, s_ids]
    rank_io = 1 + (logits > logit_io[:, None]).sum(axis=1)
    rank_s = 1 + (logits > logit_s[:, None]).sum(axis=1)
    return {
        "logit_IO": logit_io,
        "logit_S": logit_s,
        "rank_IO": rank_io.astype(float),
        "rank_S": rank_s.astype(float),
        "ld": logit_io - logit_s,
    }


def summarize_effect(
    clean,
    changed,
    tids,
    pids,
    *,
    n_boot: int,
    seed: int,
    with_ci: bool = True,
) -> dict:
    delta = changed["ld"] - clean["ld"]
    return {
        "n": int(len(delta)),
        "d_ld_mean": float(delta.mean()),
        "d_ld_ci": (
            list(crossed_cluster_bootstrap(
                delta, tids, pids, n_boot=n_boot, seed=seed,
            ))
            if with_ci else None
        ),
        "d_logit_IO_mean": float((changed["logit_IO"] - clean["logit_IO"]).mean()),
        "d_logit_S_mean": float((changed["logit_S"] - clean["logit_S"]).mean()),
        "d_rank_IO_mean": float((changed["rank_IO"] - clean["rank_IO"]).mean()),
        "d_rank_S_mean": float((changed["rank_S"] - clean["rank_S"]).mean()),
        "acc_clean": float((clean["ld"] > 0).mean()),
        "acc_intervened": float((changed["ld"] > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    benchmark = IOIBenchmark()
    split = benchmark.split("test")
    if split.prompt_hash() != EXPECTED_TEST_HASH:
        raise RuntimeError("Locked benchmark hash does not match the released artifact")

    model = load_hooked_model(args.model_id, args.step, args.device)
    examples = split.examples
    clean = score(model, examples, "clean", args.batch_size)
    dedup = score(model, examples, "dedup", args.batch_size)
    dedup_alt = score(model, examples, "dedup_alt", args.batch_size)
    placebo = score(model, examples, "placebo", args.batch_size)

    tids = [ex.template_idx for ex in examples]
    pids = [ex.pair_id for ex in examples]
    accuracy = (clean["ld"] > 0).astype(float)
    result = {
        "schema": "locked_input_control.v2",
        "model": args.model_id,
        "step": args.step,
        "benchmark": {
            "split": "test",
            "prompt_hash": split.prompt_hash(),
            "n_examples": len(examples),
            "cluster_units": ["template_family", "unordered_name_pair"],
        },
        "metrics": {
            "accuracy": float(accuracy.mean()),
            "accuracy_ci": list(crossed_cluster_bootstrap(
                accuracy, tids, pids, n_boot=args.bootstrap, seed=args.step,
            )),
            "ld_mean": float(clean["ld"].mean()),
            "dedup_effect": summarize_effect(
                clean, dedup, tids, pids,
                n_boot=args.bootstrap, seed=args.step + 1,
            ),
            "dedup_alt_effect": summarize_effect(
                clean, dedup_alt, tids, pids,
                n_boot=args.bootstrap, seed=args.step + 2, with_ci=False,
            ),
            "placebo_effect": summarize_effect(
                clean, placebo, tids, pids,
                n_boot=args.bootstrap, seed=args.step + 2,
            ),
        },
        "protocol": {
            "standard_dedup": "replace S2 with base.third",
            "alternate_dedup": "replace S2 with independent base.third_alt",
            "placebo": "change a neutral place/adjective while preserving S2",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

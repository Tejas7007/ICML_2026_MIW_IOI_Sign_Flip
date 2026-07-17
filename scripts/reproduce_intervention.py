#!/usr/bin/env python3
"""Recompute the core matched-S2 residual-stream intervention.

This is the runnable reference implementation for the historical ten-template,
300-prompt protocol used by the cross-scale paper result. It intentionally
contains no Hugging Face token, no absolute workstation path, and no silent
fallback that skips malformed prompts.

Example
-------
python scripts/reproduce_intervention.py \
    --model pythia-160m \
    --step 2000 \
    --output results/reproduced_pythia-160m_step2000.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from lib.ioi_dataset import (
    build_prompt_records,
    locate_s2_positions,
    prompt_manifest_hash,
)

ROOT = Path(__file__).resolve().parents[1]
PATCH_CONFIG = ROOT / "config" / "patch_windows.json"
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matched-S2 residual patch at one Pythia checkpoint."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "pythia-160m", "pythia-410m", "pythia-1b",
            "pythia-2.8b", "pythia-6.9b", "pythia-12b",
        ),
    )
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--save-per-example",
        action="store_true",
        help="Store aligned clean and patched per-example values in the output JSON.",
    )
    return parser.parse_args()


def bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty array")
    samples = rng.choice(values, size=(draws, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return [float(lo), float(hi)]


def load_model(model_key: str, step: int, device: str | None) -> tuple[Any, str, list[int]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise SystemExit(
            "Model inference requires torch, transformers, and transformer_lens. "
            "Install requirements.txt in a CUDA-capable environment."
        ) from exc

    config = json.loads(PATCH_CONFIG.read_text())
    entry = config["windows"][model_key]
    repo_id = entry["model_id"]
    patch_layers = [int(x) for x in entry["layers"]]
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if resolved_device == "cpu" and model_key in {"pythia-2.8b", "pythia-6.9b", "pythia-12b"}:
        raise SystemExit("Large-scale reproduction requires a CUDA device.")

    dtype = (
        torch.float16
        if model_key in {"pythia-2.8b", "pythia-6.9b", "pythia-12b"}
        else torch.float32
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=f"step{step}",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = HookedTransformer.from_pretrained(
        repo_id,
        hf_model=hf_model,
        device=resolved_device,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
        dtype=dtype,
    )
    model.eval()
    return model, repo_id, patch_layers


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, repo_id, patch_layers = load_model(args.model, args.step, args.device)
    records = build_prompt_records(
        model.tokenizer,
        prompts_per_template=30,
        seed=args.seed,
        control_seed=args.seed + 1,
    )
    s2_positions = locate_s2_positions(model.tokenizer, records)
    prompt_hash = prompt_manifest_hash(records)

    clean_ld: list[float] = []
    patched_ld: list[float] = []
    per_example: list[dict[str, Any]] = []

    def autocast_context():
        if str(model.cfg.device).startswith("cuda") and model.cfg.dtype == torch.float16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        batch_s2 = s2_positions[start : start + args.batch_size]
        prompts = [r.prompt for r in batch_records]
        controls = [r.control_prompt for r in batch_records]

        clean_tokens = model.to_tokens(prompts, prepend_bos=True)
        control_tokens = model.to_tokens(controls, prepend_bos=True)
        if clean_tokens.shape != control_tokens.shape:
            raise RuntimeError("Original and matched-control token tensors differ in shape")

        with torch.inference_mode(), autocast_context():
            control_cache = model.run_with_cache(
                control_tokens,
                names_filter=lambda name: name in {
                    f"blocks.{layer}.hook_resid_post" for layer in patch_layers
                },
            )[1]

            def patch_hook(resid: torch.Tensor, hook: Any) -> torch.Tensor:
                layer = int(hook.name.split(".")[1])
                donor = control_cache[f"blocks.{layer}.hook_resid_post"]
                patched = resid.clone()
                row_index = torch.arange(resid.shape[0], device=resid.device)
                position_index = torch.tensor(batch_s2, device=resid.device) + 1
                patched[row_index, position_index, :] = donor[row_index, position_index, :]
                return patched

            hooks = [
                (f"blocks.{layer}.hook_resid_post", patch_hook)
                for layer in patch_layers
            ]
            clean_logits = model(clean_tokens, return_type="logits")[:, -1, :]
            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=hooks,
                return_type="logits",
            )[:, -1, :]

        for row, record in enumerate(batch_records):
            io_id = model.to_single_token(" " + record.io_name)
            s_id = model.to_single_token(" " + record.s_name)
            base = float((clean_logits[row, io_id] - clean_logits[row, s_id]).item())
            changed = float((patched_logits[row, io_id] - patched_logits[row, s_id]).item())
            clean_ld.append(base)
            patched_ld.append(changed)

            if args.save_per_example:
                per_example.append(
                    {
                        **asdict(record),
                        "s2_token_position_without_bos": int(batch_s2[row]),
                        "clean_ld": base,
                        "patched_ld": changed,
                        "delta_ld": changed - base,
                    }
                )

    clean = np.asarray(clean_ld, dtype=np.float64)
    patched = np.asarray(patched_ld, dtype=np.float64)
    delta = patched - clean

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "matched_s2_residual_patch",
        "model_key": args.model,
        "model_id": repo_id,
        "revision": f"step{args.step}",
        "step": args.step,
        "patch_layers": patch_layers,
        "hook": "blocks.{layer}.hook_resid_post",
        "seed": args.seed,
        "n_prompts": len(records),
        "n_templates": 10,
        "prompts_per_template": 30,
        "template_family": "first ten historical BABA-style IOI templates",
        "symmetric_name_role_swaps": True,
        "prompt_manifest_sha256": prompt_hash,
        "ioi_accuracy": float(np.mean(clean > 0)),
        "base_ld_mean": float(clean.mean()),
        "patched_ld_mean": float(patched.mean()),
        "delta_ld_mean": float(delta.mean()),
        "delta_ld_ci_prompt": bootstrap_mean(delta, draws=args.bootstrap, seed=args.seed),
        "bootstrap_draws": args.bootstrap,
    }
    if args.save_per_example:
        result["examples"] = per_example
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    keys = (
        "model_key", "step", "n_prompts", "ioi_accuracy", "base_ld_mean",
        "delta_ld_mean", "delta_ld_ci_prompt", "prompt_manifest_sha256",
    )
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()

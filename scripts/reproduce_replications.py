#!/usr/bin/env python3
"""Reproduce the Stanford GPT-2 and PolyPythia sign-reversal replications.

The script uses the same 300-prompt, ten-family matched-S2 procedure as the
primary Pythia analysis. The PolyPythia intervention checkpoint is step 2000
for every variant; it is deliberately separate from the 15-family behavioral
floor sweep reported elsewhere.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from lib.ioi_dataset import build_prompt_records
from lib.runtime import bootstrap_mean

PATCH_LAYERS = (3, 4, 5)
POLYPYTHIAS = (
    "EleutherAI/pythia-160m-seed1",
    "EleutherAI/pythia-160m-seed3",
    "EleutherAI/pythia-160m-seed5",
    "EleutherAI/pythia-160m-data-seed1",
    "EleutherAI/pythia-160m-data-seed2",
    "EleutherAI/pythia-160m-data-seed3",
    "EleutherAI/pythia-160m-weight-seed1",
    "EleutherAI/pythia-160m-weight-seed2",
    "EleutherAI/pythia-160m-weight-seed3",
)
STANFORD_ID = "stanford-crfm/alias-gpt2-small-x21"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("stanford", "polypythias", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--stanford-output", type=Path, default=Path("results/stanford_gpt2_signflip.json"))
    parser.add_argument("--polypythia-output", type=Path, default=Path("results/polypythias_signflip_9variants.json"))
    return parser.parse_args()


def resolved_device(requested: str | None) -> str:
    import torch
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def load_stanford(step: int, device: str | None):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        STANFORD_ID,
        revision=f"checkpoint-{step}",
        device=resolved_device(device),
    )
    model.eval()
    return model


def load_polypythia(repo_id: str, step: int, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    hf_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=f"step{step}",
        torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        hf_model=hf_model,
        device=resolved_device(device),
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
        dtype=torch.float32,
    )
    model.eval()
    return model


def evaluate(model, *, bootstrap: int) -> dict:
    import torch

    records = build_prompt_records(
        model.tokenizer,
        prompts_per_template=30,
        seed=42,
        control_seed=43,
    )
    clean_ld: list[float] = []
    patched_ld: list[float] = []

    for template_id in range(10):
        batch = [record for record in records if record.template_id == template_id]
        clean_tokens = torch.stack([
            model.to_tokens(record.prompt, prepend_bos=True)[0]
            for record in batch
        ]).to(model.cfg.device)
        control_tokens = torch.stack([
            model.to_tokens(record.control_prompt, prepend_bos=True)[0]
            for record in batch
        ]).to(model.cfg.device)
        if clean_tokens.shape != control_tokens.shape:
            raise RuntimeError("Original and control token tensors differ in shape")

        s2_positions = []
        io_ids = []
        s_ids = []
        for row, record in enumerate(batch):
            s_id = model.to_single_token(" " + record.s_name)
            io_id = model.to_single_token(" " + record.io_name)
            matches = (clean_tokens[row] == s_id).nonzero(as_tuple=False).flatten()
            if len(matches) != 2:
                raise RuntimeError(f"{record.example_id}: expected two repeated-name tokens")
            s2_positions.append(int(matches[1]))
            io_ids.append(int(io_id))
            s_ids.append(int(s_id))

        hook_names = {f"blocks.{layer}.hook_resid_post" for layer in PATCH_LAYERS}
        with torch.inference_mode():
            _, control_cache = model.run_with_cache(
                control_tokens,
                names_filter=lambda name: name in hook_names,
            )
            base_logits = model(clean_tokens, return_type="logits")[:, -1]

        positions = torch.tensor(s2_positions, device=model.cfg.device)
        rows = torch.arange(len(batch), device=model.cfg.device)
        hooks = []
        for layer in PATCH_LAYERS:
            hook_name = f"blocks.{layer}.hook_resid_post"
            donor = control_cache[hook_name].detach()

            def patch(activation, hook, *, source=donor, row_index=rows, position_index=positions):
                changed = activation.clone()
                changed[row_index, position_index] = source[row_index, position_index].to(changed.dtype)
                return changed

            hooks.append((hook_name, patch))

        with torch.inference_mode():
            changed_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=hooks,
                return_type="logits",
            )[:, -1]

        io = torch.tensor(io_ids, device=model.cfg.device)
        repeated = torch.tensor(s_ids, device=model.cfg.device)
        base = (base_logits[rows, io] - base_logits[rows, repeated]).float().cpu().numpy()
        changed = (changed_logits[rows, io] - changed_logits[rows, repeated]).float().cpu().numpy()
        clean_ld.extend(base.tolist())
        patched_ld.extend(changed.tolist())

    clean = np.asarray(clean_ld, dtype=float)
    patched = np.asarray(patched_ld, dtype=float)
    delta = patched - clean
    return {
        "n_prompts": len(clean),
        "ioi_acc": float(np.mean(clean > 0)),
        "base_ld_mean": float(clean.mean()),
        "base_ld_ci95": bootstrap_mean(clean, draws=bootstrap, seed=42),
        "patched_ld_mean": float(patched.mean()),
        "delta_ld_mean": float(delta.mean()),
        "delta_ld_ci95": bootstrap_mean(delta, draws=bootstrap, seed=42),
    }


def free_model(model) -> None:
    try:
        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def reproduce_stanford(args) -> dict:
    checkpoints = (100, 1500, 3000, 10000, 100000)
    result = {
        "config": {
            "patch_layers": list(PATCH_LAYERS),
            "templates": 10,
            "prompts_per_template": 30,
            "n_total": 300,
            "n_bootstrap": args.bootstrap,
            "model": {
                "name": "stanford_alias",
                "hf_repo": STANFORD_ID,
                "revision_format": "checkpoint-{step}",
                "checkpoints": list(checkpoints),
            },
        },
        "by_model": {"stanford_alias": {}},
    }
    for step in checkpoints:
        model = load_stanford(step, args.device)
        result["by_model"]["stanford_alias"][f"step_{step}"] = evaluate(model, bootstrap=args.bootstrap)
        free_model(model)
    return result


def reproduce_polypythias(args) -> dict:
    result = {
        "protocol": {
            "patch_layers": list(PATCH_LAYERS),
            "templates": 10,
            "prompts_per_template": 30,
            "n_total": 300,
            "window_step": 2000,
            "mature_step": 143000,
            "note": "The intervention checkpoint is separate from the 15-family behavioral-floor sweep.",
        },
        "seeds": {},
    }
    flips = 0
    for repo_id in POLYPYTHIAS:
        stages = {}
        for stage, step in (("dip", 2000), ("mature", 143000)):
            model = load_polypythia(repo_id, step, args.device)
            stages[stage] = evaluate(model, bootstrap=args.bootstrap)
            free_model(model)
        stages["flips"] = bool(
            stages["dip"]["delta_ld_mean"] > 0
            and stages["mature"]["delta_ld_mean"] < 0
        )
        flips += int(stages["flips"])
        result["seeds"][repo_id] = stages
    result["n_flips"] = f"{flips}/{len(POLYPYTHIAS)}"
    return result


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.suite in ("stanford", "all"):
        write(args.stanford_output, reproduce_stanford(args))
    if args.suite in ("polypythias", "all"):
        write(args.polypythia_output, reproduce_polypythias(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_intervention.py
=========================

Reference implementation of the matched-S2 residual-stream intervention from

    "A Training-Time Sign Flip in IOI Circuit Formation"
    ICML 2026 Mechanistic Interpretability Workshop.

Given a Pythia checkpoint, this script measures the effect of replacing the
residual-stream state at the second occurrence of the repeated subject (S2)
with the state from a matched control prompt in which that name is a third,
non-repeating single-token name. A positive effect means the intervention
moves the model toward the correct (indirect-object) name.

Running this on the released checkpoints reproduces the "matched S2 intervention"
column of Table 5 and the trajectory in Figure 2 (centre panel). The patch
window per model is given in ``config/patch_windows.json`` and reproduced in
``MANIFEST.md``; for Pythia-160M it is layers 3 to 5.

This script performs model inference and therefore requires ``torch`` and
``transformer_lens`` (see ``requirements.txt``). It is provided as a readable
specification of the method; the exact numbers reported in the paper are
committed under ``data/`` and checked by ``verify_claims.py``.

Usage
-----
    python scripts/reproduce_intervention.py --model pythia-160m --step 2000
    python scripts/reproduce_intervention.py --model pythia-160m --step 143000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ten single-token-name templates in balanced ABBA / BABA order (Wang et al., 2023).
TEMPLATES = [
    "When {A} and {B} went to the store, {S} gave a drink to",
    "When {A} and {B} went to the park, {S} gave the ball to",
    "Then {A} and {B} went to the shop, {S} handed a book to",
    "After {A} and {B} left the office, {S} passed the keys to",
    "While {A} and {B} sat in the cafe, {S} gave the menu to",
    "Then {A} and {B} were at the school, {S} lent a pen to",
    "When {A} and {B} met at the party, {S} gave a gift to",
    "After {A} and {B} finished lunch, {S} showed the photo to",
    "While {A} and {B} walked to class, {S} gave the notes to",
    "When {A} and {B} arrived at home, {S} handed the mail to",
]

# A small pool of single-token names; the third (donor) name is drawn disjointly.
NAMES = ["John", "Mary", "Tom", "Anna", "Mark", "Kate", "Paul", "Emma", "Luke", "Sara"]

PATCH_WINDOWS = {
    "pythia-160m": [3, 4, 5],
    "pythia-410m": [6, 7, 8, 9, 10],
    "pythia-1b":   [4, 5, 6, 7, 8],
    "pythia-2.8b": [8, 9, 10, 11, 12, 13, 14],
    "pythia-6.9b": [8, 9, 10, 11, 12, 13, 14],
    "pythia-12b":  [9, 10, 11, 12, 13, 14, 15, 16],
}


def build_prompt_pairs(n_per_template: int = 30):
    """Yield (ioi_prompt, control_prompt, io_name, s_name) tuples.

    In the IOI prompt the subject ``S`` equals one of the two introduced names,
    so it is repeated. In the matched control the same slot is filled by a third
    name that does not appear elsewhere, so it is non-repeating. Everything else
    is identical, which isolates the effect of the repetition at S2.
    """
    import random

    rng = random.Random(0)
    for template in TEMPLATES:
        for _ in range(n_per_template):
            a, b = rng.sample(NAMES, 2)
            donor = rng.choice([x for x in NAMES if x not in (a, b)])
            subject, io = (a, b) if rng.random() < 0.5 else (b, a)   # ABBA / BABA balance
            ioi = template.format(A=a, B=b, S=subject)
            control = template.format(A=a, B=b, S=donor)
            yield ioi, control, io, subject


def s2_token_index(model, prompt: str, subject: str) -> int:
    """Index of the token at the second occurrence of the repeated subject."""
    tokens = model.to_str_tokens(prompt)
    hits = [i for i, t in enumerate(tokens) if t.strip() == subject]
    return hits[-1] if hits else len(tokens) - 1


def logit_difference(model, prompt: str, io: str, s: str) -> float:
    """logit(correct name) - logit(repeated name) at the final position."""
    import torch

    logits = model(prompt)[0, -1]
    io_id = model.to_single_token(" " + io)
    s_id = model.to_single_token(" " + s)
    return float(logits[io_id] - logits[s_id])


def run(model_name: str, step: int, n_per_template: int) -> dict:
    """Measure the matched-S2 intervention effect at one checkpoint."""
    import torch
    from transformer_lens import HookedTransformer

    layers = PATCH_WINDOWS[model_name]
    model = HookedTransformer.from_pretrained(
        f"EleutherAI/{model_name}",
        checkpoint_value=step,
    )
    model.eval()

    base_deltas, patched_deltas = [], []
    for ioi, control, io, s in build_prompt_pairs(n_per_template):
        idx = s2_token_index(model, ioi, s)

        # Cache the control's residual stream at the S2 position for the patch window.
        _, control_cache = model.run_with_cache(control)
        donor = {L: control_cache[f"blocks.{L}.hook_resid_post"][0, idx].clone() for L in layers}

        def patch(resid, hook, layer=None):
            resid[0, idx] = donor[layer]
            return resid

        hooks = [(f"blocks.{L}.hook_resid_post", lambda r, h, L=L: patch(r, h, L)) for L in layers]

        base = logit_difference(model, ioi, io, s)
        with model.hooks(fwd_hooks=hooks):
            patched_logits = model(ioi)[0, -1]
            io_id = model.to_single_token(" " + io)
            s_id = model.to_single_token(" " + s)
            patched = float(patched_logits[io_id] - patched_logits[s_id])

        base_deltas.append(base)
        patched_deltas.append(patched - base)

    import statistics

    return {
        "model": model_name,
        "step": step,
        "patch_window": layers,
        "n_prompts": len(base_deltas),
        "mean_intervention_effect": round(statistics.fmean(patched_deltas), 4),
        "mean_base_logit_difference": round(statistics.fmean(base_deltas), 4),
        "accuracy": round(sum(d > 0 for d in base_deltas) / len(base_deltas), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="pythia-160m", choices=sorted(PATCH_WINDOWS))
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--n-per-template", type=int, default=30)
    args = parser.parse_args()

    result = run(args.model, args.step, args.n_per_template)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

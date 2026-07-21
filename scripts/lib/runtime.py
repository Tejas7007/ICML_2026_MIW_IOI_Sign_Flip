"""Small runtime helpers shared by appendix reproduction scripts."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def load_hooked_model(model_id: str, step: int, device: str | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before running model experiments") from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=f"step{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        model_id,
        hf_model=hf_model,
        device=resolved_device,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
        dtype=torch.float32,
    )
    model.eval()
    return model


def token_ids(tokenizer, prompt: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]


def locate_name_occurrence(tokenizer, prompt: str, name: str, occurrence: int) -> int:
    """Return token position after the prepended BOS token."""
    target = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(target) != 1:
        raise ValueError(f"{name!r} is not a single token in context")
    ids = token_ids(tokenizer, prompt)
    matches = [i for i, tok in enumerate(ids) if tok == int(target[0])]
    if len(matches) < occurrence:
        raise ValueError(f"Could not find occurrence {occurrence} of {name!r}")
    return matches[occurrence - 1] + 1


def _group_rows(model, prompts: Sequence[str]):
    rows = [model.to_tokens(prompt, prepend_bos=True)[0] for prompt in prompts]
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row.numel())].append(i)
    return rows, groups


def final_logits(model, prompts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Run prompts without padding artifacts by batching only equal-length rows."""
    import torch

    rows, groups = _group_rows(model, prompts)
    result: list[np.ndarray | None] = [None] * len(prompts)
    with torch.inference_mode():
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start:start + batch_size]
                tokens = torch.stack([rows[i] for i in chunk]).to(model.cfg.device)
                logits = model(tokens, return_type="logits")[:, -1].float().cpu().numpy()
                for local, original in enumerate(chunk):
                    result[original] = logits[local]
    if any(value is None for value in result):
        raise RuntimeError("Missing model output")
    return np.stack(result)  # type: ignore[arg-type]


def residual_activations(
    model,
    prompts: Sequence[str],
    positions: Sequence[int],
    layers: Iterable[int],
    batch_size: int = 64,
) -> dict[int, np.ndarray]:
    """Read residual-stream activations at one aligned position per prompt."""
    import torch

    if len(prompts) != len(positions):
        raise ValueError("prompts and positions must be aligned")
    wanted = sorted(set(int(layer) for layer in layers))
    rows, groups = _group_rows(model, prompts)
    outputs: dict[int, list[np.ndarray | None]] = {
        layer: [None] * len(prompts) for layer in wanted
    }
    hooks = {f"blocks.{layer}.hook_resid_post" for layer in wanted}
    with torch.inference_mode():
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start:start + batch_size]
                tokens = torch.stack([rows[i] for i in chunk]).to(model.cfg.device)
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name in hooks,
                )
                local_rows = torch.arange(len(chunk), device=tokens.device)
                local_pos = torch.tensor(
                    [positions[i] for i in chunk], device=tokens.device
                )
                for layer in wanted:
                    values = cache[f"blocks.{layer}.hook_resid_post"][
                        local_rows, local_pos
                    ].float().cpu().numpy()
                    for local, original in enumerate(chunk):
                        outputs[layer][original] = values[local]
                del cache
    return {
        layer: np.stack(values)  # type: ignore[arg-type]
        for layer, values in outputs.items()
    }


def bootstrap_mean(values: Sequence[float], *, draws: int = 10_000, seed: int = 42):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(draws, len(arr)))
    means = arr[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]

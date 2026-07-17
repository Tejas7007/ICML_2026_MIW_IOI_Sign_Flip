#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============

Regenerates the three figures in

    "A Training-Time Sign Flip in IOI Circuit Formation"
    ICML 2026 Mechanistic Interpretability Workshop

directly from the released result files in ``data/``. Running this script
reproduces ``figures/fig1_below_chance_dip.pdf``,
``figures/fig2_sign_flip.pdf`` and ``figures/fig3_loss_vs_accuracy.pdf``.

All figures embed TrueType (Type 42) fonts. No model inference is performed;
the figures are drawn from the same numbers that ``verify_claims.py`` checks.

Usage
-----
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # embed TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "serif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)


def _load(name: str) -> dict:
    with open(DATA / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _accuracy_series(by_step: dict):
    pts = []
    for key, rec in by_step.items():
        step = int(key.split("_")[1])
        if step > 0 and rec.get("ioi_acc") is not None:
            pts.append((step, 100 * rec["ioi_acc"]))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


# --------------------------------------------------------------------------- #
# Figure 1: the below-chance window across six Pythia scales                  #
# --------------------------------------------------------------------------- #

def figure_1() -> None:
    small = _load("signflip_across_scale_160m_410m_1b.json")
    big = _load("behavior_6.9b_12b.json")
    blob = json.dumps(_load("behavior_2.8b.json"))

    def series_2_8b():
        pairs = re.findall(r'step_?(\d+)"?\s*:\s*\{[^{}]*?"ioi_acc"\s*:\s*([0-9.]+)', blob)
        pts = sorted((int(a), 100 * float(b)) for a, b in pairs if int(a) > 0)
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    curves = [
        ("160M", *_accuracy_series(small["pythia-160m"]["by_step"]), "#1f77b4"),
        ("410M", *_accuracy_series(small["pythia-410m"]["by_step"]), "#ff7f0e"),
        ("1B",   *_accuracy_series(small["pythia-1b"]["by_step"]),   "#2ca02c"),
        ("2.8B", *series_2_8b(),                                     "#9467bd"),
        ("6.9B", *_accuracy_series(big["pythia-6.9b"]["by_step"]),   "#8c564b"),
        ("12B",  *_accuracy_series(big["pythia-12b"]["by_step"]),    "#e377c2"),
    ]
    for label, xs, ys, color in curves:
        ax.plot(xs, ys, "-o", ms=2.0, lw=0.9, color=color, label=label)
    ax.axhline(50, ls="--", lw=0.8, color="0.45")
    ax.text(1.3e5, 52, "chance", fontsize=6, color="0.45", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("IOI accuracy (%)")
    ax.set_ylim(0, 104)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right", columnspacing=1.0)
    ax.set_title("Below-chance window across six scales", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "fig1_below_chance_dip.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: accuracy, matched-S2 intervention, and suppressor mean-ablation   #
# --------------------------------------------------------------------------- #

def figure_2() -> None:
    small = _load("signflip_across_scale_160m_410m_1b.json")["pythia-160m"]["by_step"]
    rows = []
    for key, rec in small.items():
        step = int(key.split("_")[1])
        if step > 0 and rec.get("ioi_acc") is not None and rec.get("delta_ld_mean") is not None:
            ci = rec.get("delta_ld_ci", [rec["delta_ld_mean"]] * 2)
            rows.append((step, 100 * rec["ioi_acc"], rec["delta_ld_mean"], ci[0], ci[1]))
    rows.sort()
    steps = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    eff = [r[2] for r in rows]
    lo = [r[3] for r in rows]
    hi = [r[4] for r in rows]

    supp = _load("suppressor_ablation_trajectory.json")["pythia-160m"]["by_step"]
    srows = []
    for key, rec in supp.items():
        step = int(key.split("_")[1])
        ci = rec.get("ablation_ci", [rec["ablation_delta"]] * 2)
        srows.append((step, rec["ablation_delta"], ci[0], ci[1]))
    srows.sort()
    ssteps = [r[0] for r in srows]
    seff = [r[1] for r in srows]
    slo = [r[2] for r in srows]
    shi = [r[3] for r in srows]

    floor_step, above_step = 2000, 3000
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.15))

    def markers(axis):
        axis.axvline(floor_step, ls=":", lw=0.8, color="0.45")
        axis.axvline(above_step, ls=":", lw=0.8, color="0.45")

    ax[0].plot(steps, acc, "-o", ms=3, lw=1.1, color="#1f77b4")
    ax[0].axhline(50, ls="--", lw=0.7, color="0.5")
    ax[0].set_ylabel("IOI accuracy (%)")
    ax[0].set_ylim(20, 105)
    ax[0].set_title("Accuracy", fontsize=8)
    markers(ax[0])

    ax[1].fill_between(steps, lo, hi, color="#7b3294", alpha=0.22, lw=0)
    ax[1].plot(steps, eff, "-o", ms=3, lw=1.1, color="#7b3294")
    ax[1].axhline(0, lw=0.6, color="0.3")
    ax[1].set_ylabel("S2 intervention effect")
    ax[1].set_title("Matched S2 intervention", fontsize=8)
    markers(ax[1])

    ax[2].fill_between(ssteps, slo, shi, color="#008837", alpha=0.20, lw=0)
    ax[2].plot(ssteps, seff, "-o", ms=3, lw=1.1, color="#008837")
    ax[2].axhline(0, lw=0.6, color="0.3")
    ax[2].set_ylabel("Suppressor mean-ablation effect")
    ax[2].set_title("Mean-ablate suppressor", fontsize=8)
    markers(ax[2])

    for axis in ax:
        axis.set_xscale("log")
        axis.set_xlabel("Training step")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(FIGS / "fig2_sign_flip.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: Pile evaluation-sample loss and accuracy, two aligned panels      #
# --------------------------------------------------------------------------- #

def figure_3() -> None:
    loss = _load("loss_and_head_sweep.json")["exp_c_loss_comparison"]
    ls = sorted(int(k.split("_")[1]) for k in loss)
    lv = [loss[f"step_{s}"]["original_loss"] for s in ls]

    small = _load("signflip_across_scale_160m_410m_1b.json")["pythia-160m"]["by_step"]
    acc_by_step = {
        int(k.split("_")[1]): 100 * r["ioi_acc"]
        for k, r in small.items()
        if r.get("ioi_acc") is not None
    }

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.3, 3.0), sharex=True)
    top.plot(ls, lv, "-o", ms=3, lw=1.1, color="#444")
    top.set_ylabel("Pile eval-sample loss")
    top.axvspan(1500, 3000, color="#1f77b4", alpha=0.10, zorder=0)
    top.set_title("Loss decreases across the sampled checkpoints", fontsize=8)

    ax = [s for s in ls if s in acc_by_step]
    bottom.plot(ax, [acc_by_step[s] for s in ax], "-s", ms=3, lw=1.1, color="#1f77b4")
    bottom.axhline(50, ls="--", lw=0.7, color="0.5")
    bottom.axvspan(1500, 3000, color="#1f77b4", alpha=0.10, zorder=0)
    bottom.set_ylabel("IOI accuracy (%)")
    bottom.set_xlabel("Training step")
    bottom.set_xscale("log")
    bottom.set_ylim(20, 102)
    fig.tight_layout(pad=0.3, h_pad=0.5)
    fig.savefig(FIGS / "fig3_loss_vs_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    figure_1()
    figure_2()
    figure_3()
    print("Regenerated:")
    for name in ("fig1_below_chance_dip.pdf", "fig2_sign_flip.pdf", "fig3_loss_vs_accuracy.pdf"):
        print("  figures/" + name)


if __name__ == "__main__":
    main()

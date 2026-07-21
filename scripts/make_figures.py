#!/usr/bin/env python3
"""Regenerate the three paper figures from the committed JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
PDF_METADATA = {
    "Title": "A Training-Time Sign Flip in IOI Circuit Formation",
    "Author": "Tejas Dahiya and Cole Blondin",
    "Creator": "scripts/make_figures.py",
    "CreationDate": None,
    "ModDate": None,
}


def load(name: str) -> Any:
    return json.loads((DATA / name).read_text())


def rows(by_step: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    output = []
    for key, value in by_step.items():
        if not key.startswith("step_") or not isinstance(value, dict):
            continue
        if "error" in value or value.get("status") == "error":
            continue
        output.append((int(key.split("_", 1)[1]), value))
    return sorted(output)


def save(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(
        out_dir / f"{stem}.png",
        bbox_inches="tight",
        dpi=240,
        metadata={"Software": "scripts/make_figures.py"},
    )
    plt.close(fig)


def figure1(out_dir: Path) -> None:
    small = load("signflip_across_scale_160m_410m_1b.json")
    middle = load("behavior_2.8b.json")
    large = load("behavior_6.9b_12b.json")
    series = {
        "160M": rows(small["pythia-160m"]["by_step"]),
        "410M": rows(small["pythia-410m"]["by_step"]),
        "1B": rows(small["pythia-1b"]["by_step"]),
        "2.8B": rows(middle["by_step"]),
        "6.9B": rows(large["pythia-6.9b"]["by_step"]),
        "12B": rows(large["pythia-12b"]["by_step"]),
    }
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    for label, values in series.items():
        step = np.asarray([x for x, _ in values], dtype=float)
        accuracy = 100 * np.asarray([float(value["ioi_acc"]) for _, value in values])
        ax.plot(step, accuracy, marker="o", linewidth=1.3, markersize=2.8, label=label)
    ax.axhline(50, linestyle="--", linewidth=0.8, color="0.55")
    ax.text(1.05e5, 52, "chance", ha="right", va="bottom", color="0.45", fontsize=7)
    ax.set(
        xscale="log",
        ylim=(0, 105),
        xlabel="Training step",
        ylabel="IOI accuracy (%)",
        title="Below-chance window across six scales",
    )
    ax.legend(ncol=2, frameon=False, loc="lower right")
    save(fig, "fig1_below_chance_dip", out_dir)


def figure2(out_dir: Path) -> None:
    trajectory = rows(
        load("signflip_across_scale_160m_410m_1b.json")["pythia-160m"]["by_step"]
    )
    suppressor = rows(
        load("suppressor_ablation_trajectory.json")["pythia-160m"]["by_step"]
    )
    step = np.asarray([x for x, _ in trajectory], dtype=float)
    accuracy = 100 * np.asarray([float(value["ioi_acc"]) for _, value in trajectory])
    effect = np.asarray([float(value["delta_ld_mean"]) for _, value in trajectory])
    effect_lo = np.asarray([float(value["delta_ld_ci"][0]) for _, value in trajectory])
    effect_hi = np.asarray([float(value["delta_ld_ci"][1]) for _, value in trajectory])
    suppressor_step = np.asarray([x for x, _ in suppressor], dtype=float)
    suppressor_effect = np.asarray(
        [float(value["ablation_delta"]) for _, value in suppressor]
    )
    suppressor_lo = np.asarray(
        [float(value["ablation_ci"][0]) for _, value in suppressor]
    )
    suppressor_hi = np.asarray(
        [float(value["ablation_ci"][1]) for _, value in suppressor]
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    for ax in axes:
        ax.set_xscale("log")
        ax.axvspan(2000, 3000, color="0.88", alpha=0.45, zorder=0)
        ax.axvline(2000, linestyle=":", linewidth=0.7, color="0.55")
        ax.axvline(3000, linestyle=":", linewidth=0.7, color="0.55")
        ax.set_xlabel("Training step")

    axes[0].plot(step, accuracy, marker="o", linewidth=1.4, markersize=3)
    axes[0].axhline(50, linestyle="--", linewidth=0.8, color="0.55")
    axes[0].set(ylabel="IOI accuracy (%)", title="Accuracy", ylim=(20, 105))
    axes[0].annotate(
        "floor",
        (2000, 31.7),
        xytext=(0, -15),
        textcoords="offset points",
        ha="center",
        fontsize=7,
    )
    axes[0].annotate(
        ">chance",
        (3000, 57.3),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=7,
    )

    axes[1].plot(step, effect, marker="o", linewidth=1.4, markersize=3)
    axes[1].fill_between(step, effect_lo, effect_hi, alpha=0.18)
    axes[1].axhline(0, linewidth=0.8, color="0.45")
    axes[1].set(ylabel="S2 intervention effect", title="Matched S2 intervention")

    axes[2].plot(
        suppressor_step,
        suppressor_effect,
        marker="o",
        linewidth=1.4,
        markersize=3,
    )
    axes[2].fill_between(suppressor_step, suppressor_lo, suppressor_hi, alpha=0.18)
    axes[2].axhline(0, linewidth=0.8, color="0.45")
    axes[2].set(
        ylabel="Suppressor mean-ablation effect",
        title="Mean-ablate suppressor",
    )
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_sign_flip", out_dir)


def figure3(out_dir: Path) -> None:
    loss = load("pile_loss_sample.json")
    trajectory = rows(
        load("signflip_across_scale_160m_410m_1b.json")["pythia-160m"]["by_step"]
    )
    loss_rows = sorted(
        (
            int(key.split("_", 1)[1]),
            float(value["released_pythia_160m_loss"]),
        )
        for key, value in loss["by_step"].items()
    )
    loss_step = np.asarray([x for x, _ in loss_rows], dtype=float)
    loss_value = np.asarray([y for _, y in loss_rows])
    accuracy_step = np.asarray([x for x, _ in trajectory], dtype=float)
    accuracy_value = 100 * np.asarray(
        [float(value["ioi_acc"]) for _, value in trajectory]
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True)
    axes[0].plot(loss_step, loss_value, marker="o", linewidth=1.3, markersize=3)
    axes[0].set(
        ylabel="Pile sample loss",
        title="Loss decreases across sampled checkpoints",
    )
    axes[1].plot(
        accuracy_step,
        accuracy_value,
        marker="o",
        linewidth=1.3,
        markersize=3,
    )
    axes[1].axhline(50, linestyle="--", linewidth=0.8, color="0.55")
    axes[1].set(
        xscale="log",
        ylim=(20, 105),
        xlabel="Training step",
        ylabel="IOI accuracy (%)",
    )
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig3_loss_vs_accuracy", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()
    selected = {"1", "2", "3"} if args.figure == "all" else {args.figure}
    if "1" in selected:
        print("Generating Figure 1", flush=True)
        figure1(args.output_dir)
    if "2" in selected:
        print("Generating Figure 2", flush=True)
        figure2(args.output_dir)
    if "3" in selected:
        print("Generating Figure 3", flush=True)
        figure3(args.output_dir)
    print(f"Wrote selected paper figures to {args.output_dir}")


if __name__ == "__main__":
    main()

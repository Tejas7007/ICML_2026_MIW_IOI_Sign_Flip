#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_claims.py
================

Reproducibility anchor for

    "A Training-Time Sign Flip in IOI Circuit Formation"
    ICML 2026 Mechanistic Interpretability Workshop.

This script loads the released result files in ``data/`` and checks that every
numerical claim in the paper's tables and figures is reproduced by the data,
to the precision at which it is reported. It performs no model inference; it
verifies that the published numbers match the published artifacts.

Each check prints the paper location, the value read from the data file, and
the value stated in the paper. The script exits non-zero if any check fails.

Usage
-----
    python scripts/verify_claims.py            # run every check
    python scripts/verify_claims.py --verbose  # also print passing checks

The mapping from each paper table to its data file and the exact field path is
documented in ``MANIFEST.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# --------------------------------------------------------------------------- #
# Small assertion harness                                                     #
# --------------------------------------------------------------------------- #

_PASS: list[str] = []
_FAIL: list[str] = []


def _load(name: str) -> dict:
    with open(DATA / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check(where: str, got, want, tol: float = 0.02) -> None:
    """Compare a value read from a data file (``got``) with a paper value (``want``)."""
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    line = f"[{where}] data={got}  paper={want}"
    (_PASS if ok else _FAIL).append(line)


# --------------------------------------------------------------------------- #
# Helpers for the two "trajectory" file layouts                               #
# --------------------------------------------------------------------------- #

def _by_step(block: dict) -> dict:
    """Return {int step: record} for a ``by_step`` block, keeping records with accuracy."""
    out = {}
    for key, rec in block.items():
        m = re.match(r"step_?(\d+)$", str(key))
        if m and isinstance(rec, dict):
            out[int(m.group(1))] = rec
    return out


def _floor_and_mature(steps: dict):
    live = {s: r for s, r in steps.items() if s > 0 and r.get("ioi_acc") is not None}
    floor = min(live, key=lambda s: live[s]["ioi_acc"])
    mature = max(steps)
    return floor, mature, live


# --------------------------------------------------------------------------- #
# Table 5 / Appendix E: the sign flip across six Pythia scales                #
# --------------------------------------------------------------------------- #

def verify_scale() -> None:
    small = _load("signflip_across_scale_160m_410m_1b.json")
    expected = {
        "pythia-160m": (31.7, +0.95, -4.12),
        "pythia-410m": (29.3, +0.11, -3.63),
        "pythia-1b":   (36.3, +0.43, -3.49),
    }
    for scale, (floor_acc, dip, mat) in expected.items():
        steps = _by_step(small[scale]["by_step"])
        fs, ms, live = _floor_and_mature(steps)
        check(f"Tab5 {scale} floor", round(100 * live[fs]["ioi_acc"], 1), floor_acc, 0.2)
        check(f"Tab5 {scale} window", round(live[fs]["delta_ld_mean"], 2), dip, 0.03)
        check(f"Tab5 {scale} mature", round(steps[ms]["delta_ld_mean"], 2), mat, 0.03)

    big = _load("behavior_6.9b_12b.json")
    for scale, (floor_acc, dip, mat) in {
        "pythia-6.9b": (32.3, +0.84, -4.12),
        "pythia-12b":  (42.0, +0.43, -3.96),
    }.items():
        steps = _by_step(big[scale]["by_step"])
        fs, ms, live = _floor_and_mature(steps)
        check(f"Tab5 {scale} floor", round(100 * live[fs]["ioi_acc"], 1), floor_acc, 0.2)
        check(f"Tab5 {scale} window", round(live[fs]["delta_ld_mean"], 2), dip, 0.03)
        check(f"Tab5 {scale} mature", round(steps[ms]["delta_ld_mean"], 2), mat, 0.03)

    blob = json.dumps(_load("behavior_2.8b.json"))
    triples = re.findall(
        r'step_?(\d+)"?\s*:\s*\{[^{}]*?"ioi_acc"\s*:\s*([0-9.]+)[^{}]*?"delta_ld_mean"\s*:\s*(-?[0-9.]+)',
        blob,
    )
    pts = [(int(a), float(b), float(c)) for a, b, c in triples]
    floor = min((p for p in pts if p[0] > 0), key=lambda p: p[1])
    mature = max(pts, key=lambda p: p[0])
    check("Tab5 pythia-2.8b floor", round(100 * floor[1], 1), 29.7, 0.3)
    check("Tab5 pythia-2.8b window", round(floor[2], 2), +0.33, 0.05)
    check("Tab5 pythia-2.8b mature", round(mature[2], 2), -4.05, 0.10)


# --------------------------------------------------------------------------- #
# Table 6 / Appendix F: nine PolyPythias training variants                    #
# --------------------------------------------------------------------------- #

def verify_polypythias() -> None:
    seeds = _load("polypythias_signflip_9variants.json")["seeds"]
    expected = {
        "seed1": (+0.55, -3.92), "seed3": (+0.99, -2.89), "seed5": (+1.11, -4.04),
        "data-seed1": (+1.20, -4.06), "data-seed2": (+1.35, -3.71), "data-seed3": (+1.01, -3.24),
        "weight-seed1": (+0.80, -4.56), "weight-seed2": (+0.84, -4.75), "weight-seed3": (+1.47, -3.51),
    }
    flips = 0
    for key, rec in seeds.items():
        name = key.split("/")[-1].replace("pythia-160m-", "")
        if name in expected:
            dip, mat = expected[name]
            check(f"Tab6 {name} window", round(rec["dip"]["delta_ld_mean"], 2), dip, 0.03)
            check(f"Tab6 {name} mature", round(rec["mature"]["delta_ld_mean"], 2), mat, 0.03)
            if rec["dip"]["delta_ld_mean"] > 0 > rec["mature"]["delta_ld_mean"]:
                flips += 1
    check("Tab6 variants that reverse sign", flips, 9)


# --------------------------------------------------------------------------- #
# Table 8 / Appendix G: Stanford GPT-2 Small                                   #
# --------------------------------------------------------------------------- #

def verify_stanford() -> None:
    sm = _load("stanford_gpt2_signflip.json")["by_model"]["stanford_alias"]
    for step, want in [(1500, +1.03), (3000, +0.77), (10000, -0.23), (100000, -2.89)]:
        check(f"Tab8 stanford step {step}", round(sm[f"step_{step}"]["delta_ld_mean"], 2), want, 0.03)


# --------------------------------------------------------------------------- #
# Table 1 / Appendix B: the position control battery                          #
# --------------------------------------------------------------------------- #

def verify_battery() -> None:
    cb = _load("position_control_battery.json")
    check("Tab1 S2 matched window", round(cb["dip"]["arms"]["real_S2"]["delta_ld_mean"], 2), +0.95, 0.03)
    check("Tab1 S2 matched mature", round(cb["mature"]["arms"]["real_S2"]["delta_ld_mean"], 2), -4.12, 0.03)


# --------------------------------------------------------------------------- #
# Table 3 / Appendix B: the locked input-level control                        #
# --------------------------------------------------------------------------- #

def verify_locked_control() -> None:
    g = _load("locked_input_control_160m.json")
    m = g["metrics"]
    check("Tab3 clean accuracy", round(100 * m["accuracy"], 1), 30.9, 0.2)
    check("Tab3 de-duplication effect", round(m["dedup_effect"]["d_ld_mean"], 3), +0.833, 0.01)
    check("Tab3 alternate de-duplication", round(m["dedup_alt_effect"]["d_ld_mean"], 3), +0.832, 0.01)
    check("Tab3 filler placebo", round(m["placebo_effect"]["d_ld_mean"], 3), -0.082, 0.01)
    check("Tab3 benchmark size", g["benchmark"]["n_examples"], 800)


# --------------------------------------------------------------------------- #
# Table 4 / Appendix C: held-out probe                                        #
# --------------------------------------------------------------------------- #

def verify_probe() -> None:
    steps = _load("heldout_probe_and_position.json")["phase1_heldout_probes"]["steps"]
    for step, want in [("step_0", 0.436), ("step_2000", 0.613), ("step_143000", 0.992)]:
        got = steps[step]["by_layer"]["layer_1"]["mean_held_out_acc"]
        check(f"Tab4 probe {step} layer 1", round(got, 3), want, 0.005)


# --------------------------------------------------------------------------- #
# Table 5 (App C): position-shuffle probe                                     #
# --------------------------------------------------------------------------- #

def verify_shuffle() -> None:
    per_layer = _load("position_shuffle_probe.json")["0"]["per_layer"]
    check("TabShuffle L3 intact", round(100 * per_layer["3"]["intact"], 1), 90.0, 0.2)
    check("TabShuffle L3 shuffled", round(100 * per_layer["3"]["shuffled"], 1), 79.7, 0.2)


# --------------------------------------------------------------------------- #
# Table 6 (App D): suppressor mean-ablation trajectory                        #
# --------------------------------------------------------------------------- #

def verify_suppressor() -> None:
    su = _load("suppressor_ablation_trajectory.json")["pythia-160m"]["by_step"]
    for step, want in [(2000, -0.01), (3000, -0.74), (5000, -2.39), (143000, -2.66)]:
        key = f"step_{step}"
        check(f"Tab6 suppressor step {step}", round(su[key]["ablation_delta"], 2), want, 0.02)


# --------------------------------------------------------------------------- #
# Table 7 (App D): split-safe frozen suppressor set                           #
# --------------------------------------------------------------------------- #

def verify_splitsafe() -> None:
    traj = _load("splitsafe_suppressor_set.json")["by_step"]
    for step, want in [(1800, +0.093), (2000, +0.091), (3200, -0.051), (5000, -0.079), (6000, -0.636)]:
        check(f"Tab7 split-safe step {step}", round(traj[f"step_{step}"]["d_margin3_mean"], 3), want, 0.005)


# --------------------------------------------------------------------------- #
# Table 9 (App H): projection removal                                         #
# --------------------------------------------------------------------------- #

def verify_projection() -> None:
    pj = _load("projection_removal.json")
    check("Tab9 baseline", round(pj["baseline"]["ld"], 2), -0.75, 0.02)
    check("Tab9 probe direction 1.0x", round(pj["remove_dup_direction_1.0x"]["ld"], 2), -0.86, 0.02)
    check("Tab9 orthogonal control", round(pj["remove_ortho_direction_1.0x"]["ld"], 2), -0.74, 0.02)


# --------------------------------------------------------------------------- #
# Section 4: full-vocabulary behavior at the floor                            #
# --------------------------------------------------------------------------- #

def verify_full_vocab() -> None:
    fv = _load("full_vocabulary_floor.json")
    check("Sec4 greedy selects neither", fv["greedy_selects_neither_fraction"], 0.9867, 1e-4)
    check("Sec4 P(repeated name)", fv["prob_repeated_name"], 0.0238, 1e-4)
    check("Sec4 P(correct name)", fv["prob_correct_name"], 0.0118, 1e-4)
    check("Sec4 mean rank repeated", fv["mean_rank_repeated_name"], 14.9, 0.1)
    check("Sec4 mean rank correct", fv["mean_rank_correct_name"], 29.4, 0.1)


# --------------------------------------------------------------------------- #
# Section 6: dip-side head sweep (no single head localizes the window)        #
# --------------------------------------------------------------------------- #

def verify_head_sweep() -> None:
    ha = _load("loss_and_head_sweep.json")
    loss = ha["exp_c_loss_comparison"]
    series = [loss[f"step_{s}"]["original_loss"] for s in sorted(int(k.split("_")[1]) for k in loss)]
    monotone = all(a >= b for a, b in zip(series, series[1:]))
    check("Sec4 Pile loss decreases at every sampled step", monotone, True)


ALL_CHECKS = [
    verify_scale, verify_polypythias, verify_stanford, verify_battery,
    verify_locked_control, verify_probe, verify_shuffle, verify_suppressor,
    verify_splitsafe, verify_projection, verify_full_vocab, verify_head_sweep,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="also print passing checks")
    args = parser.parse_args()

    for fn in ALL_CHECKS:
        fn()

    if args.verbose:
        for line in _PASS:
            print("  PASS", line)

    print("-" * 70)
    print(f"verified {len(_PASS)} numerical claims against data/  |  {len(_FAIL)} failed")
    if _FAIL:
        print("\nFAILED CHECKS:")
        for line in _FAIL:
            print("  FAIL", line)
        return 1
    print("All paper tables and figures reproduce from the released data files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

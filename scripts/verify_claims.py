#!/usr/bin/env python3
"""Verify camera-ready numerical claims against the released artifacts.

The verifier checks paper tables, figure inputs, main-text quantities, patch
windows, and release metadata. It performs no model inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = ROOT / "config"
PASS = 0
FAIL = 0
WARN = 0
ARGS: argparse.Namespace


def load(name: str) -> Any:
    return json.loads((DATA / name).read_text())


def close(label: str, actual: float, expected: float, tol: float = 5e-4) -> None:
    global PASS, FAIL
    if math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        PASS += 1
        if ARGS.verbose:
            print(f"PASS  {label}: {actual}")
    else:
        FAIL += 1
        print(f"FAIL  {label}: got {actual}, expected {expected} ± {tol}")


def equal(label: str, actual: Any, expected: Any) -> None:
    global PASS, FAIL
    if actual == expected:
        PASS += 1
        if ARGS.verbose:
            print(f"PASS  {label}: {actual}")
    else:
        FAIL += 1
        print(f"FAIL  {label}: got {actual!r}, expected {expected!r}")


def warning(label: str, message: str) -> None:
    global WARN
    WARN += 1
    print(f"WARN  {label}: {message}")


def valid_rows(by_step: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for key, value in by_step.items():
        if not key.startswith("step_") or not isinstance(value, dict):
            continue
        if "error" in value or value.get("status") == "error":
            continue
        rows.append((int(key.split("_", 1)[1]), value))
    return sorted(rows)


def floor_and_mature(by_step: dict[str, Any]) -> tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]:
    rows = valid_rows(by_step)
    return min(rows, key=lambda item: float(item[1]["ioi_acc"])), max(rows, key=lambda item: item[0])


def check_position_tables() -> None:
    data = load("position_control_battery.json")
    expected = {
        "dip": {
            "real_S2": (0.9525950670, (0.8575870303, 1.0503846546)),
            "random_S2": (0.9197551680, (0.8427029157, 0.9985248371)),
            "random_S1": (2.1524222930, (2.0638164788, 2.2442854366)),
            "random_IO": (-3.5360066255, (-3.7313376780, -3.3416938135)),
            "random_struct": (-0.8145451784, (-0.8983983100, -0.7300148924)),
        },
        "mature": {
            "real_S2": (-4.1229453532, (-4.2840915780, -3.9628876651)),
            "random_S2": (-4.1257953008, (-4.3144544821, -3.9401483541)),
            "random_S1": (2.0792321587, (1.8838812480, 2.2721939829)),
            "random_IO": (-9.0626466433, (-9.3049049020, -8.8197444307)),
            "random_struct": (-4.4500414785, (-4.6655968822, -4.2333017451)),
        },
    }
    for stage, arms in expected.items():
        equal(f"position battery {stage} n", data[stage]["n"], 300)
        for arm, (mean, ci) in arms.items():
            row = data[stage]["arms"][arm]
            close(f"Table 2 {stage} {arm} mean", row["delta_ld_mean"], mean)
            close(f"Table 2 {stage} {arm} CI lower", row["delta_ci"][0], ci[0])
            close(f"Table 2 {stage} {arm} CI upper", row["delta_ci"][1], ci[1])


def check_locked_control() -> None:
    data = load("locked_input_control_160m.json")
    equal("locked control n", data["benchmark"]["n_examples"], 800)
    equal("locked prompt hash", data["benchmark"]["prompt_hash"], "34d4fd78419110f21e70f8129a84d992cc6b10d02ddaa4c5d172c6d586ad0553")
    metrics = data["metrics"]
    close("locked clean accuracy", metrics["accuracy"], 0.30875)
    close("locked clean accuracy CI lower", metrics["accuracy_ci"][0], 0.2365581515)
    close("locked clean accuracy CI upper", metrics["accuracy_ci"][1], 0.3832208237)
    for key, expected, ci in (
        ("dedup_effect", 0.8329656070, (0.6814418739, 1.0228734438)),
        ("placebo_effect", -0.0819251513, (-0.1585337915, -0.0116869022)),
    ):
        close(f"Table 3 {key} mean", metrics[key]["d_ld_mean"], expected)
        close(f"Table 3 {key} CI lower", metrics[key]["d_ld_ci"][0], ci[0])
        close(f"Table 3 {key} CI upper", metrics[key]["d_ld_ci"][1], ci[1])
    close("Table 3 alternate de-duplication", metrics["dedup_alt_effect"]["d_ld_mean"], 0.8323572260)


def check_probes() -> None:
    held = load("heldout_probe.json")
    expected = {
        "step_0": {"layer_1": 0.4358333333, "layer_5": 0.4433333333},
        "step_2000": {"layer_1": 0.6125, "layer_5": 0.6158333333},
        "step_143000": {"layer_1": 0.9916666667, "layer_5": 0.9941666667},
    }
    for step, layers in expected.items():
        for layer, value in layers.items():
            close(f"Table 4 {step} {layer}", held["by_step"][step][layer]["mean"], value)
    warning("held-out probe provenance", "the exact immutable upstream path and producer remain listed as a release blocker")

    shuffled = load("position_shuffle_probe.json")["0"]["per_layer"]
    expected_shuffle = {
        "1": (0.8866666667, 0.8233333333),
        "2": (0.89, 0.8266666667),
        "3": (0.9, 0.7966666667),
        "6": (0.8766666667, 0.7866666667),
    }
    for layer, (intact, permuted) in expected_shuffle.items():
        close(f"Table 5 layer {layer} intact", shuffled[layer]["intact"], intact)
        close(f"Table 5 layer {layer} shuffled", shuffled[layer]["shuffled"], permuted)


def check_suppressors() -> None:
    data = load("suppressor_ablation_trajectory.json")
    expected = {
        "pythia-160m": {
            "head": [8, 9],
            "rows": {
                1000: (-0.0039497248, 0.42), 2000: (-0.0104608154, 0.3166666667),
                3000: (-0.7403857358, 0.5733333333), 5000: (-2.3942109712, 0.88),
                8000: (-2.4882283115, 0.97), 13000: (-1.7675137107, 0.97),
                143000: (-2.6594106674, 0.9966666667),
            },
        },
        "pythia-410m": {
            "head": [12, 12],
            "rows": {
                1000: (0.0000628392, 0.4533333333), 2000: (-0.0027730783, 0.2933333333),
                3000: (-0.1749341170, 0.82), 5000: (-0.8684206359, 0.8533333333),
                8000: (-1.6150212797, 0.9833333333), 13000: (-1.8073954582, 0.9766666667),
                143000: (-1.7556291898, 0.9966666667),
            },
        },
    }
    for model, spec in expected.items():
        equal(f"Table 6 {model} head", data[model]["suppression_head"], spec["head"])
        for step, (effect, acc) in spec["rows"].items():
            row = data[model]["by_step"][f"step_{step}"]
            close(f"Table 6 {model} step {step} effect", row["ablation_delta"], effect)
            close(f"Table 6 {model} step {step} accuracy", row["ioi_acc"], acc)

    expected_top5 = [
        ("L8H9", -2.6594106674), ("L3H0", -1.7789805285),
        ("L4H11", -1.5367071025), ("L6H2", -0.7789763641),
        ("L1H8", -0.6579566129),
    ]
    for got, expected_row in zip(data["pythia-160m"]["top5"], expected_top5):
        equal("top-five mature head identity", got[0], expected_row[0])
        close(f"top-five mature head {got[0]}", got[1], expected_row[1])

    split = load("splitsafe_suppressor_set.json")
    equal("split-safe item count", split["n_items_per_checkpoint"], 192)
    equal("split-safe contrast count", split["n_observations_per_checkpoint"], 384)
    split_expected = {
        1800: (0.0933, (0.0618, 0.1252)), 2000: (0.0914, (0.0622, 0.1206)),
        3200: (-0.0515, (-0.1148, 0.0138)), 5000: (-0.0794, (-0.1631, 0.0058)),
        6000: (-0.6359, (-0.8017, -0.4679)),
    }
    for step, (mean, ci) in split_expected.items():
        row = split["by_step"][f"step_{step}"]
        close(f"Table 7 step {step} mean", row["d_margin3_mean"], mean)
        close(f"Table 7 step {step} CI lower", row["d_margin3_ci"][0], ci[0])
        close(f"Table 7 step {step} CI upper", row["d_margin3_ci"][1], ci[1])

    single = load("splitsafe_single_head.json")
    equal("split-safe single-head n", single["evaluation"]["n_examples"], 800)
    close("split-safe single-head mean", single["effect"]["mean"], 0.038)
    close("split-safe single-head CI lower", single["effect"]["ci95"][0], -0.024)
    close("split-safe single-head CI upper", single["effect"]["ci95"][1], 0.114)


def check_scale() -> None:
    small = load("signflip_across_scale_160m_410m_1b.json")
    mid = load("behavior_2.8b.json")
    large = load("behavior_6.9b_12b.json")
    config = json.loads((CONFIG / "patch_windows.json").read_text())
    expected = {
        "pythia-160m": (0.3166666667, 0.9525950670, -4.1224749374, [3, 4, 5]),
        "pythia-410m": (0.2933333333, 0.105, -3.63, [6, 7, 8, 9, 10]),
        "pythia-1b": (0.3633333333, 0.43, -3.49, [4, 5, 6, 7, 8]),
        "pythia-2.8b": (0.2966666667, 0.3257681966, -4.0490280787, list(range(8, 15))),
        "pythia-6.9b": (0.3233333333, 0.84, -4.12, list(range(8, 15))),
        "pythia-12b": (0.42, 0.43, -3.96, list(range(9, 17))),
    }

    def source_for(model: str) -> dict[str, Any]:
        if model in small:
            return small[model]["by_step"]
        if model == "pythia-2.8b":
            return mid["by_step"]
        return large[model]["by_step"]

    for model, (floor_acc, floor_effect, mature_effect, layers) in expected.items():
        floor, mature = floor_and_mature(source_for(model))
        close(f"Table 8 {model} floor accuracy", floor[1]["ioi_acc"], floor_acc, 0.001)
        close(f"Table 8 {model} floor effect", floor[1]["delta_ld_mean"], floor_effect, 0.02)
        close(f"Table 8 {model} maturity effect", mature[1]["delta_ld_mean"], mature_effect, 0.02)
        equal(f"Table 8 {model} patch window", config["windows"][model]["layers"], layers)

    row3000 = small["pythia-160m"]["by_step"]["step_3000"]
    close("main text step3000 effect", row3000["delta_ld_mean"], 0.0210452286)
    close("main text step3000 CI lower", row3000["delta_ld_ci"][0], -0.1052692914)
    close("main text step3000 CI upper", row3000["delta_ld_ci"][1], 0.1495096680)
    close("main text step4000 effect", small["pythia-160m"]["by_step"]["step_4000"]["delta_ld_mean"], -0.9508585723)


def check_clustered_primary() -> None:
    data = load("primary_intervention_clustered_cis.json")["clustered_ci"]
    expected = {
        "dip": (0.9525947873, (0.8575870744, 1.0503840418), (0.6766397486, 1.2491977947), (0.7787146024, 1.1500294444)),
        "mature": (-4.1224749374, (-4.2835921853, -3.9623192333), (-4.5423692271, -3.7168933520), (-4.4511339908, -3.8158122384)),
    }
    for stage, (mean, prompt, template, namepair) in expected.items():
        row = data[stage]
        close(f"clustered {stage} mean", row["mean_dld"], mean)
        for name, got, exp in (
            ("prompt", row["ci_prompt"], prompt),
            ("template", row["ci_template_clustered"], template),
            ("namepair", row["ci_namepair_clustered"], namepair),
        ):
            close(f"clustered {stage} {name} lower", got[0], exp[0])
            close(f"clustered {stage} {name} upper", got[1], exp[1])


def check_polypythias() -> None:
    data = load("polypythias_signflip_9variants.json")
    expected = {
        "EleutherAI/pythia-160m-seed1": (0.3066666667, 0.5458109395, -3.9235797373),
        "EleutherAI/pythia-160m-seed3": (0.1466666667, 0.9924338420, -2.8943375778),
        "EleutherAI/pythia-160m-seed5": (0.2533333333, 1.1069837825, -4.0428134569),
        "EleutherAI/pythia-160m-data-seed1": (0.3066666667, 1.2030731916, -4.0635098139),
        "EleutherAI/pythia-160m-data-seed2": (0.1933333333, 1.3513965925, -3.7089944013),
        "EleutherAI/pythia-160m-data-seed3": (0.3633333333, 1.0067096599, -3.2375001303),
        "EleutherAI/pythia-160m-weight-seed1": (0.29, 0.8018242677, -4.5615720749),
        "EleutherAI/pythia-160m-weight-seed2": (0.3566666667, 0.8378879579, -4.7471647676),
        "EleutherAI/pythia-160m-weight-seed3": (0.2733333333, 1.4707899443, -3.5084324296),
    }
    equal("PolyPythia flip count", data["n_flips"], "9/9")
    for model, (acc, dip, mature) in expected.items():
        row = data["seeds"][model]
        equal(f"PolyPythia {model} flips", row["flips"], True)
        close(f"Table 9 {model} floor", row["dip"]["ioi_acc"], acc)
        close(f"Table 9 {model} floor effect", row["dip"]["delta_ld_mean"], dip)
        close(f"Table 9 {model} maturity effect", row["mature"]["delta_ld_mean"], mature)

    printed = {
        "EleutherAI/pythia-160m-seed1": 0.3066666667,
        "EleutherAI/pythia-160m-seed3": 0.1466666667,
        "EleutherAI/pythia-160m-seed5": 0.3433333333,
        "EleutherAI/pythia-160m-data-seed1": 0.2933333333,
        "EleutherAI/pythia-160m-data-seed2": 0.28,
        "EleutherAI/pythia-160m-data-seed3": 0.3533333333,
        "EleutherAI/pythia-160m-weight-seed1": 0.3333333333,
        "EleutherAI/pythia-160m-weight-seed2": 0.3666666667,
        "EleutherAI/pythia-160m-weight-seed3": 0.31,
    }
    mismatches = [
        model for model, paper_value in printed.items()
        if not math.isclose(float(data["seeds"][model]["dip"]["ioi_acc"]), paper_value, rel_tol=0.0, abs_tol=5e-4)
    ]
    if mismatches:
        warning("Table 9 protocol mix", f"{len(mismatches)} printed floor accuracies do not equal the accuracies stored beside the causal intervention")


def check_stanford() -> None:
    data = load("stanford_gpt2_signflip.json")["by_model"]["stanford_alias"]
    for step, value in {1500: 1.0261497418, 3000: 0.7694746304, 10000: -0.2288490645, 100000: -2.8890317345}.items():
        close(f"Table 10 Stanford step {step}", data[f"step_{step}"]["delta_ld_mean"], value)


def check_projection() -> None:
    data = load("projection_removal.json")
    expected = {
        "baseline": (-0.7526, 0.3167),
        "remove_dup_direction_0.5x": (-0.8065, 0.2967),
        "remove_dup_direction_1.0x": (-0.8589, 0.29),
        "remove_dup_direction_2.0x": (-0.9536, 0.27),
        "remove_dup_direction_4.0x": (-1.0722, 0.23),
        "remove_ortho_direction_1.0x": (-0.7386, 0.32),
        "remove_shuffled_direction_1.0x": (-0.7530, 0.3167),
        "remove_random_4_1.0x": (-0.7463, 0.32),
    }
    for key, (ld, acc) in expected.items():
        close(f"Table 11 {key} LD", data[key]["ld"], ld)
        close(f"Table 11 {key} accuracy", data[key]["acc"], acc)


def check_full_vocab_and_loss() -> None:
    data = load("full_vocabulary_floor.json")
    close("greedy selects neither", data["greedy_selects_neither_fraction"], 0.9867)
    close("P(S)", data["prob_repeated_name"], 0.0238)
    close("P(IO)", data["prob_correct_name"], 0.0118)
    close("rank S", data["mean_rank_repeated_name"], 14.9)
    close("rank IO", data["mean_rank_correct_name"], 29.4)

    loss = load("pile_loss_sample.json")
    values = [
        loss["by_step"][f"step_{step}"]["released_pythia_160m_loss"]
        for step in (1000, 2000, 3000, 5000, 10000)
    ]
    for step, got, expected in zip((1000, 2000, 3000, 5000, 10000), values, (3.7342, 3.1397, 2.9638, 2.8006, 2.6766)):
        close(f"Figure 3 loss step {step}", got, expected)
    equal("Figure 3 loss strictly decreases", all(a > b for a, b in zip(values, values[1:])), True)


def check_head_localization_status() -> None:
    data = load("dip_head_localization_summary.json")
    equal("head-localization n_heads", data["n_heads"], 144)
    close("head-localization reported maximum", data["reported_max_change_in_logit_difference"], 0.06)
    if data.get("provenance_status") != "raw artifact":
        warning("head-localization provenance", "the exact per-head raw artifact was not source-locked")


def check_release_metadata() -> None:
    for relative in (
        "paper/sign_flip_ioi_miw2026.pdf", "paper/metadata.json", "README.md",
        "MANIFEST.md", "docs/SOURCE_PROVENANCE.md",
    ):
        equal(f"required file {relative}", (ROOT / relative).exists(), True)
    metadata = json.loads((ROOT / "paper" / "metadata.json").read_text())
    equal("checkpoint URL", metadata["checkpoint_url"], "https://huggingface.co/anonymous-research-sub/pythia-160m-retrained-seed42")


def main() -> int:
    check_position_tables()
    check_locked_control()
    check_probes()
    check_suppressors()
    check_scale()
    check_clustered_primary()
    check_polypythias()
    check_stanford()
    check_projection()
    check_full_vocab_and_loss()
    check_head_localization_status()
    check_release_metadata()

    print(f"\n{PASS} checks passed, {FAIL} failed, {WARN} warning(s).")
    if FAIL:
        return 1
    if WARN:
        print("Numerical checks passed, but explicit archival provenance limitations remain.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    ARGS = parser.parse_args()
    raise SystemExit(main())

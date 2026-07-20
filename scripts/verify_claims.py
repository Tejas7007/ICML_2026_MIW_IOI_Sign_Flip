#!/usr/bin/env python3
"""Verify numerical claims and public files released with the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_ID = "teys7007/pythia-160m-seed42-dense"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"


class Checks:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.passed = 0
        self.failed = 0

    def equal(self, label: str, actual: Any, expected: Any) -> None:
        if actual == expected:
            self.passed += 1
            if self.verbose:
                print(f"PASS  {label}: {actual}")
        else:
            self.failed += 1
            print(f"FAIL  {label}: got {actual!r}, expected {expected!r}")

    def near(self, label: str, actual: float, expected: float, tol: float = 5e-4) -> None:
        if math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
            self.passed += 1
            if self.verbose:
                print(f"PASS  {label}: {actual}")
        else:
            self.failed += 1
            print(f"FAIL  {label}: got {actual}, expected {expected} +/- {tol}")


def load(name: str) -> Any:
    return json.loads((DATA / name).read_text())


def rows(by_step: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return sorted(
        (int(key.removeprefix("step_")), value)
        for key, value in by_step.items()
        if key.startswith("step_")
        and isinstance(value, dict)
        and "error" not in value
        and value.get("status") != "error"
    )


def verify_primary(c: Checks) -> None:
    battery = load("position_control_battery.json")
    expected = {
        "dip": {
            "real_S2": (0.9525950670, 0.8575870303, 1.0503846546),
            "random_S2": (0.9197551680, 0.8427029157, 0.9985248371),
            "random_S1": (2.1524222930, 2.0638164788, 2.2442854366),
            "random_IO": (-3.5360066255, -3.7313376780, -3.3416938135),
            "random_struct": (-0.8145451784, -0.8983983100, -0.7300148924),
        },
        "mature": {
            "real_S2": (-4.1229453532, -4.2840915780, -3.9628876651),
            "random_S2": (-4.1257953008, -4.3144544821, -3.9401483541),
            "random_S1": (2.0792321587, 1.8838812480, 2.2721939829),
            "random_IO": (-9.0626466433, -9.3049049020, -8.8197444307),
            "random_struct": (-4.4500414785, -4.6655968822, -4.2333017451),
        },
    }
    for stage, arms in expected.items():
        c.equal(f"position battery {stage} n", battery[stage]["n"], 300)
        for arm, (mean, lower, upper) in arms.items():
            result = battery[stage]["arms"][arm]
            c.near(f"{stage} {arm} mean", result["delta_ld_mean"], mean)
            c.near(f"{stage} {arm} lower", result["delta_ci"][0], lower)
            c.near(f"{stage} {arm} upper", result["delta_ci"][1], upper)

    clustered = load("primary_intervention_clustered_cis.json")["clustered_ci"]
    for stage, mean, template, name_pair in (
        ("dip", 0.9525947873, (0.6766397486, 1.2491977947), (0.7787146024, 1.1500294444)),
        ("mature", -4.1224749374, (-4.5423692271, -3.7168933520), (-4.4511339908, -3.8158122384)),
    ):
        result = clustered[stage]
        c.near(f"{stage} clustered mean", result["mean_dld"], mean)
        for label, key, interval in (
            ("template", "ci_template_clustered", template),
            ("name-pair", "ci_namepair_clustered", name_pair),
        ):
            c.near(f"{stage} {label} lower", result[key][0], interval[0])
            c.near(f"{stage} {label} upper", result[key][1], interval[1])

    trajectory = load("signflip_across_scale_160m_410m_1b.json")["pythia-160m"]["by_step"]
    c.near("step 3000 intervention mean", trajectory["step_3000"]["delta_ld_mean"], 0.02, 0.03)
    c.near("step 3000 intervention lower", trajectory["step_3000"]["delta_ld_ci"][0], -0.11, 0.02)
    c.near("step 3000 intervention upper", trajectory["step_3000"]["delta_ld_ci"][1], 0.15, 0.02)
    c.near("step 4000 intervention mean", trajectory["step_4000"]["delta_ld_mean"], -0.95, 0.03)


def verify_controls(c: Checks) -> None:
    locked = load("locked_input_control_160m.json")
    c.equal("locked examples", locked["benchmark"]["n_examples"], 800)
    c.equal(
        "locked prompt hash",
        locked["benchmark"]["prompt_hash"],
        "34d4fd78419110f21e70f8129a84d992cc6b10d02ddaa4c5d172c6d586ad0553",
    )
    metrics = locked["metrics"]
    for label, actual, expected in (
        ("locked accuracy", metrics["accuracy"], 0.30875),
        ("locked accuracy lower", metrics["accuracy_ci"][0], 0.2365581515),
        ("locked accuracy upper", metrics["accuracy_ci"][1], 0.3832208237),
        ("de-duplication mean", metrics["dedup_effect"]["d_ld_mean"], 0.8329656070),
        ("de-duplication lower", metrics["dedup_effect"]["d_ld_ci"][0], 0.6814418739),
        ("de-duplication upper", metrics["dedup_effect"]["d_ld_ci"][1], 1.0228734438),
        ("alternate de-duplication", metrics["dedup_alt_effect"]["d_ld_mean"], 0.8323572260),
        ("placebo mean", metrics["placebo_effect"]["d_ld_mean"], -0.0819251513),
        ("placebo lower", metrics["placebo_effect"]["d_ld_ci"][0], -0.1585337915),
        ("placebo upper", metrics["placebo_effect"]["d_ld_ci"][1], -0.0116869022),
    ):
        c.near(label, actual, expected)

    floor = load("full_vocabulary_floor.json")
    for label, key, expected in (
        ("greedy selects neither", "greedy_selects_neither_fraction", 0.9867),
        ("repeated-name probability", "prob_repeated_name", 0.0238),
        ("correct-name probability", "prob_correct_name", 0.0118),
        ("repeated-name rank", "mean_rank_repeated_name", 14.9),
        ("correct-name rank", "mean_rank_correct_name", 29.4),
    ):
        c.near(label, floor[key], expected)


def verify_heads_and_probes(c: Checks) -> None:
    sweep = load("head_zero_ablation_step1000.json")
    c.equal("head sweep checkpoint", sweep["checkpoint"], 1000)
    c.equal("head sweep template selection", sweep["protocol"]["template_selection"], "ALL_TEMPLATES[:8]")
    c.equal("head sweep examples", sweep["protocol"]["n_examples"], 160)
    c.equal("head sweep head count", sweep["protocol"]["n_heads_tested"], 144)
    c.near("head sweep largest absolute change", sweep["extrema"]["largest_absolute_change"]["absolute_delta_logit_difference"], 0.0704)

    heldout = load("heldout_probe.json")
    c.equal("held-out probe model", heldout["model"], MODEL_ID)
    for step, layer, expected in (
        (0, 1, 0.4358333333), (0, 5, 0.4433333333),
        (2000, 1, 0.6125), (2000, 5, 0.6158333333),
        (143000, 1, 0.9916666667), (143000, 5, 0.9941666667),
    ):
        c.near(f"held-out probe {step} L{layer}", heldout["by_step"][f"step_{step}"][f"layer_{layer}"]["mean"], expected)

    shuffled = load("position_shuffle_probe.json")["0"]["per_layer"]
    for layer, intact, permuted in (
        ("1", 0.8866666667, 0.8233333333),
        ("2", 0.89, 0.8266666667),
        ("3", 0.9, 0.7966666667),
        ("6", 0.8766666667, 0.7866666667),
    ):
        c.near(f"shuffle {layer} intact", shuffled[layer]["intact"], intact)
        c.near(f"shuffle {layer} permuted", shuffled[layer]["shuffled"], permuted)

    projection = load("projection_removal.json")
    for key, ld, accuracy in (
        ("baseline", -0.7526, 0.3167),
        ("remove_dup_direction_0.5x", -0.8065, 0.2967),
        ("remove_dup_direction_1.0x", -0.8589, 0.29),
        ("remove_dup_direction_2.0x", -0.9536, 0.27),
        ("remove_dup_direction_4.0x", -1.0722, 0.23),
        ("remove_ortho_direction_1.0x", -0.7386, 0.32),
        ("remove_shuffled_direction_1.0x", -0.7530, 0.3167),
        ("remove_random_4_1.0x", -0.7463, 0.32),
    ):
        c.near(f"projection {key} LD", projection[key]["ld"], ld)
        c.near(f"projection {key} accuracy", projection[key]["acc"], accuracy)


def verify_suppressors(c: Checks) -> None:
    trajectories = load("suppressor_ablation_trajectory.json")
    expected = {
        "pythia-160m": ([8, 9], {1000: -0.0039497248, 2000: -0.0104608154, 3000: -0.7403857358, 5000: -2.3942109712, 8000: -2.4882283115, 13000: -1.7675137107, 143000: -2.6594106674}),
        "pythia-410m": ([12, 12], {1000: 0.0000628392, 2000: -0.0027730783, 3000: -0.1749341170, 5000: -0.8684206359, 8000: -1.6150212797, 13000: -1.8073954582, 143000: -1.7556291898}),
    }
    for model, (head, by_step) in expected.items():
        c.equal(f"{model} suppressor head", trajectories[model]["suppression_head"], head)
        for step, value in by_step.items():
            c.near(f"{model} suppressor {step}", trajectories[model]["by_step"][f"step_{step}"]["ablation_delta"], value)

    split_set = load("splitsafe_suppressor_set.json")
    c.equal("split-safe model", split_set["model"], MODEL_ID)
    c.equal("split-safe items", split_set["n_items_per_checkpoint"], 192)
    c.equal("split-safe contrasts", split_set["n_observations_per_checkpoint"], 384)
    for step, mean, lower, upper in (
        (1800, 0.0933, 0.0618, 0.1252),
        (2000, 0.0914, 0.0622, 0.1206),
        (3200, -0.0515, -0.1148, 0.0138),
        (5000, -0.0794, -0.1631, 0.0058),
        (6000, -0.6359, -0.8017, -0.4679),
    ):
        result = split_set["by_step"][f"step_{step}"]
        c.near(f"split-safe {step} mean", result["d_margin3_mean"], mean)
        c.near(f"split-safe {step} lower", result["d_margin3_ci"][0], lower)
        c.near(f"split-safe {step} upper", result["d_margin3_ci"][1], upper)

    characterization = load("suppressor_characterization.json")
    pythia = characterization["pythia-160m"]
    c.equal("Pythia suppressor head", pythia["head"], "L8H9")
    for step, attention, projection in (
        (1000, 0.0226, 0.0020),
        (3000, 0.8685, 0.9859),
        (143000, 0.9216, 5.6691),
    ):
        result = pythia["by_step"][f"step_{step}"]
        c.near(f"Pythia attention {step}", result["attention_to_S2"], attention)
        c.near(f"Pythia projection {step}", result["IO_minus_S_projection"], projection)

    stanford = characterization["stanford-gpt2-small"]
    c.equal("Stanford suppressor head", stanford["head"], "L10H10")
    c.near("Stanford suppressor ablation", stanford["zero_ablation_accuracy_change"], -0.09)
    c.near("Stanford attention", stanford["attention_to_S2"], 0.589)
    c.near("Stanford projection", stanford["IO_minus_S_projection"], 6.2399)


def verify_scale_and_replications(c: Checks) -> None:
    small = load("signflip_across_scale_160m_410m_1b.json")
    middle = load("behavior_2.8b.json")
    large = load("behavior_6.9b_12b.json")
    windows = json.loads((ROOT / "config" / "patch_windows.json").read_text())["windows"]
    expected = {
        "pythia-160m": (0.3166666667, 0.9525950670, -4.1229453532, [3, 4, 5]),
        "pythia-410m": (0.2933333333, 0.1049352249, -3.63, [6, 7, 8, 9, 10]),
        "pythia-1b": (0.3633333333, 0.43, -3.49, [4, 5, 6, 7, 8]),
        "pythia-2.8b": (0.2966666667, 0.3257681966, -4.0490280787, list(range(8, 15))),
        "pythia-6.9b": (0.3233333333, 0.8366666667, -4.1184114583, list(range(8, 15))),
        "pythia-12b": (0.42, 0.4340625, -3.95515625, list(range(9, 17))),
    }
    for model, (floor_accuracy, floor_effect, final_effect, layers) in expected.items():
        by_step = small[model]["by_step"] if model in small else middle["by_step"] if model == "pythia-2.8b" else large[model]["by_step"]
        trajectory = rows(by_step)
        floor = min(trajectory, key=lambda item: float(item[1]["ioi_acc"]))[1]
        final = trajectory[-1][1]
        c.near(f"{model} floor accuracy", floor["ioi_acc"], floor_accuracy, 0.001)
        c.near(f"{model} floor effect", floor["delta_ld_mean"], floor_effect, 0.02)
        c.near(f"{model} final effect", final["delta_ld_mean"], final_effect, 0.02)
        c.equal(f"{model} patch window", windows[model]["layers"], layers)

    effects = load("polypythias_signflip_9variants.json")
    floors = load("polypythias_floors.json")
    c.equal("PolyPythia reversal count", effects["n_flips"], "9/9")
    c.equal("PolyPythia floor templates", floors["protocol"]["template_selection"], "ALL_TEMPLATES[:15]")
    c.equal("PolyPythia floor examples", floors["protocol"]["n_examples_per_checkpoint"], 300)

    stanford = load("stanford_gpt2_signflip.json")["by_model"]["stanford_alias"]
    for step, expected_effect in ((1500, 1.0261497418), (3000, 0.7694746304), (10000, -0.2288490645), (100000, -2.8890317345)):
        c.near(f"Stanford effect {step}", stanford[f"step_{step}"]["delta_ld_mean"], expected_effect)


def verify_loss(c: Checks) -> None:
    loss = load("pile_loss_sample.json")
    values = []
    for step, expected in ((1000, 3.7342), (2000, 3.1397), (3000, 2.9638), (5000, 2.8006), (10000, 2.6766)):
        actual = loss["by_step"][f"step_{step}"]["released_pythia_160m_loss"]
        values.append(actual)
        c.near(f"Pile loss {step}", actual, expected)
    c.equal("sampled Pile loss is monotone", all(a > b for a, b in zip(values, values[1:])), True)


def verify_package(c: Checks) -> None:
    required = [
        "README.md", "CITATION.cff", "LICENSE", "requirements.txt",
        "paper/sign_flip_ioi_miw2026.pdf", "config/patch_windows.json",
        "scripts/verify_claims.py", "scripts/make_figures.py",
        "scripts/reproduce_intervention.py", "scripts/reproduce_head_ablation.py",
        "scripts/reproduce_position_controls.py", "scripts/reproduce_input_control.py",
        "scripts/reproduce_probes.py", "scripts/reproduce_suppressor_trajectory.py",
        "scripts/reproduce_splitsafe_suppressors.py", "scripts/reproduce_projection_removal.py",
        "scripts/reproduce_replications.py", "scripts/reproduce_loss_and_vocabulary.py",
        "scripts/lib/ioi_dataset.py", "scripts/lib/locked_ioi_benchmark.py",
        "data/polypythias_floors.json", "data/polypythias_signflip_9variants.json",
        "data/head_zero_ablation_step1000.json", "data/suppressor_characterization.json",
        "data/splitsafe_suppressor_set.json", "assets/sign_flip.svg",
        "assets/three_timelines.svg", "assets/scale_replication.svg",
    ]
    figures = ((1, "below_chance_dip"), (2, "sign_flip"), (3, "loss_vs_accuracy"))
    required.extend(f"figures/fig{number}_{name}.{ext}" for number, name in figures for ext in ("pdf", "png"))
    for relative in required:
        c.equal(f"required file {relative}", (ROOT / relative).is_file(), True)

    c.equal("unsupported single-head artifact absent", (DATA / "splitsafe_single_head.json").exists(), False)
    c.equal("standalone reproducibility document absent", (ROOT / "REPRODUCIBILITY.md").exists(), False)

    for relative in ["paper/sign_flip_ioi_miw2026.pdf"] + [f"figures/fig{number}_{name}.pdf" for number, name in figures]:
        c.equal(f"PDF signature {relative}", (ROOT / relative).read_bytes().startswith(b"%PDF-"), True)
    for relative in [f"figures/fig{number}_{name}.png" for number, name in figures]:
        c.equal(f"PNG signature {relative}", (ROOT / relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), True)
    for relative in ("assets/sign_flip.svg", "assets/three_timelines.svg", "assets/scale_replication.svg"):
        try:
            ET.parse(ROOT / relative)
            valid = True
        except ET.ParseError:
            valid = False
        c.equal(f"SVG XML {relative}", valid, True)

    for path in DATA.glob("*.json"):
        try:
            json.loads(path.read_text())
            valid = True
        except json.JSONDecodeError:
            valid = False
        c.equal(f"JSON {path.name}", valid, True)

    readme = (ROOT / "README.md").read_text()
    c.equal("README model URL", MODEL_URL in readme, True)
    c.equal("README reproduction scripts", "Reproduction scripts" in readme, True)

    patterns = (
        ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
        ("GitHub token", re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}\b")),
        ("secret key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
        ("absolute path", re.compile(r"(?:/Users/|/workspace/|/home/)[^\s\"']+")),
    )
    text_extensions = {".py", ".md", ".json", ".cff", ".txt", ".svg", ".yml", ".yaml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix not in text_extensions and path.name not in {"LICENSE", ".gitignore"}:
            continue
        text = path.read_text(errors="replace")
        for label, pattern in patterns:
            c.equal(f"{path.relative_to(ROOT)} contains no {label}", pattern.search(text) is None, True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    checks = Checks(parser.parse_args().verbose)
    for verify in (
        verify_primary,
        verify_controls,
        verify_heads_and_probes,
        verify_suppressors,
        verify_scale_and_replications,
        verify_loss,
        verify_package,
    ):
        verify(checks)
    print(f"\n{checks.passed} checks passed, {checks.failed} failed.")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

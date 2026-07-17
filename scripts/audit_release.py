#!/usr/bin/env python3
"""Static release audit for secrets, stale identifiers, and missing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".txt", ".cff", ".svg"}
EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "env"}
EXPECTED_REPO_URL = "https://github.com/Tejas7007/ICML_2026_MIW_IOI_Sign_Flip"
EXPECTED_HF_URL = "https://huggingface.co/anonymous-research-sub/pythia-160m-retrained-seed42"

FORBIDDEN = {
    "hard-coded Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub personal access token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "absolute workspace path": re.compile(r"(?<![\w.-])/(?:workspace|home|Users)/[^\s\"']+"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\"),
    "stale checkpoint identifier": re.compile(r"teys7007/pythia-160m-seed42-dense"),
}

BANNED_RELEASE_FILES = {
    Path("data/polypythias_floors.json"),
    Path("data/heldout_probe_and_position.json"),
    Path("data/loss_and_head_sweep.json"),
}


def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            yield path


def scan_text() -> list[str]:
    errors: list[str] = []
    for path in iter_text_files():
        text = path.read_text(errors="replace")
        relative = path.relative_to(ROOT)

        # This file contains the patterns it searches for, so it is excluded.
        if relative == Path("scripts/audit_release.py"):
            continue

        for label, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative}:{line}: {label}: {match.group(0)!r}")

    return errors


def check_json() -> list[str]:
    errors: list[str] = []
    for path in (ROOT / "data").glob("*.json"):
        try:
            json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: malformed JSON: {exc}")
    return errors


def check_structure() -> list[str]:
    errors: list[str] = []
    required = {
        Path("README.md"),
        Path("MANIFEST.md"),
        Path("docs/REPRODUCIBILITY.md"),
        Path("docs/SOURCE_PROVENANCE.md"),
        Path("scripts/verify_claims.py"),
        Path("scripts/make_figures.py"),
        Path("scripts/reproduce_intervention.py"),
        Path("paper/sign_flip_ioi_miw2026.pdf"),
        Path("paper/metadata.json"),
        Path("release_status.json"),
        Path("config/patch_windows.json"),
        Path("data/primary_intervention_clustered_cis.json"),
        Path("data/heldout_probe.json"),
        Path("data/pile_loss_sample.json"),
        Path("data/splitsafe_single_head.json"),
    }
    for relative in sorted(required):
        if not (ROOT / relative).exists():
            errors.append(f"missing required release file: {relative}")

    for relative in BANNED_RELEASE_FILES:
        if (ROOT / relative).exists():
            errors.append(
                "superseded or scope-contaminated file must not ship in the "
                f"paper-facing data directory: {relative}"
            )

    readme = (ROOT / "README.md").read_text()
    if EXPECTED_HF_URL not in readme:
        errors.append("README does not contain the canonical retrained-checkpoint URL")
    if EXPECTED_REPO_URL not in readme:
        errors.append("README does not contain the canonical repository URL")

    return errors


def check_release_status(*, allow_blocked: bool) -> list[str]:
    errors: list[str] = []
    status_path = ROOT / "release_status.json"
    if not status_path.exists():
        return ["release_status.json is missing"]
    status = json.loads(status_path.read_text())
    blockers = status.get("blockers", [])
    if status.get("status") != "ready" or blockers:
        ids = ", ".join(item.get("id", "unknown") for item in blockers)
        if allow_blocked:
            print(f"Release-readiness blockers are recorded: {ids}")
            return errors
        errors.append(f"release status is not ready; unresolved blockers: {ids}")
        return errors

    path = ROOT / "paper" / "sign_flip_ioi_miw2026.pdf"
    if not path.exists():
        return ["camera-ready PDF is missing"]
    raw = path.read_bytes()
    if EXPECTED_REPO_URL.encode() not in raw:
        errors.append("camera-ready PDF does not contain the repository URI")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit camera-ready release hygiene and readiness.")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Pass hygiene checks while release_status.json still records explicit blockers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = scan_text() + check_json() + check_structure() + check_release_status(
        allow_blocked=args.allow_blocked
    )
    if errors:
        print("Release audit failed:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Release audit passed.")
    print(
        "No credential-like strings, stale checkpoint identifiers, forbidden paths, "
        "malformed JSON, or missing release artifacts were found."
    )
    if args.allow_blocked:
        print("Explicit release-readiness blockers remain recorded in release_status.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

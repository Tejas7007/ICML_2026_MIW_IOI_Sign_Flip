# Reproducibility guide

## Fast verification without model inference

```bash
python scripts/verify_claims.py --verbose
python scripts/audit_release.py
python scripts/make_figures.py
```

These commands require only the committed JSON files and a standard Python environment. They verify source-locked values, scan the release for accidental secrets or stale identifiers, and redraw the three figures.

## Core inference reproduction

The central residual-stream intervention can be recomputed with:

```bash
python scripts/reproduce_intervention.py \
  --model pythia-160m \
  --step 2000 \
  --output results/reproduced_pythia-160m_step2000.json
```

The runner downloads the public deduplicated Pythia checkpoint. No Hugging Face token is required. It materializes the exact historical first-ten-BABA-family protocol with symmetric name-role swaps and fails rather than silently skipping malformed prompts.

Large models require substantial accelerator memory. The paper-facing JSON artifacts are provided so that verification and figure generation do not require repeating these expensive runs.

## Environment

The tested release environment is recorded in `requirements-lock.txt`. `requirements.txt` gives the minimum supported ranges. The reference inference runner was designed for Python 3.10 or 3.11, PyTorch with CUDA, Transformers, and TransformerLens. Verification and plotting are CPU-only.

## Data and uncertainty

The primary 160M effect includes prompt-level, template-clustered, and name-pair-clustered intervals. The locked input control stores per-example outputs and crossed cluster-aware intervals. Other historical cross-scale cells are released with the uncertainty estimates preserved by their producing analyses.

The 410M floor intervention is intentionally treated as a point estimate in the paper. No cluster-aware significance claim is made for that cell.

## Historical loss sample

The loss analysis evaluated the first 50 qualifying 512-token sequences after skipping an initial block of the `monology/pile-uncopyrighted` training stream. The script saved aggregate loss values but not text identifiers or token hashes. The exact historical sample therefore cannot be reconstructed byte-for-byte. This limitation is stated in the paper and released artifact.

## Camera-ready PDF

The final corrected PDF is pending. The release audit will require a clickable repository URI and a recorded hash before the release can pass.

<div align="center">

<img src="assets/sign_flip.svg" alt="A training-time sign flip in IOI circuit formation" width="820"/>

# A Training-Time Sign Flip in IOI Circuit Formation

Mechanistic Interpretability Workshop at ICML 2026

[![Paper](https://img.shields.io/badge/paper-camera--ready_PDF-b31b1b.svg)](paper/sign_flip_ioi_miw2026.pdf)
[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97%20checkpoint-retrained_Pythia--160M-yellow.svg)](https://huggingface.co/anonymous-research-sub/pythia-160m-retrained-seed42)
[![Verification](https://img.shields.io/github/actions/workflow/status/Tejas7007/ICML_2026_MIW_IOI_Sign_Flip/verify.yml?branch=main&label=artifact%20verification)](https://github.com/Tejas7007/ICML_2026_MIW_IOI_Sign_Flip/actions/workflows/verify.yml)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20CC--BY--4.0-3fb950.svg)](LICENSE)

</div>

---

## Release-audit status

This branch is a hardened camera-ready evidence package, but it is intentionally **not marked final** while the source audit items in [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md) remain open. The numerical artifacts are preserved rather than silently altered to make an unresolved paper sentence pass verification.

## What this repository establishes

A language model can depend on the same position-level internal state in opposite ways at different stages of training.

We study indirect object identification (IOI). In a prompt such as

> When Mary and John went to the store, John gave a drink to ___

the correct continuation is **Mary**, the name that appears once, rather than **John**, the repeated name. During early training, the model passes through a transient window in which its two-way IO-versus-repeated-name accuracy falls below 50 percent. At the Pythia-160M floor, replacing the internal state at the repeated name's second occurrence, **S2**, with a matched non-repeating control improves the logit difference by **+0.95**. At the final checkpoint, the same intervention family changes it by **-4.12**.

The positive-to-negative reversal replicates across all six tested Pythia scales, all nine PolyPythias training variants, and Stanford GPT-2 Small. The paper separately tracks linear accessibility, position-level causal dependence, and suppressor function. These measurements follow different training trajectories and should not be treated as one event.

<div align="center">
<img src="assets/three_timelines.svg" alt="Accessibility, causal dependence, and suppressor function follow different training trajectories" width="820"/>
</div>

## Headline evidence

| Model | Floor accuracy | Intervention at floor | Intervention at maturity | Residual patch window |
|:--|--:|--:|--:|:--|
| Pythia-160M | 31.7% | +0.95 | -4.12 | layers 3 to 5 |
| Pythia-410M | 29.3% | +0.11 | -3.63 | layers 6 to 10 |
| Pythia-1B | 36.3% | +0.43 | -3.49 | layers 4 to 8 |
| Pythia-2.8B | 29.7% | +0.33 | -4.05 | layers 8 to 14 |
| Pythia-6.9B | 32.3% | +0.84 | -4.12 | layers 8 to 14 |
| Pythia-12B | 42.0% | +0.43 | -3.96 | layers 9 to 16 |

For the primary 160M result, the sign survives dependence-aware resampling. The template-clustered interval is **[+0.68, +1.25]** at the floor and **[-4.54, -3.72]** at maturity. The non-160M scale results are released as prompt-level estimates; the paper does not claim that a cluster-aware interval excludes zero for the small 410M floor effect.

<div align="center">
<img src="assets/scale_replication.svg" alt="Matched-S2 intervention effects at the floor and at maturity across six Pythia scales" width="820"/>
</div>

## Paper figures

<div align="center">
<img src="figures/fig2_sign_flip.png" alt="Accuracy, matched-S2 intervention, and suppressor mean-ablation across Pythia-160M training" width="94%"/>
</div>

<table>
<tr>
<td width="50%" valign="top" align="center"><img src="figures/fig1_below_chance_dip.png" alt="Below-chance IOI window across six Pythia scales" width="86%"/></td>
<td width="50%" valign="top" align="center"><img src="figures/fig3_loss_vs_accuracy.png" alt="Pile evaluation-sample loss and IOI accuracy across sampled checkpoints" width="66%"/></td>
</tr>
</table>

The figures are generated only from the committed JSON artifacts. No model download is required to verify the source-locked numbers or redraw the plots.

## Three reproducibility levels

### 1. Verify every source-locked value

```bash
python scripts/verify_claims.py --verbose
```

The verifier checks the released numerical evidence for Tables 1 through 11, main-text quantities, patch-window configuration, result schemas, and figure inputs. The release audit separately blocks publication when a paper-facing claim is not source-locked or does not match its canonical artifact.

### 2. Regenerate the paper figures

```bash
python scripts/make_figures.py
```

This writes both PDF and PNG versions of Figures 1 through 3 from the committed artifacts.

### 3. Re-run the central residual-stream intervention

This requires a CUDA-capable environment and downloads public Pythia checkpoints.

```bash
pip install -r requirements.txt
python scripts/reproduce_intervention.py \
  --model pythia-160m \
  --step 2000 \
  --output results/reproduced_pythia-160m_step2000.json
```

The reference runner implements the exact historical ten-template, 300-prompt protocol used for the core cross-scale result. The producer uses the first ten BABA-style IOI template families with symmetric name-role swaps; it is not a five-ABBA/five-BABA design. It uses released deduplicated Pythia checkpoints, checkpoint-local matched controls, the model-specific patch window from `config/patch_windows.json`, 10,000 bootstrap draws, and no hard-coded authentication token.

The independently trained model used in the split-safe analyses is available at [`anonymous-research-sub/pythia-160m-retrained-seed42`](https://huggingface.co/anonymous-research-sub/pythia-160m-retrained-seed42).

## Repository map

```text
.
|-- paper/
|   `-- sign_flip_ioi_miw2026.pdf
|-- figures/
|   |-- fig1_below_chance_dip.{pdf,png}
|   |-- fig2_sign_flip.{pdf,png}
|   `-- fig3_loss_vs_accuracy.{pdf,png}
|-- data/
|   `-- paper-facing result artifacts
|-- config/
|   `-- patch_windows.json
|-- scripts/
|   |-- verify_claims.py
|   |-- make_figures.py
|   |-- reproduce_intervention.py
|   |-- audit_release.py
|   `-- lib/ioi_dataset.py
|-- docs/
|   |-- REPRODUCIBILITY.md
|   |-- SOURCE_PROVENANCE.md
|   |-- PRODUCER_INDEX.md
|   `-- RELEASE_AUDIT.md
|-- MANIFEST.md
|-- release_status.json
|-- CITATION.cff
`-- LICENSE
```

## Evidence policy

This repository is a camera-ready evidence package, not a dump of every exploratory experiment performed during the project.

- Each source-locked paper claim maps to one canonical released artifact in [`MANIFEST.md`](MANIFEST.md); unresolved claims are listed explicitly rather than forced to pass.
- Exact copies and compact derivatives are distinguished in [`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md), and original producer scripts are indexed in [`docs/PRODUCER_INDEX.md`](docs/PRODUCER_INDEX.md).
- Historical files that used different prompt protocols are excluded rather than presented as independent replications.
- Later mechanism experiments about copying heads, MLP localization, equality tests, routing, and final-logit closure are outside this workshop paper's scope.
- The fixed Pile training-stream sample cannot be reconstructed exactly because sampled text identities were not stored. The procedure and aggregate values are released, and the paper states this limitation.
- The 410M floor intervention is retained as a cross-scale point estimate. No dependence-aware interval is claimed for that cell.

## Automated release audit

```bash
python scripts/audit_release.py
```

The audit checks credential-like strings, hard-coded tokens, absolute workstation paths, stale model identifiers, malformed JSON, missing artifacts, superseded files, final PDF metadata, and unresolved release blockers. The same checks run in GitHub Actions.

## Citation

```bibtex
@inproceedings{dahiya2026signflip,
  title     = {A Training-Time Sign Flip in {IOI} Circuit Formation},
  author    = {Dahiya, Tejas and Blondin, Cole},
  booktitle = {Mechanistic Interpretability Workshop at the
               43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## License

Code and configuration are released under the MIT License. Released data, figures, and documentation are provided under CC BY 4.0. Model checkpoints remain under the licenses selected by their respective publishers.

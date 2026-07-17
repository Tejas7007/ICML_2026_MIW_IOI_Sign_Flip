<div align="center">

<img src="assets/sign_flip.svg" alt="A training-time sign flip in IOI circuit formation" width="820"/>

# A Training-Time Sign Flip in IOI Circuit Formation

Mechanistic Interpretability Workshop at ICML 2026

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/sign_flip_ioi_miw2026.pdf)
[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97%20checkpoint-pythia--160m--seed42--dense-yellow.svg)](https://huggingface.co/teys7007/pythia-160m-seed42-dense)
[![Data](https://img.shields.io/badge/data-15%20result%20files-0969da.svg)](data/)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20CC--BY--4.0-3fb950.svg)](LICENSE)

</div>

---

## Overview

Circuit analyses usually describe a trained model at a single checkpoint, yet the
role a component plays can differ earlier in training. We show a case where it
reverses. An internal state that is harmful to a behavior early in training
becomes useful for it later.

We study indirect object identification (IOI). The task is to complete a sentence
such as *"When Mary and John went to the store, John gave a drink to ___"* with the
name that occurred once (Mary) rather than the repeated name (John). During
training, accuracy on this two-way choice falls below chance in a transient
window and then recovers, while the language-modeling loss keeps decreasing.
Replacing the internal state at the repeated name's second occurrence (S2) with a
matched non-repeating one helps the choice during the window and harms it after
recovery. The model's dependence on that state has flipped sign.

## The central result

Three properties are usually assumed to appear together during training: whether
a signal is readable, whether a position is causally used, and whether a working
component has formed. In IOI they resolve at different training steps.

<div align="center">
<img src="assets/three_timelines.svg" alt="The three properties resolve at different training steps" width="820"/>
</div>

The repeated-name signal becomes linearly readable by a probe on one timeline. The
causal role of the S2 state flips from positive to negative on a second timeline. A
suppressor head that writes against the repeated name reaches a detectable causal
effect on a third, later timeline. A description that is faithful at maturity can
therefore be wrong about an earlier checkpoint.

## Three findings

1. **A below-chance window.** IOI accuracy dips below the 50 percent two-way
   baseline and recovers, across six model scales, nine PolyPythias training
   variants, and a second model family.
2. **The causal role of S2 reverses.** The matched-S2 intervention is significantly
   positive at the floor and significantly negative at maturity. This
   positive-to-negative reversal replicates across all six Pythia scales, all nine
   PolyPythias variants, and Stanford GPT-2 Small.
3. **Three properties on three timelines.** The suppressor head develops its causal
   effect during recovery, after the harmful S2 dependence is already present, and
   the direction a probe uses to read the repeated name is not the direction whose
   removal changes behavior.

## The reversal replicates at every scale

<div align="center">
<img src="assets/scale_replication.svg" alt="The sign flip replicates across six model scales" width="820"/>
</div>

Matched-S2 intervention effect at the floor and at maturity. A positive value moves
the model toward the correct name; a negative value moves it toward the repeated
name.

| Model | Floor accuracy | Window effect | Maturity effect | Patch window |
|:--|--:|--:|--:|:--|
| Pythia-160M | 31.7% | +0.95 | -4.12 | layers 3 to 5 |
| Pythia-410M | 29.3% | +0.11 | -3.63 | layers 6 to 10 |
| Pythia-1B | 36.3% | +0.43 | -3.49 | layers 4 to 8 |
| Pythia-2.8B | 29.7% | +0.33 | -4.05 | layers 8 to 14 |
| Pythia-6.9B | 32.3% | +0.84 | -4.12 | layers 8 to 14 |
| Pythia-12B | 42.0% | +0.43 | -3.96 | layers 9 to 16 |

All nine PolyPythias training variants reverse sign. These vary the random seed,
the data order, and the initialization, with floors from 14.7 percent to 36.7
percent, window effects up to +1.47, and maturity effects down to -4.75. Stanford
GPT-2 Small moves from +1.03 at step 1,500 to -2.89 at step 100,000.

Every number, with its data file and exact field path, is listed in
[`MANIFEST.md`](MANIFEST.md).

## Figures from the paper

<div align="center">
<img src="figures/fig2_sign_flip.png" alt="Figure 2" width="94%"/>
<br>
<sub><b>Figure 2.</b> Three measurements on Pythia-160M across training: IOI accuracy, the matched-S2 intervention with a bootstrap band, and suppressor mean-ablation. The intervention helps at the floor and harms at maturity, and the suppressor develops its effect during recovery.</sub>
</div>

<br>

<table>
<tr>
<td width="50%" valign="top" align="center">
<img src="figures/fig1_below_chance_dip.png" alt="Figure 1" width="86%"/>
<br><sub><b>Figure 1.</b> The below-chance window across six model scales.</sub>
</td>
<td width="50%" valign="top" align="center">
<img src="figures/fig3_loss_vs_accuracy.png" alt="Figure 3" width="66%"/>
<br><sub><b>Figure 3.</b> Pile evaluation-sample loss and IOI accuracy.</sub>
</td>
</tr>
</table>

These are the figures printed in the paper. Run `python scripts/make_figures.py`
to regenerate them from the released data files.

## Reproduce it

```bash
git clone https://github.com/Tejas7007/ICML_2026_MIW_IOI_Sign_Flip.git
cd ICML_2026_MIW_IOI_Sign_Flip
pip install -r requirements.txt

# Check every reported number against the released data files
python scripts/verify_claims.py --verbose

# Redraw the three paper figures from the same files
python scripts/make_figures.py
```

To recompute the core measurement from the model itself, which requires `torch`
and `transformer_lens`:

```bash
python scripts/reproduce_intervention.py --model pythia-160m --step 2000     # window: helps
python scripts/reproduce_intervention.py --model pythia-160m --step 143000   # mature: harms
```

The retrained Pythia-160M checkpoint (seed 42) used for the split-safe analyses is
on the Hugging Face Hub at
[`teys7007/pythia-160m-seed42-dense`](https://huggingface.co/teys7007/pythia-160m-seed42-dense).
All other checkpoints are the public [Pythia](https://huggingface.co/EleutherAI)
suite and [Stanford GPT-2](https://huggingface.co/stanford-crfm).

## Repository layout

```
.
|-- paper/          Camera-ready PDF
|-- figures/        The three figures, exactly as they appear in the paper
|-- data/           One result file per table and figure (see MANIFEST.md)
|-- scripts/
|   |-- verify_claims.py            Checks every reported number against data/
|   |-- make_figures.py             Redraws Figures 1 to 3 from data/
|   |-- reproduce_intervention.py   Reference implementation of the S2 intervention
|-- config/
|   |-- patch_windows.json          Per-model patch windows and provenance
|-- assets/         Animated figures used in this README
|-- MANIFEST.md     Claim to data file to field path, for every table and figure
```

## What the data files contain

Each file in [`data/`](data/) is the result artifact behind one part of the paper:
the sign flip across scale, the nine PolyPythias variants, Stanford GPT-2, the
position control battery, the locked 800-prompt input control (with a frozen prompt
hash and cluster-aware intervals), the held-out and position-shuffle probes, the
suppressor mean-ablation trajectory, the split-safe frozen suppressor-set
trajectory, projection removal, the full-vocabulary floor behavior, and the Pile
evaluation-sample loss.

Two quantities are reported as point estimates or as non-reconstructable by design.
The Pythia-410M window effect is a cross-scale point estimate, so no cluster-aware
interval is claimed for it. The Pile evaluation-sample loss is computed on a fixed
sample of the Pile training stream, and the analysis script records the procedure
and the aggregate losses but not the sampled-text identities. Both points are
documented in [`MANIFEST.md`](MANIFEST.md) and in the paper. Component identities
for the suppressor set are intentionally omitted from the released split-safe file.

## Citation

```bibtex
@inproceedings{dahiya2026signflip,
  title     = {A Training-Time Sign Flip in {IOI} Circuit Formation},
  author    = {Dahiya, Tejas and Blondin, Cole},
  booktitle = {Mechanistic Interpretability Workshop at the
               43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

Code in `scripts/` and `config/` is released under the MIT License. Data, figures,
and text are released under CC-BY-4.0. The analyzed models are released by their
respective authors under their own licenses.

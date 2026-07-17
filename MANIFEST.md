# Paper evidence manifest

This manifest maps every numbered paper object and every quantitative main-text claim to one canonical released artifact. It deliberately excludes exploratory or superseded files that used different protocols.

## Sign conventions

- **Logit difference** is `logit(IO) - logit(S)`.
- A positive matched-S2 intervention effect moves the model toward the correct name.
- A negative suppressor-ablation or suppressor-set replacement effect means the native suppressor supports the correct answer.
- “Window” means the reported below-chance checkpoint for the relevant model, not a universal training step.

## Main figures

| Paper object | Canonical input | Generator |
|---|---|---|
| Figure 1, six-scale IOI trajectories | `data/signflip_across_scale_160m_410m_1b.json`, `data/behavior_2.8b.json`, `data/behavior_6.9b_12b.json` | `scripts/make_figures.py` |
| Figure 2, 160M accuracy and matched-S2 trajectory | `data/signflip_across_scale_160m_410m_1b.json` | `scripts/make_figures.py` |
| Figure 2, 160M suppressor mean-ablation trajectory | `data/suppressor_ablation_trajectory.json` | `scripts/make_figures.py` |
| Figure 3, evaluation-sample loss and accuracy | `data/pile_loss_sample.json`, `data/signflip_across_scale_160m_410m_1b.json` | `scripts/make_figures.py` |

## Main-text table

| Paper object | Canonical input | Fields |
|---|---|---|
| Table 1, position perturbation summary | `data/position_control_battery.json` | `dip.arms.*.delta_ld_mean`, `mature.arms.*.delta_ld_mean` |

## Appendix tables

| Paper object | Canonical input | Fields |
|---|---|---|
| Table 2, full position battery | `data/position_control_battery.json` | all five arms, both checkpoints, means and intervals |
| Table 3, locked input control | `data/locked_input_control_160m.json` | `benchmark.n_examples`, `metrics.{dedup_effect,dedup_alt_effect,placebo_effect}` |
| Table 4, held-out probe | `data/heldout_probe.json` | layers 1 and 5 at steps 0, 2000, and 143000; raw result source-locked, generating script unresolved |
| Table 5, position-shuffle probe | `data/position_shuffle_probe.json` | initialization, layers 1, 2, 3, and 6 |
| Table 6, suppressor trajectory | `data/suppressor_ablation_trajectory.json` | 160M L8H9 and 410M L12H12 trajectories |
| Table 7, split-safe suppressor set | `data/splitsafe_suppressor_set.json` | all five checkpoints, means and cluster-aware intervals |
| Table 8, six-scale reversal | `data/signflip_across_scale_160m_410m_1b.json`, `data/behavior_2.8b.json`, `data/behavior_6.9b_12b.json`, `config/patch_windows.json` | floor accuracy, floor effect, maturity effect, patch window |
| Table 9, nine PolyPythias variants | `data/polypythias_signflip_9variants.json` | step-2000 accuracies and floor/maturity intervention effects; current PDF contains mixed-protocol accuracy cells, see release audit |
| Table 10, Stanford GPT-2 Small | `data/stanford_gpt2_signflip.json` | steps 1500, 3000, 10000, and 100000 |
| Table 11, probe-direction removal | `data/projection_removal.json` | baseline, four probe strengths, and three controls |

## Quantitative main-text claims outside tables

| Claim | Canonical input |
|---|---|
| 160M floor is 31.7 percent at step 2000 | `data/signflip_across_scale_160m_410m_1b.json` |
| At the floor, greedy decoding selects neither candidate on 98.67 percent of prompts | `data/full_vocabulary_floor.json` |
| Mean candidate probabilities are 0.0238 for S and 0.0118 for IO | `data/full_vocabulary_floor.json` |
| Mean ranks are 14.9 for S and 29.4 for IO | `data/full_vocabulary_floor.json` |
| Primary 160M template-clustered and name-pair-clustered intervals | `data/primary_intervention_clustered_cis.json` |
| Locked input-control clean accuracy and interval | `data/locked_input_control_160m.json` |
| Reported 0.06 single-head localization maximum | `data/dip_head_localization_summary.json` (paper summary only; raw source unresolved) |
| L8H9 mature effect and top-five mature heads | `data/suppressor_ablation_trajectory.json` |
| Split-safe single-head effect at step 2000 | `data/splitsafe_single_head.json` |
| Pile training-stream evaluation values at five sampled checkpoints | `data/pile_loss_sample.json` |

## Model and protocol identifiers

| Item | Canonical source |
|---|---|
| Six residual patch windows | `config/patch_windows.json` |
| Released retrained checkpoint | `https://huggingface.co/anonymous-research-sub/pythia-160m-retrained-seed42` |
| Historical ten-template protocol | first ten BABA-style families with symmetric name-role swaps; `scripts/lib/ioi_dataset.py`, `scripts/reproduce_intervention.py` |
| Source-artifact provenance | `docs/SOURCE_PROVENANCE.md` |

## Files intentionally excluded

The following classes are not part of this camera-ready release:

- alternative PolyPythia floor files generated under a different behavioral protocol
- broad post-EMNLP mechanism bundles containing copying, MLP, equality, or routing experiments
- mirrored copies of identical JSON files
- failed or superseded experiments
- raw files containing unrelated exploratory phases

Run `python scripts/verify_claims.py --verbose` for numeric checks and `python scripts/audit_release.py` for release hygiene.

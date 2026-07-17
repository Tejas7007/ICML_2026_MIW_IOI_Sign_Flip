# Source provenance

This document records the immutable source for each released result artifact and whether the camera-ready file is an exact copy, a compact paper-facing derivative, or an explicitly limited summary.

## Provenance classes

- **Exact copy** preserves the audited source blob byte-for-byte.
- **Compact derivative** removes unrelated phases or internal-only fields while preserving the paper-facing values exactly.
- **Paper summary** records a manuscript aggregate for which the exact raw result artifact has not yet been located. It is never presented as raw evidence.

## Artifact ledger

| Released artifact | Class | Audited source |
|---|---|---|
| `signflip_across_scale_160m_410m_1b.json` | Exact copy | `Tejas7007/detection-before-suppression`, `results/emnlp_consistent_signflip.json`, blob `e659ab6dc04f8c1b18b8c1471b6d57ac1e74bac3` |
| `position_control_battery.json` | Exact copy | `Tejas7007/detection-before-suppression`, `results/emnlp_control_battery.json`, blob `e8d276aed4b2fa2d45702edc885f93131a251d98` |
| `primary_intervention_clustered_cis.json` | Compact derivative | `Tejas7007/detection-before-suppression`, `results/emnlp_robustness_cis_layers.json`, blob `6af4ae987c96f0bf1b8030e70336717881aa7dbf` |
| `suppressor_ablation_trajectory.json` | Exact copy | `Tejas7007/detection-before-suppression`, `results/emnlp_suppression_head_ablation.json`, blob `e1e103683f2977661efce99321c429ba65162325` |
| `behavior_2.8b.json` | Exact copy | `Tejas7007/ioi-sign-flip-post-emnlp`, `results/mega_exp14_pythia_28b.json`, blob `914dcdad64d0726c717f528081f58b6055d3a553` |
| `behavior_6.9b_12b.json` | Exact copy | `Tejas7007/ioi-sign-flip-post-emnlp`, `results/mega2_r3_scaling_law.json`, blob `16e04e8fd5d54926750c4ac91aadd5b30589cc7c` |
| `polypythias_signflip_9variants.json` | Exact copy | `Tejas7007/ioi-sign-flip-post-emnlp`, `results/mega3_t6_multiseed_signflip.json`, blob `c6ecb618fd6185ad5110ac24668fcc92f6e1aac8` |
| `stanford_gpt2_signflip.json` | Exact copy | `Tejas7007/detection-before-suppression`, `results/emnlp_cross_model.json`, blob `0a5ec5e93bd6de0befe0925010346866c63273c8` |
| `locked_input_control_160m.json` | Exact production artifact | Locked official-160M benchmark with 800 examples, prompt hash, checkpoint and tokenizer fingerprints, raw secondary-control outputs, and crossed-cluster intervals |
| `heldout_probe.json` | Compact derivative | Original target-repository release commit `f0cab441749a0e577ee954c48cbeb00a2b156015`, `data/heldout_probe_and_position.json`, blob `aec79770a7e22bc592994610f82c3db0256af7a1`, JSON path `phase1_heldout_probes`; generating script unresolved |
| `position_shuffle_probe.json` | Exact copy | `Tejas7007/ioi-sign-flip-post-emnlp`, `results/mega9_e_arch_baseline_89.json`, blob `adfb665e04e5db04c2c566f570db179ee3d52a1d` |
| `projection_removal.json` | Exact copy | `Tejas7007/detection-before-suppression`, `results/projection_controls.json`, blob `12cdc6aa9ed65a408d9c0c4919310f3a8f6ad33d` |
| `splitsafe_suppressor_set.json` | Compact derivative | Frozen mature-suppressor-set trajectory on 192 held-out items and 384 matched contrasts per checkpoint |
| `splitsafe_single_head.json` | Compact derivative | Disjoint 800-example single-head confirmation at step 2000 |
| `full_vocabulary_floor.json` | Compact derivative | `Tejas7007/ioi-sign-flip-post-emnlp`, `results/mega5_q2_distribution_metric.json`, blob `d7e348e94136e5be074ba89d44003acc9182383c` |
| `pile_loss_sample.json` | Compact derivative | Original target-repository release commit `f0cab441749a0e577ee954c48cbeb00a2b156015`, `data/loss_and_head_sweep.json`, blob `25c4f8cccee2f02ee561634e9b7654ac35e34526`, JSON path `exp_c_loss_comparison` |
| `dip_head_localization_summary.json` | Paper summary | Manuscript aggregate only. Exact raw per-head artifact not yet source-locked |

## Protocol separation

The core residual-patch producers evaluate the first ten BABA-style template families from `ALL_TEMPLATES = BABA_TEMPLATES + ABBA_TEMPLATES`, with 30 prompts per family and symmetric name-role swaps. They do not evaluate five ABBA and five BABA families.

The historical PolyPythia behavior sweep is a different experiment. It uses the first 15 template families and 20 prompts per family. Its floor accuracies must not be combined with the nine-variant residual-patch effects as though they came from one prompt set. The producer is `scripts/polypythias_sweep.py`, blob `aea577c51ea4ab6d131f0ca07991ccba9b05e99d`.

Broad post-EMNLP bundles containing copying, MLP, equality, routing, or other mechanism phases are excluded from the paper-facing data directory.

## Producing code

`scripts/reproduce_intervention.py` is a cleaned standalone implementation of the historical core residual patch. It uses the first ten BABA-style families, tokenizer-verified single-token vocabulary items, 30 prompts per family, symmetric name-role swaps, seed 42, a NumPy checkpoint-local donor generator, `hook_resid_post` replacement at S2, the model-specific windows in `config/patch_windows.json`, and 10,000 bootstrap draws.

The repository does not claim that this one reference runner reproduces every auxiliary experiment. Immutable producer paths for each auxiliary artifact are listed above or marked unresolved in the release audit.

No released source file may contain an access token, private API key, or absolute workstation path.

## Checkpoint release

The independently trained Pythia-160M checkpoint used by the split-safe analyses is available at `https://huggingface.co/anonymous-research-sub/pythia-160m-retrained-seed42`.

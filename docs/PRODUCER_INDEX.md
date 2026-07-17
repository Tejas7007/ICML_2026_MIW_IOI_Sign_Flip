# Producer index

This index separates the script that originally produced an artifact from the CPU-only scripts that verify and plot the released values. Historical producer paths are immutable provenance references. They are not copied into the paper-facing root when they also contain out-of-scope experiments or workstation assumptions.

| Paper evidence | Original producer | Released artifact | Status |
|---|---|---|---|
| 160M, 410M, and 1B matched-S2 trajectories | `Tejas7007/detection-before-suppression/scripts/emnlp_consistent_signflip.py` | `data/signflip_across_scale_160m_410m_1b.json` | source-locked |
| Position battery | `Tejas7007/detection-before-suppression/scripts/emnlp_control_battery.py` | `data/position_control_battery.json` | source-locked |
| Primary clustered intervals | `Tejas7007/detection-before-suppression/scripts/emnlp_robustness_cis_layers.py` | `data/primary_intervention_clustered_cis.json` | source-locked |
| Single-head suppressor trajectory | `Tejas7007/detection-before-suppression/scripts/emnlp_suppression_head_ablation.py` | `data/suppressor_ablation_trajectory.json` | source-locked |
| Stanford GPT-2 trajectory | `Tejas7007/detection-before-suppression/scripts/emnlp_cross_model.py` | `data/stanford_gpt2_signflip.json` | source-locked |
| 2.8B trajectory | `Tejas7007/ioi-sign-flip-post-emnlp/experiments/core.py`, experiment 14 | `data/behavior_2.8b.json` | result source-locked; producer function should be pinned before archival tag |
| 6.9B and 12B trajectories | `Tejas7007/ioi-sign-flip-post-emnlp/experiments/r2_scaling_taxonomy.py`, scaling-law experiment | `data/behavior_6.9b_12b.json` | result source-locked; producer function should be pinned before archival tag |
| Nine PolyPythia causal reversals | `Tejas7007/ioi-sign-flip-post-emnlp/experiments/r3_mechanism_localization.py`, `t6_multiseed_signflip` | `data/polypythias_signflip_9variants.json` | source-locked |
| Historical PolyPythia behavior sweep | `Tejas7007/detection-before-suppression/scripts/polypythias_sweep.py` | intentionally excluded | separate 15-template protocol |
| Full-vocabulary floor metrics | `Tejas7007/ioi-sign-flip-post-emnlp/experiments/r5_robustness.py`, distribution-metric experiment | `data/full_vocabulary_floor.json` | source-locked result |
| Position-shuffle probe | `Tejas7007/ioi-sign-flip-post-emnlp/experiments/r9_reviewer_responses.py`, `e_arch_baseline_89` | `data/position_shuffle_probe.json` | source-locked |
| Probe-direction removal | `Tejas7007/detection-before-suppression/scripts/projection_control.py` | `data/projection_removal.json` | source-locked |
| Held-out probe | producer not yet located; raw source is original release `data/heldout_probe_and_position.json`, blob `aec79770a7e22bc592994610f82c3db0256af7a1` | `data/heldout_probe.json` | result source-locked, producer unresolved |
| Window head-localization maximum | producer and exact per-head result not yet located | `data/dip_head_localization_summary.json` | blocking provenance gap |
| Locked 800-prompt input control | modern paper-pipeline `gateA_dip_screen` experiment | `data/locked_input_control_160m.json` | production artifact retained; pin producer path before archival tag |
| Frozen suppressor set | modern fair-developmental-trajectory experiment | `data/splitsafe_suppressor_set.json` | paper-facing derivative; pin producer path before archival tag |
| Disjoint single-head suppressor check | modern dense-causal-map experiment | `data/splitsafe_single_head.json` | paper-facing derivative; pin producer path before archival tag |
| Pile evaluation-sample loss | producer not yet pinned; raw aggregate is original release `data/loss_and_head_sweep.json`, blob `25c4f8cccee2f02ee561634e9b7654ac35e34526`, JSON path `exp_c_loss_comparison` | `data/pile_loss_sample.json` | aggregate source-locked; exact sampled texts unrecoverable |

## Clean reference implementation

`scripts/reproduce_intervention.py` is the maintained standalone implementation for the core residual patch. It replaces historical absolute-path fallbacks and broad exception handling with explicit configuration, deterministic prompt materialization, fail-fast token checks, a prompt manifest hash, and optional per-example outputs.

The existence of the reference runner does not change the provenance of committed historical artifacts. Any newly generated result must be stored under `results/` and must not overwrite `data/` without a reviewed reconciliation.

# Reproducibility manifest

Every table and figure in the paper is backed by one file in [`data/`](data/).
The script [`scripts/verify_claims.py`](scripts/verify_claims.py) loads these
files and checks all 71 reported numbers to the precision at which they appear
in the paper; [`scripts/make_figures.py`](scripts/make_figures.py) redraws the
three figures from the same files.

Run both from the repository root:

```bash
python scripts/verify_claims.py --verbose
python scripts/make_figures.py
```

## Tables

| Paper object | Claim | Data file | Field path |
|---|---|---|---|
| Table 1 / Table 2 (App. B) | Position control battery; only S2 reverses sign | `position_control_battery.json` | `dip.arms.<arm>.delta_ld_mean`, `mature.arms.<arm>.delta_ld_mean` |
| Table 3 (App. B) | Locked input-level control, 800 prompts, cluster-aware CIs | `locked_input_control_160m.json` | `metrics.dedup_effect`, `metrics.placebo_effect`, `metrics.accuracy` |
| Table 4 (App. C) | Held-out probe generalisation (near chance → ~61% → ~99%) | `heldout_probe_and_position.json` | `phase1_heldout_probes.steps.<step>.by_layer.<layer>.mean_held_out_acc` |
| Table 5 (App. C) | Position-shuffle probe (90.0% vs 79.7% at init) | `position_shuffle_probe.json` | `0.per_layer.<layer>.{intact,shuffled}` |
| Table 6 (App. D) | Suppressor mean-ablation trajectory (160M, 410M) | `suppressor_ablation_trajectory.json` | `<model>.by_step.<step>.ablation_delta` |
| Table 7 (App. D) | Split-safe frozen suppressor-set margin trajectory | `splitsafe_suppressor_set.json` | `by_step.<step>.d_margin3_mean` |
| Table 5 (App. E) | Sign flip across six Pythia scales | `signflip_across_scale_160m_410m_1b.json`, `behavior_6.9b_12b.json`, `behavior_2.8b.json` | `<model>.by_step.<step>.{ioi_acc,delta_ld_mean}` |
| Table 6 (App. F) | Sign flip across nine PolyPythias variants (9/9) | `polypythias_signflip_9variants.json`, `polypythias_floors.json` | `seeds.<variant>.{dip,mature}.delta_ld_mean` |
| Table 8 (App. G) | Stanford GPT-2 Small sign flip | `stanford_gpt2_signflip.json` | `by_model.stanford_alias.step_<n>.delta_ld_mean` |
| Table 9 (App. H) | Projection removal at four strengths | `projection_removal.json` | `remove_dup_direction_<k>x.ld`, control directions |

## Figures

| Figure | Content | Data file(s) |
|---|---|---|
| Figure 1 | Below-chance window across six scales | `signflip_across_scale_160m_410m_1b.json`, `behavior_6.9b_12b.json`, `behavior_2.8b.json` |
| Figure 2 | Accuracy, matched-S2 intervention (with band), suppressor mean-ablation (with band) | `signflip_across_scale_160m_410m_1b.json`, `suppressor_ablation_trajectory.json` |
| Figure 3 | Pile evaluation-sample loss and accuracy | `loss_and_head_sweep.json` (`exp_c_loss_comparison`) |

## In-text quantities

| Section | Claim | Data file | Field |
|---|---|---|---|
| Section 4 | At the floor the model selects neither name on 98.67% of prompts; the repeated name outranks the correct one | `full_vocabulary_floor.json` | `greedy_selects_neither_fraction`, `prob_*`, `mean_rank_*` |
| Section 6 | No single head has a large zero-ablation effect during the window; Pile loss decreases at every sampled step | `loss_and_head_sweep.json` | `exp_c_loss_comparison`, `exp_a_path_patching` |

## Patch windows

The residual-stream patch window per model (used by
[`scripts/reproduce_intervention.py`](scripts/reproduce_intervention.py)) is in
[`config/patch_windows.json`](config/patch_windows.json): 160M layers 3-5,
410M layers 6-10, 1B layers 4-8, 2.8B and 6.9B layers 8-14, 12B layers 9-16.

## Notes on scope

Two quantities are reported in the paper as point estimates or non-reconstructable
by design, and the data files reflect this:

- The Pythia-410M window effect (+0.11) is a cross-scale point estimate; the
  released file stores its mean and prompt-level interval, not per-example
  cluster identifiers.
- The Pile evaluation-sample loss is computed on a fixed sample of the Pile
  training stream (`monology/pile-uncopyrighted`, `split="train"`); the analysis
  script records the procedure and aggregate losses but not the sampled-text
  identities, so the exact historical sample is not byte-reconstructable.

The `splitsafe_suppressor_set.json` file reports the frozen suppressor-set
margin trajectory only; individual component (head) identities are intentionally
omitted.

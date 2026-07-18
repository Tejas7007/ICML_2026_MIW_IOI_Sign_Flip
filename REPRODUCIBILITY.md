# Reproducing the results

## Environment

The numerical checks and figure generation require Python 3.10 or newer, NumPy, and Matplotlib.

```bash
python -m pip install numpy matplotlib
```

Model inference additionally requires PyTorch, Transformers, and TransformerLens.

```bash
python -m pip install -r requirements.txt
```

## Verify the reported values

```bash
python scripts/verify_claims.py --verbose
```

The verifier checks the released numerical results, protocol metadata, figure and paper files, model links, and common credential or machine-path leaks.

## Regenerate the figures

```bash
python scripts/make_figures.py
```

This writes PDF and PNG versions of Figures 1 through 3 to `figures/`. A different output directory can be supplied with:

```bash
python scripts/make_figures.py --output-dir /tmp/ioi-sign-flip-figures
```

## Reproduce the matched-S2 intervention

```bash
python scripts/reproduce_intervention.py \
  --model pythia-160m \
  --step 2000 \
  --output results/pythia-160m_step2000.json \
  --save-per-example
```

The primary intervention uses the first ten BABA-style IOI template families, 30 prompts per family, symmetric exchanges of the indirect-object and repeated-subject names, and seed 42. For every prompt, the control replaces S2 with a third single-token name. The control residual state at S2 is inserted at the model-specific layers in `config/patch_windows.json`.

The metric is the change in:

```text
logit(IO) - logit(S)
```

Positive values move the model toward the correct name. Negative values move it toward the repeated name. The runner reports a 95 percent prompt-bootstrap interval. Template-clustered and name-pair-clustered intervals for the primary 160M result are in `data/primary_intervention_clustered_cis.json`.

## Reproduce the single-head localization sweep

```bash
python scripts/reproduce_head_ablation.py \
  --step 1000 \
  --output results/head_zero_ablation_step1000.json
```

This evaluation uses the first eight BABA-style template families, 20 prompts per family, symmetric name-role swaps, and seed 42. It independently zeroes all 144 attention-head outputs in Pythia-160M and stores every head-level change in mean IO-minus-S logit difference.

## PolyPythias protocols

Two PolyPythias evaluations are reported and stored separately.

- `data/polypythias_floors.json` contains the minimum forced-choice accuracy from a behavioral sweep using the first fifteen BABA-style template families and 20 prompts per family.
- `data/polypythias_signflip_9variants.json` contains the matched-S2 intervention effects using the ten-family, 30-prompt-per-family intervention protocol.

The accuracy at the intervention checkpoint in the second file is not used as the behavioral floor.

## Independently trained checkpoint

The Pythia-160M checkpoint used in the split-safe analyses is available at:

```text
https://huggingface.co/teys7007/pythia-160m-seed42-dense
```

## Paper-to-code map

| Paper result | Data | Reproduction or plotting code |
|---|---|---|
| Figure 1 and six-scale behavior | `signflip_across_scale_160m_410m_1b.json`, `behavior_2.8b.json`, `behavior_6.9b_12b.json` | `scripts/make_figures.py` |
| Figure 2 and primary sign reversal | `signflip_across_scale_160m_410m_1b.json`, `primary_intervention_clustered_cis.json`, `suppressor_ablation_trajectory.json` | `scripts/reproduce_intervention.py`, `scripts/make_figures.py` |
| Figure 3 and sampled Pile loss | `pile_loss_sample.json` | `scripts/make_figures.py` |
| Position battery | `position_control_battery.json` | verified by `scripts/verify_claims.py` |
| Locked input-level control | `locked_input_control_160m.json` | verified by `scripts/verify_claims.py` |
| Full-vocabulary floor analysis | `full_vocabulary_floor.json` | verified by `scripts/verify_claims.py` |
| Step-1000 single-head sweep | `head_zero_ablation_step1000.json` | `scripts/reproduce_head_ablation.py` |
| Held-out and position-shuffle probes | `heldout_probe.json`, `position_shuffle_probe.json` | verified by `scripts/verify_claims.py` |
| Suppressor trajectory, characterization, and split-safe checks | `suppressor_ablation_trajectory.json`, `suppressor_characterization.json`, `splitsafe_suppressor_set.json`, `splitsafe_single_head.json` | verified by `scripts/verify_claims.py` |
| Independently trained model behavior | `retrained_model_behavior.json` | verified by `scripts/verify_claims.py` |
| Six-scale intervention reversal | cross-scale behavior files and `config/patch_windows.json` | `scripts/reproduce_intervention.py` |
| PolyPythias behavioral floors | `polypythias_floors.json` | verified by `scripts/verify_claims.py` |
| PolyPythias intervention reversal | `polypythias_signflip_9variants.json` | verified by `scripts/verify_claims.py` |
| Stanford GPT-2 replication | `stanford_gpt2_signflip.json` | verified by `scripts/verify_claims.py` |
| Probe-direction removal | `projection_removal.json` | verified by `scripts/verify_claims.py` |

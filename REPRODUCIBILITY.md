# Reproducing the results

This guide distinguishes rerunning a model experiment from checking a stored
result. The distinction matters because `scripts/verify_claims.py` validates
the committed paper numbers without downloading models, while the
`reproduce_*.py` scripts perform new model inference.

## Environment

Numerical verification and figure generation require Python 3.10 or newer,
NumPy, and Matplotlib.

```bash
python -m pip install numpy matplotlib
```

Model inference additionally requires PyTorch, Transformers,
TransformerLens, scikit-learn, and Hugging Face Datasets.

```bash
python -m pip install -r requirements.txt
```

## Support levels

| Status | Meaning |
|---|---|
| End-to-end reproducible | A public script rebuilds the result from model checkpoints and deterministic prompts. |
| Regenerated from released results | A public script recreates a figure or table from committed JSON. |
| Numerically verified stored artifact | The committed values are checked, but the exact original producer is not consolidated here. |

## Verify the released package

```bash
python scripts/verify_claims.py --verbose
```

This is a fast, CPU-only consistency check. It does not rerun model inference.

## Regenerate the figures

```bash
python scripts/make_figures.py
```

This writes PDF and PNG versions of Figures 1 through 3 to `figures/`.

## Primary matched-S2 intervention

```bash
python scripts/reproduce_intervention.py \
  --model pythia-160m \
  --step 2000 \
  --output results/pythia-160m_step2000.json \
  --save-per-example
```

The intervention uses ten BABA-style IOI template families, 30 prompts per
family, symmetric name-role swaps, and seed 42. For every prompt, the matched
control changes only S2, replacing the repeated name with a third
single-token name. At each configured layer, the residual-stream activation
at S2 in the original run is replaced with the activation at the same position
from the control run.

The measured effect is the change in

```text
logit(IO) - logit(S)
```

Positive values move the model toward the correct name. Negative values move
it toward the repeated name. Each checkpoint supplies its own control
activations, so the experiment compares the effect of one fixed procedure,
not one fixed activation vector.

The runner reports a prompt-bootstrap interval. Dependence-aware intervals for
Pythia-160M are stored in `data/primary_intervention_clustered_cis.json`.

## Position-control battery

```bash
python scripts/reproduce_position_controls.py
```

The script evaluates Pythia-160M at steps 2000 and 143000 on the same
300-prompt set. It compares the matched S2 replacement with random replacements
at S2, S1, the correct-name position, and a structural position.

For every random arm, each prompt and layer receives an independent Gaussian
direction normalized to the norm of the clean residual-stream activation at
that position. The activation is replaced rather than incremented. All arms
use layers 3 through 5. The intervals in
`data/position_control_battery.json` are prompt-bootstrap intervals.

## Locked 800-prompt input-level control

```bash
python scripts/reproduce_input_control.py
```

The deterministic test split has 800 examples and prompt hash:

```text
34d4fd78419110f21e70f8129a84d992cc6b10d02ddaa4c5d172c6d586ad0553
```

The conditions are the original repeated-name prompt, two independent choices
of third name at S2, and a placebo that changes a neutral place or adjective
while preserving repetition.

Confidence intervals use a crossed, two-way pigeonhole bootstrap over template
family and unordered name pair. Matched ABBA/BABA examples retain the same
cluster weight in every bootstrap draw.

## Full-vocabulary behavior and sampled Pile loss

```bash
python scripts/reproduce_loss_and_vocabulary.py --analysis vocabulary
python scripts/reproduce_loss_and_vocabulary.py --analysis loss
```

The vocabulary analysis evaluates all 300 prompts at Pythia-160M step 2000 and
reports the candidate probabilities, vocabulary ranks, and fraction for which
the top token is neither candidate.

The sampled-loss procedure streams the `monology/pile-uncopyrighted` training
split, skips 100,000 records, reads the next 100 candidate texts, keeps the
first 50 with at least 512 tokens, truncates each to 512 tokens, and averages
causal language-model loss at steps 1000, 2000, 3000, 5000, and 10000.

This is a sample from the training stream, not a held-out validation set. Exact
numerical reproduction depends on the external streaming dataset retaining its
record order.

## Step-1000 all-head sweep

```bash
python scripts/reproduce_head_ablation.py \
  --step 1000 \
  --output results/head_zero_ablation_step1000.json
```

This evaluation uses the first eight BABA-style template families and 20
prompts per family, giving 160 prompts. It independently zeroes all 144
attention-head outputs in Pythia-160M. This is separate from the 300-prompt
mature suppressor analysis.

## Held-out and position-shuffle probes

```bash
python scripts/reproduce_probes.py
```

For the held-out probe, the positive class is the residual-stream activation
at S2 in the repeated-name prompt. The negative class is the activation at the
same position in its matched non-repeating control. Selection and validation
partitions use disjoint name sets and template families. Each fold trains on
600 activation examples and evaluates on 600 examples; the folds swap the
partitions.

The position-shuffle analysis is a separate in-distribution control on the
300-prompt, ten-family set at initialization. Within each template batch, the
same token-position permutation is applied to the repeated and control prompts,
and the S2 index is updated. A standardized ridge classifier with
regularization 50 is trained on one random half and evaluated on the other.

## Mature-selected suppressor trajectory

```bash
python scripts/reproduce_suppressor_trajectory.py
```

For Pythia-160M and Pythia-410M, the script mean-ablates every head at
maturity, selects the head whose ablation most reduces mature IO-minus-S logit
difference, fixes that identity, and mean-ablates the same head at each earlier
checkpoint. A negative ablation effect means that removing the head lowers the
IO-minus-S logit difference, so the intact head supports the correct name.

The script also measures attention to S2 and direct IO-minus-S output
projection for Pythia-160M L8H9 at selected checkpoints, and characterizes
Stanford GPT-2 Small L10H10 at maturity. It does not claim a Stanford
training-time suppressor trajectory.

## Split-safe frozen suppressor-set trajectory

```bash
python scripts/reproduce_splitsafe_suppressors.py
```

The independently trained model is:

```text
https://huggingface.co/teys7007/pythia-160m-seed42-dense
```

The mature suppressor set is selected on one partition and retained only when
its sign also validates on a separate partition. The selection rule requires
negative selection and validation confidence intervals, a positive effect on
the repeated-name logit when the native head is restored, and the suppressive
sign in both matched directions.

The resulting 15-head set is frozen before the developmental trajectory is
evaluated. The strict holdout excludes every name and template family used by
the discovery partitions.

Each of 192 held-out items supplies two contrasts, `XX <- YX` and `YY <- XY`.
The S2 token is held fixed. The complete `hook_z` output at END for every head
in the frozen set is copied from the matched donor into the target. This yields
384 observations per checkpoint.

The fair three-way margin is:

```text
logit(IO) - max(logit(repeated S), logit(donor alternate name))
```

The two directions are averaged within each item before bootstrapping the 192
item means.

### Historical single-head artifact

`data/splitsafe_single_head.json` records a separate 800-example result with
mean 0.038 and interval [-0.024, 0.114]. The exact selected-head identity and
original producer were not retained in the consolidated release. It is
therefore classified as a numerically verified stored artifact, not an
end-to-end reproduction. The fully documented set-level analysis above is the
recommended split-safe robustness result.

## Probe-direction removal

```bash
python scripts/reproduce_projection_removal.py
```

At Pythia-160M step 2000, a logistic-regression probe is trained on layer-5 S2
activations. Its normalized weight vector `d` is removed at strengths 0.5, 1,
2, and 4 using:

```text
h' = h - strength * (h dot d) * d
```

The script also tests an orthogonal direction, a shuffled-label probe
direction, and five random unit directions. The reported values are point
estimates. This experiment tests one learned direction at one layer and
checkpoint; it does not rule out other linear directions, distributed
representations, or nonlinear encodings.

## Cross-model replications

```bash
python scripts/reproduce_replications.py --suite stanford
python scripts/reproduce_replications.py --suite polypythias
```

Stanford GPT-2 Small uses `stanford-crfm/alias-gpt2-small-x21`, revisions of
the form `checkpoint-{step}`, 300 prompts, and layers 3 through 5.

The nine PolyPythia variants use step 2000 for the causal window evaluation and
step 143000 for maturity. These intervention checkpoints are separate from the
15-family behavioral-floor sweep in `data/polypythias_floors.json`.
The accuracy at the intervention checkpoint in the second file is not used as
the behavioral floor.

## Paper-to-code map

| Paper result | Released data | Public producer or support | Status |
|---|---|---|---|
| Figures 1–3 | cross-scale, suppressor, and loss JSONs | `scripts/make_figures.py` | Regenerated from released results |
| Primary matched-S2 reversal | `signflip_across_scale_160m_410m_1b.json`, `primary_intervention_clustered_cis.json` | `scripts/reproduce_intervention.py` | End-to-end reproducible |
| Position battery | `position_control_battery.json` | `scripts/reproduce_position_controls.py` | End-to-end reproducible |
| Locked input control | `locked_input_control_160m.json` | `scripts/reproduce_input_control.py` | End-to-end reproducible |
| Full-vocabulary floor | `full_vocabulary_floor.json` | `scripts/reproduce_loss_and_vocabulary.py` | End-to-end reproducible |
| Sampled Pile loss | `pile_loss_sample.json` | `scripts/reproduce_loss_and_vocabulary.py` | End-to-end subject to external stream ordering |
| Step-1000 head sweep | `head_zero_ablation_step1000.json` | `scripts/reproduce_head_ablation.py` | End-to-end reproducible |
| Held-out probe | `heldout_probe.json` | `scripts/reproduce_probes.py` | End-to-end reproducible |
| Position-shuffle probe | `position_shuffle_probe.json` | `scripts/reproduce_probes.py` | End-to-end reproducible |
| Suppressor trajectory | `suppressor_ablation_trajectory.json` | `scripts/reproduce_suppressor_trajectory.py` | End-to-end reproducible |
| Suppressor characterization | `suppressor_characterization.json` | `scripts/reproduce_suppressor_trajectory.py` | End-to-end reproducible |
| Split-safe suppressor set | `splitsafe_suppressor_set.json` | `scripts/reproduce_splitsafe_suppressors.py` | End-to-end reproducible |
| Split-safe single head | `splitsafe_single_head.json` | no exact producer retained | Numerically verified stored artifact |
| Six Pythia scales | cross-scale JSONs and `config/patch_windows.json` | `scripts/reproduce_intervention.py` | End-to-end reproducible |
| Nine PolyPythias | `polypythias_signflip_9variants.json` | `scripts/reproduce_replications.py` | End-to-end reproducible |
| Stanford GPT-2 | `stanford_gpt2_signflip.json` | `scripts/reproduce_replications.py` | End-to-end reproducible |
| Projection removal | `projection_removal.json` | `scripts/reproduce_projection_removal.py` | End-to-end reproducible |

## Scope boundary

These scripts reproduce only the claims used in the ICML Mechanistic
Interpretability Workshop paper. They intentionally exclude the broader
detection-before-suppression, MLP-localization, path-patching, QK/OV,
name-mover, and cross-task analyses developed for a separate EMNLP project.

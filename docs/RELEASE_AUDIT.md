# Camera-ready release audit

Audit date: 2026-07-17

## Status

The repository has been hardened on `audit/release-hardening`, but the release is **blocked** until every item below is resolved. These are source-provenance or manuscript-to-artifact mismatches, not requests for another broad experiment campaign.

## Blocking item 1: primary template-order wording

The audited producer defines `ALL_TEMPLATES = BABA_TEMPLATES + ABBA_TEMPLATES` and evaluates `ALL_TEMPLATES[:10]`. The released values therefore use the first ten BABA-style families with symmetric name-role swaps. The camera-ready PDF says that the primary protocol is balanced in ABBA and BABA order.

**Recommended resolution:** correct the Methods, relevant captions, and reproducibility appendix to describe the historical protocol exactly. A rerun is necessary only if the paper must retain the balanced-order claim.

## Blocking item 2: PolyPythia Table 9 mixes experiments

The nine-variant causal run evaluates step 2000 under the ten-template, 30-prompts-per-template residual-patch protocol. Its step-2000 accuracies are stored beside each intervention effect in `data/polypythias_signflip_9variants.json`.

Several floor accuracies printed in Table 9 instead come from `polypythias_ioi.json`, produced by a separate 15-template, 20-prompts-per-template behavioral sweep. The paper currently presents the floor accuracy and intervention effect as one ten-template experiment.

**Recommended resolution:** use the step-2000 accuracies from the causal file in Table 9, or state and label the two protocols separately. Using the causal file throughout requires no model rerun.

## Blocking item 3: held-out probe producer

The exact Table 4 result artifact is source-locked to `data/heldout_probe_and_position.json` at the original release commit, blob `aec79770a7e22bc592994610f82c3db0256af7a1`. The compact release file is a faithful extraction of `phase1_heldout_probes`.

The script that generated this historical artifact has not yet been located in the connected source repositories.

**Recommended resolution:** locate and pin the generating script. If it cannot be recovered, retain the raw result artifact and state that the producer code is unavailable rather than claiming full end-to-end reproduction of this auxiliary probe.

## Blocking item 4: window head-localization aggregate

Section 6 reports that the largest single-head zero-ablation change at step 2000 is 0.06 across 144 heads. The exact per-head result artifact that produces this aggregate was not found. The broad historical `exp_a_path_patching` block is a different intervention and cannot substitute for a zero-ablation sweep.

**Recommended resolution:** locate the exact artifact, remove the numerical 0.06 sentence, or rerun only this single-head sweep while saving all per-head outputs.

## Blocking item 5: final PDF and public-release state

The repository PDF predates the latest manuscript revision, and the repository is still private. The final corrected PDF must contain a clickable repository URI, its hash must be recorded in `paper/metadata.json`, and repository visibility must be verified after publication.

## Completed checks

- Primary 160M prompt-, template-, and name-pair-resampled intervals are source-locked.
- All six patch windows are explicit and traced to producing code or audited historical configurations.
- The locked 800-prompt input control retains its prompt hash, model and tokenizer fingerprints, raw secondary-control outputs, and crossed-cluster intervals.
- Frozen-set and disjoint single-head suppressor analyses remain separate estimands.
- Pile loss values are source-locked to the exact historical aggregate block and correctly labeled as a fixed sample from the training stream.
- The retrained checkpoint URL is the author-supplied public mirror.
- The held-out probe raw result is source-locked even though its generating script remains unresolved.
- Automated checks scan for credentials, stale identifiers, absolute workstation paths, malformed JSON, forbidden legacy files, and unresolved release blockers.

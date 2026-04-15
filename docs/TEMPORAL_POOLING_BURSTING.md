# Temporal Pooling And Bursting

This note explains the Layer 1 bursting issue seen while investigating
temporal pooling in `chat_htm`, what appears to cause it, where to look in the
code, and how to test it.

## Short Summary

The symptom is: enabling temporal pooling can increase column bursting in Layer
1 instead of smoothly improving temporal stability.

The main reason is that a column only avoids bursting when a cell was both:

- predictive at `t-1`
- backed by an active sequence segment at `t-1`

Temporal pooling can interfere with that in two ways:

1. It updates the same proximal permanence tensor already used by spatial
   learning, so TP can make columns win inhibition before distal sequence
   structure is strong enough to support non-bursting predictions.
2. If TP persistence is enabled, it can mark cells predictive without writing
   matching `active_segs_time`, which creates a "predictive but still bursts"
   mismatch.

## Why Bursting Happens

For newly active columns, the burst/no-burst decision is made in:

- `htm_flow/src/sequence_pooler/active_cells/active_cells.cpp`

The key rule is:

- if no cell in the column was predictive at `t-1` and had an active segment at
  `t-1`, the column bursts

Important supporting code paths:

- `htm_flow/src/htm_layer.cpp`
  - step order is: overlap -> inhibition -> spatial learning -> active cells ->
    predict cells -> sequence learning -> temporal pooler
  - TP runs after normal sequence memory work and writes back into shared state
- `htm_flow/src/temporal_pooler/temporal_pooler.cpp`
  - `update_proximal()` changes `col_syn_perm_`
  - `update_distal()` changes `distal_synapses_`
  - optional persistence can write `predict_cells_time` without matching segment
    state
- `htm_flow/src/sequence_pooler/predict_cells/predict_cells.cpp`
  - `active_segs_time_` is produced here from connected distal support

## Important Investigation Notes

- Temporal pooling is not a separate clean layer of logic. It shares learning
  tensors with the normal layer algorithm.
- The burst spike observed when turning TP on later was useful as a diagnosis,
  but the real goal is a static config that works from the start.
- At the moment, the safest configuration is to keep TP enabled from the
  beginning, disable TP persistence, keep TP proximal learning weak, and let TP
  distal learning dominate early.
- If someone wants to re-enable TP persistence in the future, the first place
  to inspect is the persistence logic in
  `htm_flow/src/temporal_pooler/temporal_pooler.cpp`, because that is the most
  likely path to recreate "predictive but still bursting" behavior.

## Current Config Direction

The current reference config is:

- `configs/word_rows_2layer_text.yaml`

Layer 1 temporal pooling is enabled from the start with:

- `enable_persistence: false`
- small TP proximal learning
- stronger TP distal learning

This is meant to avoid the need for timestep-specific runtime patching while
still allowing temporal pooling to learn.

## How To Test

The main regression coverage is in:

- `tests/integration/test_text_htm.cpp`

Relevant tests:

- `TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting`
  - keeps the original diagnostic scenario where TP is applied at runtime
- `TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP`
  - compares always-on TP against a TP-off control

Run them with:

```bash
cmake --build build --target chat_htm_tests
./build/chat_htm_tests --gtest_filter=TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting:TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP
```

## What To Look At During Debugging

When investigating this issue again, check:

- whether Layer 1 active columns are increasing without a matching increase in
  predictive coverage
- whether bursting rises when TP proximal learning is too strong
- whether persistence was enabled and is producing predictive state without
  valid segment evidence
- whether distal learning rates and thresholds make it too hard for TP-created
  synapses to become useful before new columns start winning

## Future Work

Future improvements should focus on making temporal learning stronger without
reintroducing burst spikes.

The likely next step is not a larger config sweep, but a closer look at whether
TP persistence should integrate differently with the burst gate and sequence
segment bookkeeping.

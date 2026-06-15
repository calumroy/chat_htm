# Temporal Pooling Config Guide

This guide explains the temporal-pooling knobs used by configs such as
`configs/word_rows_2layer_delayed_temporal_pooling_text.yaml` and
`configs/overrides/word_rows_2layer_enable_temporal_pooling.yaml`.

## Recommended Starting Point

For Layer 1 delayed temporal pooling, start with:

```yaml
temporal_pooling:
  enabled: true
  enable_persistence: false
  delay_length: 16
  spatial_permanence_inc: 0.05
  active_predict_proximal_scale: 0.25
  predictive_non_active_proximal_scale: 0.0
  post_active_proximal_scale: 0.1
  sequence_permanence_inc: 0.21
  sequence_permanence_dec: 0.002
```

This is intentionally conservative. Temporal pooling still causes some column
bursting, so do not assume a higher setting is better just because it produces
more reinforced inputs.

## What Each Knob Does

- `enabled`: turns temporal pooling on for the layer. In delayed configs, keep
  this `false` in the base config and enable it with a runtime override after
  sequence memory has had time to form.

- `enable_persistence`: lets TP keep cells predictive for a short window.
  Keep this `false` while tuning bursting. It can help continuity, but it is
  easier to make stale predictions that increase bursting pressure.

- `delay_length`: smooths the persistence estimate. It only matters when
  `enable_persistence` is true. With persistence off, it is not the main tuning
  knob.

- `spatial_permanence_inc`: proximal TP learning rate. This is the main knob for
  making predicted columns start to compete by overlap. Increase it slowly. Too
  low means TP predicts distally but does not change winners. Too high can make
  a small set of columns sticky and burst-prone.

- `active_predict_proximal_scale`: multiplier for proximal reinforcement on
  columns that were active and correctly predicted. This is the safest TP
  proximal path because the column already avoided bursting.

- `predictive_non_active_proximal_scale`: multiplier for columns that were
  segment-backed predictive but did not win inhibition. Keep this low or zero
  while tuning bursting; raising it can make predicted non-winners become future
  winners before their distal context is reliable enough.

- `post_active_proximal_scale`: multiplier for the one-step bridge after a
  correctly predicted activation. Small values can help continuity without
  turning every segment-backed prediction into immediate proximal pressure.

- `sequence_permanence_inc`: distal TP learning rate. This controls how quickly
  TP-created distal synapses become useful for prediction. This should usually
  be stronger than `spatial_permanence_inc`.

- `sequence_permanence_dec`: distal decay for non-matching TP synapses. Raise it
  if old distal context sticks around too long. Lower it if useful TP segments
  fail to stabilize.

## Tuning Order

1. Keep `enable_persistence: false`.
2. Keep `predictive_non_active_proximal_scale: 0.0` at first.
3. Tune `sequence_permanence_inc` until predictive coverage improves.
4. Increase `spatial_permanence_inc` gradually with a conservative
   `active_predict_proximal_scale`.
5. Add a small `post_active_proximal_scale` only if active-predict columns need
   more continuity.
6. Raise `predictive_non_active_proximal_scale` last, and only if non-winning
   predicted columns need help becoming active through overlap and inhibition.
7. Watch Layer 1 bursting. If bursting rises, back off the non-active scales
   before changing persistence.
8. Only test `enable_persistence: true` after non-persistence TP is stable.

## What To Watch

Useful signs:

- Layer 1 predictive coverage increases.
- `reinforced_inputs` is non-zero after TP turns on.
- active columns broaden beyond the old saturated winner set.
- bursting does not rise sharply compared with TP-off or pre-TP behavior.

Bad signs:

- `reinforced_inputs` is non-zero but active columns do not change:
  `spatial_permanence_inc` or the non-active proximal scales may be too small.
- a small set of columns dominates every timestep:
  `spatial_permanence_inc` or `predictive_non_active_proximal_scale` may be too
  high, or Layer 1 spatial pooling is too collapsed before TP turns on.
- bursting rises after enabling persistence:
  turn persistence back off and tune distal/proximal learning first.

## Current Caveat

The current implementation reduces the old failure mode by requiring local
distal evidence before TP changes proximal permanence. It does not eliminate
bursting. The goal is maximum temporal pooling without a burst spike, not simply
maximum TP learning rates.

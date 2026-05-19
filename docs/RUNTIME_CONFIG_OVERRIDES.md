# Runtime Config Overrides

Use `runtime_parameter_schedule` in a config YAML to apply small runtime patches
at specific timesteps during a headless run.

```yaml
runtime_parameter_schedule:
  - at_timestep: 1000
    override: overrides/word_rows_2layer_enable_temporal_pooling.yaml
```

`override` paths are resolved relative to the config file. The example above
loads `configs/overrides/word_rows_2layer_enable_temporal_pooling.yaml` when the
runtime timestep reaches `1000`.

Override files use the same layer order as the main config, but only
hot-swappable values are allowed. Leave unchanged layers as `{}`:

```yaml
layers:
  - {}
  - temporal_pooling:
      enabled: true
      enable_persistence: true
      delay_length: 8
      spatial_permanence_inc: 0.04
      sequence_permanence_inc: 0.2
      sequence_permanence_dec: 0.004
```

For safer delayed temporal pooling, use `spatial_permanence_inc` as the local TP
proximal reinforcement knob. It only reinforces columns that had active-predict
support in the previous temporal pooler distal update.

Run it like any other config:

```bash
./build/chat_htm --input tests/test_data/simple_sentences.txt \
  --config configs/word_rows_2layer_delayed_temporal_pooling_text.yaml \
  --steps 2000 --log
```

When the patch applies, the app prints a `[runtime_patch]` line. Runtime
schedules are processed by `TextRuntime::step()`, so they apply in both
headless runs and GUI stepping.

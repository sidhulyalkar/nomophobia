# Universal Experiment Manifest

Every active NOMOPHOBIA v0.3 research or production campaign should emit a machine-readable manifest via `src/s6e8/manifest.py`.

The schema records the evidence tier, hypothesis, falsifier, accept/kill rules, Git SHA, runtime package versions, competition CSV SHA-256 hashes, exact configuration, elapsed time, metrics, and SHA-256 hashes of registered outputs.

`ExperimentRecorder` writes when a run starts and finalizes even when an exception escapes, so a crashed job is distinguishable from a negative experiment.

Authoritative S3 runs should keep input hashing enabled. Quick local screens may use `--no-hash-inputs`.

In Kaggle snapshots without `.git`, set `NOMOPHOBIA_GIT_SHA=<commit>` when possible.

New experiments should follow:

```python
with ExperimentRecorder(...) as manifest:
    ...
    manifest.add_metrics(...)
    manifest.add_output(...)
```

The scientific contract comes first: an experiment without a falsifying measurement is not ready to run.

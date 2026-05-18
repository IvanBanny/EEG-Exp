# Training modes - pooled vs per-subject

Two entry points share `src/`, the config registry, and the data
factory. They differ only in who the model is trained on.

## Pooled - `train.py`

```
python train.py --config <name>
```

One model on the union of all subjects' training data, validated on
the union of all held-out splits. `cfg.data.subjects` is either unset
(all subjects) or a small subset - pooled training does not loop over
subjects. Multi-seed averaging is opt-in via `cfg.training.num_runs > 1`:
each seed's TB log lands under a parent dir, and a sibling `_agg/`
collects mean / std / `n_active` curves and the mean confusion matrix.

Use when you want a single deployable model across the population, or
when you are sweeping hyperparameters - the sweep harness reuses the
pooled path.

## Per-subject - `train_subjects.py`

```
python train_subjects.py --config <name> --subjects 1-14 --seeds 4269-4271
```

One model per `(subject, seed)` cell. The runner pins
`cfg.data.subjects = [subject]`, reseeds, rebuilds datasets (cache hits
make this cheap after the first cell), and trains from scratch. Cell
layout adds a `subject_<N>/` level on top of the pooled multi-seed
layout:

```
runs/<config>/subject_<N>/seed_<S>/
    events.out.tfevents.*
    predictions.npz
    result.json
checkpoints/<config>/subject_<N>/seed_<S>/best.pt
```

`predictions.npz` carries per-window val logits, labels, and
per-window metadata (subject id, windows-per-trial, sfreq).
`src/eval/aggregate.py` reads it post-hoc for per-action kappa.

Use when the headline metric is within-subject. The MI literature
reports within-subject kappa, and per-subject is the deployment-honest
regime for a BCI that adapts to one user. All reproduction commands in
`docs/status.md` use this mode.

## Picking a mode

- Within-subject benchmark - per-subject.
- Cross-subject generalisation - pooled.
- Hyperparameter sweep - pooled. (Within-subject sweeps would need one
  Optuna study per subject and are out of scope.)

Both modes share the same configs - the protocol knobs (`t_fork`,
`window_sec`, `normalize`, `split_mode`, augmentation) are fixed by
the config; the entry point only chooses pooled vs per-subject.

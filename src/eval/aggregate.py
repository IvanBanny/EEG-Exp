"""Per-action aggregation and per-cell metrics for per-subject runs.

Walks `runs/<config>/subject_<N>/seed_<S>/` for `result.json` siblings
of `predictions.npz` files written by `train_subjects.py`. Computes
real Cohen's kappa per cell (per-window from the dumped logits, and
per-action via mean-of-logits aggregation across the windows of one
trial). Emits per-cell, per-subject, and per-config summary tables.

The per-action math is the cheapest deployment lever we have: for a
trial split into `wpt` consecutive windows, the trial decision is
`argmax(mean(logits, axis=window_dim))`. Reduces to per-window when
`wpt == 1`. The post-hoc lift on the headline cells is +0.03 to +0.16
kappa depending on `wpt`.

Public surface:
    aggregate(runs_dir) -> DataFrame of per-cell rows
    per_subject_table(df) -> DataFrame of (config, subject) means
    per_config_table(df) -> DataFrame of (config) mean +- std across subjects
    cell_metrics(result_json_path, predictions_npz_path) -> dict
    per_action_aggregate(logits, labels, wpt) -> (acc, kappa, n_trials)
    write_summary(out_dir, df)
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score


def per_action_aggregate(logits: np.ndarray, labels: np.ndarray,
                         wpt: int) -> Tuple[float, float, int]:
    """Mean-of-logits aggregation over `wpt` consecutive windows per trial.

    Args:
        logits: `(n_windows, n_classes)` per-window logits in the
            ordering they were emitted by the val loader.
        labels: `(n_windows,)` per-window true labels.
        wpt: Number of windows per trial. The function expects
            `len(labels) % wpt == 0` and the `wpt` windows of a given
            trial to share the same label.

    Returns:
        `(trial_acc, trial_kappa, n_trials)`. NaNs and `n_trials=0`
        when `wpt` does not evenly divide `len(labels)`, NaNs and
        `n_trials=<count>` when trial boundaries do not align with
        `wpt` chunks (defensive bail).
    """
    if wpt <= 0 or len(labels) % wpt != 0:
        return float("nan"), float("nan"), 0
    n_trials = len(labels) // wpt
    n_cls = logits.shape[1]
    by_trial = logits.reshape(n_trials, wpt, n_cls).mean(axis=1)
    lab_by_trial = labels.reshape(n_trials, wpt)[:, 0]
    if not np.all(labels.reshape(n_trials, wpt) == lab_by_trial[:, None]):
        # Trial boundaries do not line up with `wpt` chunks; refuse to
        # silently round.
        return float("nan"), float("nan"), n_trials
    pred = by_trial.argmax(1)
    return (float(accuracy_score(lab_by_trial, pred)),
            float(cohen_kappa_score(lab_by_trial, pred)),
            int(n_trials))


def cell_metrics(result_path: Path, npz_path: Path) -> dict:
    """Compute one (config, subject, seed) row.

    Reads `result_path` for metadata (best_val_*, epochs_run, etc.)
    and `npz_path` for raw val logits + labels. The NPZ schema is
    fixed by `ClassicTrainer._dump_predictions`: `logits`, `labels`,
    `subject_ids`, `windows_per_trial`, `sfreq`, `window_sec`,
    `window_overlap`, `n_times`, `num_classes`, `dataset`.
    """
    with open(result_path) as f:
        meta = json.load(f)
    config = meta.get("config") or meta.get("phase") or result_path.parents[2].name
    subject = int(meta["subject"])
    seed = int(meta["seed"])

    if not npz_path.exists():
        return {
            "config": config, "subject": subject, "seed": seed,
            "epochs_run": meta.get("epochs_run"),
            "best_val_acc": meta.get("best_val_acc"),
            "best_val_kappa": meta.get("best_val_kappa"),
            "win_acc": float("nan"), "win_kappa": float("nan"),
            "trial_acc": float("nan"), "trial_kappa": float("nan"),
            "n_trials": 0,
        }

    d = np.load(npz_path, allow_pickle=True)
    logits = d["logits"]
    labels = d["labels"]
    wpt = int(d["windows_per_trial"])

    win_pred = logits.argmax(1)
    win_acc = float(accuracy_score(labels, win_pred))
    win_kappa = float(cohen_kappa_score(labels, win_pred))
    trial_acc, trial_kappa, n_trials = per_action_aggregate(logits, labels, wpt)

    return {
        "config": config, "subject": subject, "seed": seed,
        "epochs_run": int(meta.get("epochs_run", 0)),
        "best_val_acc": meta.get("best_val_acc"),
        "best_val_kappa": meta.get("best_val_kappa"),
        "win_acc": win_acc, "win_kappa": win_kappa,
        "trial_acc": trial_acc, "trial_kappa": trial_kappa,
        "n_trials": n_trials,
    }


def aggregate(runs_dir: Path) -> pd.DataFrame:
    """Walk `runs_dir` for cell `result.json` files and compute metrics.

    Cell layout: `<runs_dir>/[<config>/]subject_<N>/seed_<S>/{result.json,
    predictions.npz}`. Points `runs_dir` at a single config to get one
    config's cells, or at `runs/` to get multiple configs in one frame.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return pd.DataFrame()
    rows = []
    for result_path in sorted(runs_dir.rglob("result.json")):
        npz_path = result_path.parent / "predictions.npz"
        rows.append(cell_metrics(result_path, npz_path))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def per_subject_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per `(config, subject)` mean across seeds + n_seeds + std."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["config", "subject"]).agg(
        n_seeds=("seed", "count"),
        win_acc=("win_acc", "mean"),
        win_acc_std=("win_acc", "std"),
        win_kappa=("win_kappa", "mean"),
        win_kappa_std=("win_kappa", "std"),
        trial_acc=("trial_acc", "mean"),
        trial_kappa=("trial_kappa", "mean"),
    ).reset_index()


def per_config_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per `config` mean +- std across subjects of seed-means.

    Honest-reporting convention: each subject contributes the mean of
    its seeds; the outer aggregation then takes the mean and std
    across subjects.
    """
    if df.empty:
        return pd.DataFrame()
    by_subj = df.groupby(["config", "subject"]).agg(
        win_acc=("win_acc", "mean"),
        win_kappa=("win_kappa", "mean"),
        trial_acc=("trial_acc", "mean"),
        trial_kappa=("trial_kappa", "mean"),
        best_val_acc=("best_val_acc", "mean"),
        best_val_kappa=("best_val_kappa", "mean"),
    ).reset_index()
    return by_subj.groupby("config").agg(
        n_subjects=("subject", "count"),
        win_acc_mean=("win_acc", "mean"),
        win_acc_std=("win_acc", "std"),
        win_kappa_mean=("win_kappa", "mean"),
        win_kappa_std=("win_kappa", "std"),
        trial_acc_mean=("trial_acc", "mean"),
        trial_acc_std=("trial_acc", "std"),
        trial_kappa_mean=("trial_kappa", "mean"),
        trial_kappa_std=("trial_kappa", "std"),
        best_val_acc_mean=("best_val_acc", "mean"),
        best_val_kappa_mean=("best_val_kappa", "mean"),
    ).reset_index()


def write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    """Write `summary.csv`, `summary_per_subject.csv`,
    `summary_per_config.csv`, and a Markdown summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    sub = per_subject_table(df)
    sub.to_csv(out_dir / "summary_per_subject.csv", index=False)
    conf = per_config_table(df)
    conf.to_csv(out_dir / "summary_per_config.csv", index=False)

    md = ["# Aggregator summary\n",
          "## Per-config headline\n",
          conf.to_markdown(index=False, floatfmt=".4f"),
          "\n\n## Per-(config, subject) means across seeds\n",
          sub.to_markdown(index=False, floatfmt=".4f"),
          "\n\n## All cells\n",
          df.to_markdown(index=False, floatfmt=".4f")]
    (out_dir / "summary.md").write_text("\n".join(md))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-subject cell predictions into summary tables."
    )
    parser.add_argument("--runs-dir", required=True,
                        help="Cell tree root, e.g. runs/<config> or runs/.")
    parser.add_argument("--out-dir", default=None,
                        help="Write summary files here. Default: <runs-dir>.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir
    df = aggregate(runs_dir)
    if df.empty:
        print(f"No cells found under {runs_dir}")
        return
    print(df.to_string())
    write_summary(out_dir, df)
    print(f"\nWrote summary.* to {out_dir}")


if __name__ == "__main__":
    main()

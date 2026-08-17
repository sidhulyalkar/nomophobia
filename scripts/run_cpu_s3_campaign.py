#!/usr/bin/env python
"""Resumable CPU-oriented authoritative S3 campaign for Kaggle notebooks.

This wrapper keeps the scientific S3 contract unchanged while making long CPU sessions
operationally safer: completed tuning/seed stages are reused only when their configuration
matches the requested frozen iteration counts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from s6e8.artifacts import atomic_write_json, sha256_file
from s6e8.io import load_competition
from s6e8.s3 import resolution_from_promotions, summarize_s3_runs
from s6e8.validation import validate_submission

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def tuning_ok(path: Path, *, repeats: int, ceiling_fraction: float) -> bool:
    d = read_json(path)
    return bool(
        d
        and int(d.get("repeats", -1)) == repeats
        and abs(float(d.get("ceiling_fraction", -1)) - ceiling_fraction) < 1e-12
        and int(d.get("best_iteration", 0)) > 0
    )


def seed_ok(path: Path, combined: int, raw: int) -> bool:
    summary = read_json(path / "run_summary.json")
    if not summary:
        return False
    overrides = summary.get("expert_iteration_overrides") or {}
    required = {
        "lgb_combined63": int(combined),
        "lgb_raw63": int(raw),
    }
    if any(int(overrides.get(k, -1)) != v for k, v in required.items()):
        return False
    needed = [
        path / "folds.csv",
        path / "oof_lgb_combined63.npy",
        path / "oof_lgb_raw63.npy",
        path / "oof_blend.npy",
        path / "test_lgb_combined63.npy",
        path / "test_lgb_raw63.npy",
        path / "test_blend.npy",
        path / "blend.json",
    ]
    return all(p.exists() for p in needed)


def hash_inputs_once(data_dir: Path, out_root: Path) -> dict:
    path = out_root / "input_hashes.json"
    if path.exists():
        return read_json(path)
    hashes = {name: sha256_file(data_dir / name) for name in ("train.csv", "test.csv", "sample_submission.csv")}
    atomic_write_json(path, hashes)
    return hashes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/kaggle/input/playground-series-s6e8")
    p.add_argument("--out-root", default="/kaggle/working/nomophobia_cpu_s3")
    p.add_argument("--tune-rows", type=int, default=628_000)
    p.add_argument("--max-estimators", type=int, default=4_000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--tune-repeats", type=int, default=3)
    p.add_argument("--ceiling-fraction", type=float, default=0.90)
    p.add_argument("--bootstrap", type=int, default=1_200)
    p.add_argument("--seeds", nargs=3, type=int, default=[20260816, 20260817, 20260818])
    p.add_argument("--combined-iterations", type=int, default=None)
    p.add_argument("--raw-iterations", type=int, default=None)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--zip-name", default="nomophobia_cpu_s3_artifacts.zip")
    a = p.parse_args()

    if len(set(a.seeds)) != 3:
        p.error("S3 requires exactly three distinct seeds")
    if a.threads > 0:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = str(a.threads)

    data_dir = Path(a.data_dir)
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    resume = not a.no_resume
    input_hashes = hash_inputs_once(data_dir, out_root)
    py = sys.executable
    started = time.time()

    selected: dict[str, int] = {}
    tuning: dict[str, dict] = {}
    for expert, label, provided in (
        ("lgb_combined63", "combined", a.combined_iterations),
        ("lgb_raw63", "raw", a.raw_iterations),
    ):
        if provided is not None:
            selected[label] = int(provided)
            tuning[label] = {"provided": True, "best_iteration": int(provided)}
            continue
        out = out_root / f"tune_{label}.json"
        if resume and tuning_ok(out, repeats=a.tune_repeats, ceiling_fraction=a.ceiling_fraction):
            print(f"Reusing completed tuning artifact: {out}")
        else:
            run([
                py,
                str(ROOT / "tune_iterations.py"),
                "--data-dir", str(data_dir),
                "--expert", expert,
                "--rows", str(a.tune_rows),
                "--max-estimators", str(a.max_estimators),
                "--patience", str(a.patience),
                "--repeats", str(a.tune_repeats),
                "--ceiling-fraction", str(a.ceiling_fraction),
                "--device", "cpu",
                "--out", str(out),
            ])
        meta = read_json(out)
        if meta.get("ceiling_hit_any_repeat"):
            stop = {
                "version": "nomophobia-v0.3-cpu-s3",
                "status": "STOP_MAX_ESTIMATOR_CEILING",
                "expert": expert,
                "tuning": meta,
                "action": "Raise --max-estimators and rerun before spending S3 folds.",
            }
            atomic_write_json(out_root / "cpu_s3_summary.json", stop)
            print(json.dumps(stop, indent=2))
            raise SystemExit(3)
        selected[label] = int(meta["best_iteration"])
        tuning[label] = meta

    print(f"Frozen iterations: combined={selected['combined']} raw={selected['raw']}")

    s3_root = out_root / "s3"
    s3_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in a.seeds:
        run_dir = s3_root / f"seed_{seed}"
        runs.append(run_dir)
        if resume and seed_ok(run_dir, selected["combined"], selected["raw"]):
            print(f"Reusing completed S3 seed run: {run_dir}")
            continue
        run([
            py,
            str(ROOT / "train.py"),
            "--data-dir", str(data_dir),
            "--out-dir", str(run_dir),
            "--preset", "full",
            "--device", "cpu",
            "--fold-seed", str(seed),
            "--n-splits", "5",
            "--experts", "lgb_combined63", "lgb_raw63",
            "--expert-iterations",
            f"lgb_combined63={selected['combined']}",
            f"lgb_raw63={selected['raw']}",
        ])

    promotions = {}
    for baseline, candidate in (("lgb_raw63", "lgb_combined63"), ("lgb_combined63", "blend")):
        out = s3_root / f"promotion__{baseline}__to__{candidate}"
        run([
            py,
            str(ROOT / "aggregate_promotion.py"),
            "--data-dir", str(data_dir),
            "--runs", *[str(x) for x in runs],
            "--baseline", baseline,
            "--candidate", candidate,
            "--bootstrap", str(a.bootstrap),
            "--out-dir", str(out),
        ])
        promotions[(baseline, candidate)] = out / "promotion.json"

    diagnostics = summarize_s3_runs([str(x) for x in runs])
    resolution = resolution_from_promotions(
        diagnostics,
        combined_to_blend_promotion=promotions[("lgb_combined63", "blend")],
        raw_to_combined_promotion=promotions[("lgb_raw63", "lgb_combined63")],
    )

    submission = None
    route = resolution.get("route")
    if route == "FREEZE_DUAL_VIEW_BACKBONE_AND_RETRY_DIVERSITY":
        source = s3_root / "promotion__lgb_combined63__to__blend" / "submission_candidate_seedbag.csv"
        label = "s3_dualview_seedbag"
    elif route == "FREEZE_COMBINED_BACKBONE_RAW_HEDGE_NOT_PROMOTED":
        source = s3_root / "promotion__lgb_raw63__to__lgb_combined63" / "submission_candidate_seedbag.csv"
        label = "s3_combined_seedbag"
    else:
        source = None
        label = None

    if source is not None and source.exists():
        dst = out_root / "submission_s3.csv"
        shutil.copyfile(source, dst)
        _, test, _ = load_competition(data_dir)
        stats = validate_submission(pd.read_csv(dst), test)
        submission = {"label": label, "file": str(dst), "sha256": sha256_file(dst), **stats}
        atomic_write_json(out_root / "submission_s3.json", submission)

    summary = {
        "version": "nomophobia-v0.3-cpu-s3",
        "status": route or diagnostics.get("status"),
        "selected_iterations": selected,
        "tuning": tuning,
        "diagnostics": diagnostics,
        "resolution": resolution,
        "submission": submission,
        "input_hashes": input_hashes,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out_root / "cpu_s3_summary.json", summary)
    zip_base = out_root.parent / Path(a.zip_name).with_suffix("")
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out_root))
    print(json.dumps(summary, indent=2))
    print(f"\nSummary:    {out_root / 'cpu_s3_summary.json'}")
    if submission:
        print(f"Submission: {submission['file']}")
    print(f"Bundle:     {zip_path}")


if __name__ == "__main__":
    main()

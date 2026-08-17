#!/usr/bin/env python
"""Run the highest-value CPU research gates synchronously inside one Kaggle session.

The suite deliberately reuses the audited experiment scripts instead of duplicating
modeling code in a notebook. Expensive branches are gated by cheaper safety results.
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

from s6e8.artifacts import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "quick": {
        "stress": (60_000, 500, 50_000),
        "family": (40_000, 500, 3),
        "capacity": (60_000, [350, 600, 900]),
        "geometry": (40_000, 500, 3),
        "source": (40_000, 500, 3, [0.05, 0.10, 0.25]),
    },
    "balanced": {
        "stress": (90_000, 700, 80_000),
        "family": (60_000, 650, 3),
        "capacity": (100_000, [500, 800, 1200]),
        "geometry": (60_000, 650, 3),
        "source": (60_000, 650, 3, [0.05, 0.10, 0.25, 0.50]),
    },
    "thorough": {
        "stress": (120_000, 1000, 100_000),
        "family": (120_000, 1000, 5),
        "capacity": (180_000, [400, 700, 1000, 1400, 2000]),
        "geometry": (120_000, 1000, 5),
        "source": (120_000, 1000, 5, [0.05, 0.10, 0.25, 0.50, 1.0]),
    },
}


def run_step(name: str, cmd: list[str], expected: Path, *, cwd: Path) -> dict:
    print(f"\n{'=' * 88}\n{name}\n{'=' * 88}")
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
    if not expected.exists():
        raise FileNotFoundError(f"{name} did not create expected artifact: {expected}")
    payload = json.loads(expected.read_text())
    print(f"{name} finished in {elapsed / 60:.1f} minutes", flush=True)
    return {"elapsed_seconds": elapsed, "artifact": str(expected), "result": payload}


def hash_inputs_once(data_dir: Path, out_root: Path) -> dict:
    hashes = {}
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[name] = sha256_file(path)
    atomic_write_json(out_root / "input_hashes.json", hashes)
    return hashes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/kaggle/input/playground-series-s6e8")
    p.add_argument("--out-root", default="/kaggle/working/nomophobia_cpu_research")
    p.add_argument("--profile", choices=PROFILES, default="balanced")
    p.add_argument("--original-csv", default=None, help="Optional public 7,500-row source CSV")
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--threads", type=int, default=0, help="0 = keep environment/default LightGBM threading")
    p.add_argument("--skip-geometry", action="store_true")
    p.add_argument("--skip-source", action="store_true")
    p.add_argument("--skip-family", action="store_true")
    p.add_argument("--zip-name", default="nomophobia_cpu_research_artifacts.zip")
    a = p.parse_args()

    data_dir = Path(a.data_dir)
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if a.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
        os.environ["MKL_NUM_THREADS"] = str(a.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(a.threads)

    cfg = PROFILES[a.profile]
    py = sys.executable
    decision: dict = {
        "version": "nomophobia-v0.3-cpu-suite",
        "profile": a.profile,
        "seed": a.seed,
        "data_dir": str(data_dir),
        "steps": {},
        "started_at_unix": time.time(),
    }
    decision["input_hashes"] = hash_inputs_once(data_dir, out_root)

    # 1) Cheap safety gate first.
    rows, estimators, source_rows = cfg["stress"]
    stress_path = out_root / "frequency_stress.json"
    stress = run_step(
        "1/5 Frequency safety + transductive reference",
        [
            py,
            str(ROOT / "experiments" / "frequency_stress.py"),
            "--data-dir", str(data_dir),
            "--rows", str(rows),
            "--estimators", str(estimators),
            "--source-rows", str(source_rows),
            "--seed", str(a.seed),
            "--device", "cpu",
            "--out", str(stress_path),
            "--no-hash-inputs",
        ],
        stress_path,
        cwd=ROOT,
    )
    decision["steps"]["frequency_stress"] = stress
    stress_result = stress["result"]
    stop_density = bool(stress_result["decision"]["stop_density_expansion"])
    advance_tx = bool(stress_result["decision"]["advance_transductive_frequency"])

    # 2) Marginal-frequency decomposition only when the safety gate is not red.
    if not a.skip_family and not stop_density:
        rows, estimators, folds = cfg["family"]
        family_path = out_root / "frequency_family_ablation.json"
        family = run_step(
            "2/5 Marginal frequency family decomposition",
            [
                py,
                str(ROOT / "experiments" / "frequency_family_ablation.py"),
                "--data-dir", str(data_dir),
                "--rows", str(rows),
                "--estimators", str(estimators),
                "--folds", str(folds),
                "--bootstrap", "600" if folds == 3 else "1200",
                "--seed", str(a.seed),
                "--device", "cpu",
                "--out", str(family_path),
                "--no-hash-inputs",
            ],
            family_path,
            cwd=ROOT,
        )
        decision["steps"]["frequency_family_ablation"] = family
    else:
        decision["steps"]["frequency_family_ablation"] = {"skipped": True, "reason": "safety gate or user skip"}

    # 3) Raw/combined hedge question is always useful before spending S3 compute.
    rows, counts = cfg["capacity"]
    capacity_path = out_root / "capacity_diversity_curve.json"
    capacity = run_step(
        "3/5 Capacity-dependent raw/combined diversity",
        [
            py,
            str(ROOT / "experiments" / "capacity_diversity_curve.py"),
            "--data-dir", str(data_dir),
            "--rows", str(rows),
            "--estimators", *[str(x) for x in counts],
            "--raw-weight", "0.375",
            "--bootstrap", "700",
            "--seed", str(a.seed),
            "--device", "cpu",
            "--out", str(capacity_path),
            "--no-hash-inputs",
        ],
        capacity_path,
        cwd=ROOT,
    )
    decision["steps"]["capacity_diversity_curve"] = capacity

    # 4) Higher-order density is only worth paying for when the source-safety gate survives.
    geometry_advanced: list[str] = []
    if not a.skip_geometry and not stop_density and advance_tx:
        rows, estimators, folds = cfg["geometry"]
        geom_dir = out_root / "frequency_geometry"
        geom_path = geom_dir / "frequency_geometry.json"
        geometry = run_step(
            "4/5 Higher-order target-free frequency geometry",
            [
                py,
                str(ROOT / "experiments" / "frequency_geometry.py"),
                "--data-dir", str(data_dir),
                "--out-dir", str(geom_dir),
                "--rows", str(rows),
                "--estimators", str(estimators),
                "--folds", str(folds),
                "--bootstrap", "600" if folds == 3 else "1200",
                "--seed", str(a.seed),
                "--device", "cpu",
                "--no-hash-inputs",
            ],
            geom_path,
            cwd=ROOT,
        )
        decision["steps"]["frequency_geometry"] = geometry
        geometry_advanced = list(geometry["result"].get("advanced_arms", []))
    else:
        decision["steps"]["frequency_geometry"] = {
            "skipped": True,
            "reason": "transductive frequency did not advance, source-safety stop, or user skip",
        }

    # 5) Optional source-label augmentation. Auto-skip if the external source dataset is absent.
    source_advanced: list[str] = []
    original = Path(a.original_csv) if a.original_csv else None
    if not a.skip_source and original is not None and original.exists():
        rows, estimators, folds, weights = cfg["source"]
        src_dir = out_root / "source_row_augmentation"
        src_path = src_dir / "original_row_augmentation.json"
        source = run_step(
            "5/5 Low-weight labeled source-row augmentation",
            [
                py,
                str(ROOT / "experiments" / "original_row_augmentation.py"),
                "--data-dir", str(data_dir),
                "--original-csv", str(original),
                "--out-dir", str(src_dir),
                "--rows", str(rows),
                "--estimators", str(estimators),
                "--folds", str(folds),
                "--weights", *[str(x) for x in weights],
                "--bootstrap", "600" if folds == 3 else "1200",
                "--seed", str(a.seed),
                "--device", "cpu",
                "--no-hash-inputs",
            ],
            src_path,
            cwd=ROOT,
        )
        decision["steps"]["source_row_augmentation"] = source
        source_advanced = list(source["result"].get("advanced_weights", []))
    else:
        decision["steps"]["source_row_augmentation"] = {
            "skipped": True,
            "reason": "optional source CSV not attached or user skip",
        }

    cap_diag = capacity["result"].get("high_capacity_diagnosis", {})
    cap_route = cap_diag.get("route")
    if stop_density:
        route = "STOP_FREQUENCY_EXPANSION_AUDIT_SOURCE_SHIFT"
    elif geometry_advanced or source_advanced:
        route = "FREEZE_S3_BASELINE_AND_CONFIRM_NEW_CANDIDATES"
    elif cap_route == "DUAL_VIEW_STILL_PLAUSIBLE":
        route = "RUN_AUTHORITATIVE_S3_DUAL_VIEW"
    else:
        route = "RUN_S3_BASELINE_BUT_PRIORITIZE_NEW_DIVERSITY"

    decision.update(
        {
            "frequency_safety": {
                "advance_transductive_frequency": advance_tx,
                "stop_density_expansion": stop_density,
                "source_safety": stress_result.get("source_safety"),
            },
            "capacity_route": cap_route,
            "geometry_advanced_arms": geometry_advanced,
            "source_advanced_weights": source_advanced,
            "recommended_next_step": route,
            "finished_at_unix": time.time(),
        }
    )
    decision["elapsed_seconds"] = decision["finished_at_unix"] - decision["started_at_unix"]
    decision_path = out_root / "cpu_research_decision.json"
    atomic_write_json(decision_path, decision)

    # One compact artifact makes notebook chaining painless.
    zip_base = out_root.parent / Path(a.zip_name).with_suffix("")
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out_root))
    print("\nCPU research suite complete")
    print(json.dumps(decision, indent=2))
    print(f"\nDecision: {decision_path}")
    print(f"Bundle:   {zip_path}")


if __name__ == "__main__":
    main()

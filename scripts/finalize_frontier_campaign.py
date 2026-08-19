#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def honest_candidate_auc(report: dict) -> float:
    gate = report.get("residual_gate", {})
    metrics = gate.get("honest_metrics", {})
    if "candidate_auc" in metrics:
        return float(metrics["candidate_auc"])
    if "anchor_auc" in report:
        return float(report["anchor_auc"])
    standalone = report.get("standalone", {})
    if "anchor_auc" in standalone:
        return float(standalone["anchor_auc"])
    raise KeyError("report has no honest candidate or anchor AUC")


def load_report(directory: Path) -> dict | None:
    path = directory / "decision.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize a measured frontier residual campaign.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--contrast-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="artifacts/final")
    ap.add_argument("--composer-out", default="artifacts/composite")
    args = ap.parse_args()

    anchor_dir = Path(args.anchor_dir)
    anchor = json.loads((anchor_dir / "decision.json").read_text())
    anchor_auc = float(anchor["auc"]["honest_blend"])
    rows = [{"name": anchor_dir.name, "auc": anchor_auc, "accepted": True, "deploy_weight": 0.0}]
    accepted: list[Path] = []

    for text in args.contrast_dirs:
        directory = Path(text)
        report = load_report(directory)
        if report is None:
            rows.append({"name": directory.name, "status": "missing_or_failed"})
            continue
        gate = report["residual_gate"]
        row = {
            "name": directory.name,
            "auc": honest_candidate_auc(report),
            "accepted": bool(gate.get("accepted", False)),
            "deploy_weight": float(gate.get("deploy_weight", 0.0)),
        }
        if "standalone" in report:
            row["standalone_control"] = float(report["standalone"]["control_auc"])
            row["standalone_treatment"] = float(report["standalone"]["treatment_auc"])
            row["standalone_delta"] = float(report["standalone"]["treatment_minus_control"])
        rows.append(row)
        if row["accepted"]:
            accepted.append(directory)

    composite_dir: Path | None = None
    if len(accepted) >= 2:
        composite_dir = Path(args.composer_out)
        subprocess.run(
            [
                "python", "experiments/frontier_direction_composer.py",
                "--data-dir", args.data_dir,
                "--anchor-oof", str(anchor_dir / "oof.csv"),
                "--anchor-test", str(anchor_dir / "submission_blend.csv"),
                "--direction-dirs", *[str(directory) for directory in accepted],
                "--mode", "sequential_orthogonal",
                "--out-dir", str(composite_dir),
            ],
            check=True,
        )
        composite = load_report(composite_dir)
        assert composite is not None
        gate = composite["residual_gate"]
        rows.append(
            {
                "name": composite_dir.name,
                "auc": honest_candidate_auc(composite),
                "accepted": bool(gate.get("accepted", False)),
                "deploy_weight": float(gate.get("deploy_weight", 0.0)),
            }
        )
        if not gate.get("accepted", False):
            composite_dir = None

    candidates: list[tuple[str, float, Path]] = [
        (anchor_dir.name, anchor_auc, anchor_dir / "submission_blend.csv")
    ]
    for directory in accepted:
        report = load_report(directory)
        assert report is not None
        candidates.append(
            (directory.name, honest_candidate_auc(report), directory / "submission_gated.csv")
        )
    if composite_dir is not None:
        report = load_report(composite_dir)
        assert report is not None
        candidates.append(
            (composite_dir.name, honest_candidate_auc(report), composite_dir / "submission_gated.csv")
        )

    best = max(candidates, key=lambda item: item[1])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best[2], out / "submission_best.csv")
    summary = {
        "anchor_auc": anchor_auc,
        "measurements": rows,
        "accepted_directions": [directory.name for directory in accepted],
        "best": {
            "name": best[0],
            "honest_oof_auc": best[1],
            "source": str(best[2]),
        },
    }
    (out / "research_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

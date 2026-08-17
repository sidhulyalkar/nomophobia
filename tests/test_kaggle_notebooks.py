from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "kaggle" / "notebooks" / "01_nomophobia_cpu_research_suite.ipynb",
    ROOT / "kaggle" / "notebooks" / "02_nomophobia_cpu_s3_submission.ipynb",
]


def test_kaggle_cpu_notebooks_are_valid_json_and_python():
    for path in NOTEBOOKS:
        payload = json.loads(path.read_text())
        assert payload["nbformat"] == 4
        code_cells = [c for c in payload["cells"] if c.get("cell_type") == "code"]
        assert code_cells
        for i, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"{path.name}:cell-{i}")


def test_cpu_notebooks_use_one_consistent_offline_bundle_name():
    for path in NOTEBOOKS:
        text = path.read_text()
        assert "nomophobia-repo-source.zip" in text
        assert "nomophobia-source.zip" not in text


def test_cpu_notebook_contract_is_two_stage_and_cpu_only():
    first = NOTEBOOKS[0].read_text()
    second = NOTEBOOKS[1].read_text()
    assert "run_cpu_research_suite.py" in first
    assert "cpu_research_decision.json" in first
    assert "run_cpu_s3_campaign.py" in second
    assert "submission_s3.csv" in second
    assert '"--device", "cpu"' not in first  # child runner owns device routing
    assert "Accelerator:** None (CPU)" in first
    assert "Accelerator:** None (CPU)" in second

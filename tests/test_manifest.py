import json

from s6e8.manifest import ExperimentRecorder, data_provenance


def test_experiment_manifest_records_data_output_and_metrics(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    for name, content in {
        "train.csv": "id,x,addicted_label\n1,1,0\n",
        "test.csv": "id,x\n2,2\n",
        "sample_submission.csv": "id,addicted_label\n2,0.5\n",
    }.items():
        (data / name).write_text(content)
    output = tmp_path / "result.txt"; output.write_text("result")
    manifest_path = tmp_path / "manifest.json"
    with ExperimentRecorder(manifest_path, experiment_id="TEST", evidence_tier="S0", hypothesis="h", falsifier="f", accept_rule="a", kill_rule="k", data_dir=data, hash_inputs=True, repo_root=tmp_path) as recorder:
        recorder.add_metrics(score=0.9); recorder.add_output(output)
    payload = json.loads(manifest_path.read_text())
    assert payload["manifest_schema_version"] == 1
    assert payload["status"] == "COMPLETE"
    assert payload["metrics"]["score"] == 0.9
    assert payload["data"]["files"]["train.csv"]["sha256"]
    assert payload["outputs"][0]["sha256"]


def test_data_provenance_can_skip_hashing(tmp_path):
    (tmp_path / "train.csv").write_text("x\n1\n")
    payload = data_provenance(tmp_path, files=["train.csv"], hash_files=False)
    assert payload["files"]["train.csv"]["exists"]
    assert "sha256" not in payload["files"]["train.csv"]

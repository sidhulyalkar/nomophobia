import json
import pandas as pd
from s6e8.s3 import resolution_from_promotions, summarize_s3_runs


def _run(tmp_path, seed, auc, corr, raw_weights):
    run = tmp_path / f"seed_{seed}"; run.mkdir()
    (run / "run_summary.json").write_text(json.dumps({"fold_seed": seed, "blend_auc_honest": auc, "blend_selection_auc": auc + 0.0001, "blend_selection_optimism": 0.0001}))
    (run / "blend.json").write_text(json.dumps({"names": ["lgb_combined63", "lgb_raw63"], "weights": {"lgb_combined63": 0.65, "lgb_raw63": 0.35}, "rotation_weights": [[1-w, w] for w in raw_weights]}))
    pd.DataFrame([[1.0, corr], [corr, 1.0]], index=["lgb_combined63", "lgb_raw63"], columns=["lgb_combined63", "lgb_raw63"]).to_csv(run / "expert_rank_correlation.csv")
    return run


def test_s3_summary_reports_weight_stability_and_clear_status(tmp_path):
    runs = [_run(tmp_path,1,0.9600,0.984,[.30,.35,.40,.35,.35]), _run(tmp_path,2,0.9604,0.985,[.35,.40,.35,.40,.35]), _run(tmp_path,3,0.9602,0.986,[.35,.35,.40,.35,.30])]
    result = summarize_s3_runs(runs)
    assert result["status"] == "S3_DIAGNOSTICS_CLEAR"
    assert result["rotation_weight_stats"]["lgb_raw63"]["n"] == 15


def test_s3_summary_stops_on_correlation_and_resolution(tmp_path):
    runs = [_run(tmp_path,1,.9600,.989,[0,.4,.8,.2,.6]), _run(tmp_path,2,.9601,.990,[0,.4,.8,.2,.6]), _run(tmp_path,3,.9602,.991,[0,.4,.8,.2,.6])]
    diagnostics = summarize_s3_runs(runs)
    assert diagnostics["status"] == "STOP_AND_REPORT"
    blend = tmp_path / "blend.json"; representation = tmp_path / "rep.json"
    blend.write_text(json.dumps({"verdict":"PROMOTED"})); representation.write_text(json.dumps({"verdict":"PROMOTED"}))
    route = resolution_from_promotions(diagnostics, combined_to_blend_promotion=blend, raw_to_combined_promotion=representation)
    assert route["route"] == "STOP_AND_DIAGNOSE"

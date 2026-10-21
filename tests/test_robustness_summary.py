import json

import pytest

from wifi_csi_quality.robustness_summary import aggregate_model


def test_aggregate_model_uses_sample_standard_deviation(tmp_path):
    paths = []
    for seed, f1 in ((1, 0.6), (2, 0.8)):
        payload = {
            "model": "example",
            "replicate_seed": seed,
            "scenarios": {
                "clean": {
                    "sample_count": 10,
                    "metrics": {
                        "accuracy": f1,
                        "balanced_accuracy": f1,
                        "macro_f1": f1,
                        "ece": 0.1,
                        "nll": 1.0,
                    },
                }
            },
        }
        path = tmp_path / f"replicate_{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = aggregate_model(paths)

    summary = result["scenarios"]["clean"]["metrics"]["macro_f1"]
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["std"] == pytest.approx(0.1414213562)

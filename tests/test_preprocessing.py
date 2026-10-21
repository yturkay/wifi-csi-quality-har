import json

import numpy as np

from wifi_csi_quality.preprocessing import (
    PreprocessingConfig,
    _cache_relative_path,
    _load_toml,
    transform_sample,
)
from wifi_csi_quality.readers import CSISample


def test_transform_sample_has_fixed_shape_and_is_finite():
    times = np.arange(1000, dtype=np.float64) / 200.0
    base = np.sin(2 * np.pi * times)[:, None, None]
    values = (base + np.arange(30)[None, :, None] + 40.0) * np.ones((1, 1, 3))
    sample = CSISample(values=values, timestamps=times)

    tensor, diagnostics = transform_sample(sample, PreprocessingConfig())

    assert tensor.shape == (500, 64)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert diagnostics["source_time_steps"] == 1000
    assert diagnostics["right_edge_extrapolated_steps"] == 0


def test_transform_removes_duplicate_timestamps_and_tracks_edge_fill():
    times = np.array([0.0, 0.01, 0.01, 0.02])
    values = np.ones((4, 3, 1), dtype=np.float32)
    config = PreprocessingConfig(target_time_steps=5, target_sample_rate_hz=100)

    tensor, diagnostics = transform_sample(CSISample(values, times), config)

    assert tensor.shape == (5, 64)
    assert diagnostics["duplicate_or_invalid_timestamps_removed"] == 1
    assert diagnostics["right_edge_extrapolated_steps"] == 2


def test_cache_name_is_stable_and_does_not_expose_sample_id():
    record = {
        "dataset": "XRF55",
        "sample_id": "Scene_1:lb:01_31_01",
        "label": "wavinghand",
    }
    first = _cache_relative_path(record, "external_test")
    second = _cache_relative_path(json.loads(json.dumps(record)), "external_test")

    assert first == second
    assert record["sample_id"] not in first.name
    assert first.suffix == ".npz"


def test_protocol_toml_loader(tmp_path):
    path = tmp_path / "protocol.toml"
    path.write_text("[preprocessing]\ntarget_time_steps = 500\n", encoding="utf-8")

    loaded = _load_toml(path)

    assert loaded["preprocessing"]["target_time_steps"] == 500

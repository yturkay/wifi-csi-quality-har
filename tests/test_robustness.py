from wifi_csi_quality.robustness import deterministic_sample_seed


def test_sample_seed_is_stable_and_scenario_specific():
    first = deterministic_sample_seed("sample-1", "mixed_mild", 2026)
    second = deterministic_sample_seed("sample-1", "mixed_mild", 2026)
    other = deterministic_sample_seed("sample-1", "mixed_severe", 2026)

    assert first == second
    assert first != other
    assert 0 <= first < 2**32

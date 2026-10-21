"""Reproducible CSI quality-impairment simulation and quality features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .readers import CSISample


@dataclass(frozen=True)
class ImpairmentConfig:
    random_loss_probability: float = 0.0
    burst_start_probability: float = 0.0
    burst_length: int = 1
    jitter_std_seconds: float = 0.0
    jitter_std_fraction_of_median_interval: float = 0.0
    stale_start_probability: float = 0.0
    stale_length: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "random_loss_probability",
            "burst_start_probability",
            "stale_start_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.burst_length < 1 or self.stale_length < 1:
            raise ValueError("burst_length and stale_length must be positive")
        if self.jitter_std_seconds < 0 or self.jitter_std_fraction_of_median_interval < 0:
            raise ValueError("jitter standard deviations cannot be negative")


@dataclass(frozen=True)
class QualityView:
    values: Any
    timestamps: Any
    observed: Any
    stale: Any
    source_index: Any
    age_seconds: Any
    delta_t_seconds: Any
    metadata: dict[str, Any]

    @property
    def quality_features(self) -> Any:
        import numpy as np

        return np.column_stack(
            (
                self.observed.astype(np.float32),
                self.stale.astype(np.float32),
                np.nan_to_num(self.delta_t_seconds, nan=0.0).astype(np.float32),
                np.nan_to_num(self.age_seconds, nan=0.0).astype(np.float32),
            )
        )


def _start_windows(mask: Any, starts: Any, length: int) -> None:
    for start in starts:
        mask[start : min(start + length, mask.size)] = True


def simulate_impairments(sample: CSISample, config: ImpairmentConfig) -> QualityView:
    """Apply loss, timing jitter, and sample staleness without changing input."""
    import numpy as np

    rng = np.random.default_rng(config.seed)
    values = np.array(sample.values, copy=True)
    timestamps = np.asarray(sample.timestamps, dtype=np.float64).copy()
    count = values.shape[0]
    observed = rng.random(count) >= config.random_loss_probability

    burst_mask = np.zeros(count, dtype=bool)
    burst_starts = np.flatnonzero(rng.random(count) < config.burst_start_probability)
    _start_windows(burst_mask, burst_starts, config.burst_length)
    observed &= ~burst_mask

    if count > 1 and (
        config.jitter_std_seconds > 0
        or config.jitter_std_fraction_of_median_interval > 0
    ):
        intervals = np.diff(timestamps)
        positive = intervals[intervals > 0]
        median_interval = float(np.median(positive)) if positive.size else 1.0
        floor = median_interval * 1e-3
        jitter_std = config.jitter_std_seconds + (
            config.jitter_std_fraction_of_median_interval * median_interval
        )
        jitter = rng.normal(0.0, jitter_std, size=intervals.size)
        intervals = np.maximum(intervals + jitter, floor)
        timestamps[1:] = timestamps[0] + np.cumsum(intervals)

    stale = np.zeros(count, dtype=bool)
    stale_starts = np.flatnonzero(rng.random(count) < config.stale_start_probability)
    _start_windows(stale, stale_starts, config.stale_length)
    source_index = np.arange(count, dtype=np.int64)
    for start in stale_starts:
        if start == 0:
            continue
        end = min(start + config.stale_length, count)
        source_index[start:end] = start - 1
    stale &= source_index != np.arange(count)
    values = values[source_index]

    missing_value = complex(float("nan"), float("nan")) if np.iscomplexobj(values) else np.nan
    values[~observed] = missing_value
    source_index[~observed] = -1

    age_seconds = np.full(count, np.nan, dtype=np.float64)
    valid = source_index >= 0
    age_seconds[valid] = np.maximum(
        0.0, timestamps[valid] - timestamps[source_index[valid]]
    )
    delta_t = np.full(count, np.nan, dtype=np.float64)
    if count > 1:
        delta_t[1:] = np.diff(timestamps)

    return QualityView(
        values=values,
        timestamps=timestamps,
        observed=observed,
        stale=stale,
        source_index=source_index,
        age_seconds=age_seconds,
        delta_t_seconds=delta_t,
        metadata={
            **sample.metadata,
            "impairment_config": config.__dict__,
            "observed_fraction": float(observed.mean()) if count else 0.0,
            "stale_fraction": float(stale.mean()) if count else 0.0,
        },
    )


def forward_fill(view: QualityView, fill_value: float = 0.0) -> Any:
    """Forward-fill missing packets; leading missing packets use fill_value."""
    import numpy as np

    output = np.array(view.values, copy=True)
    last = None
    for index, is_observed in enumerate(view.observed):
        if is_observed:
            last = np.array(output[index], copy=True)
        elif last is None:
            output[index] = fill_value
        else:
            output[index] = last
    return output


def is_clean(config: ImpairmentConfig) -> bool:
    """Return whether a configuration leaves values and timing unchanged."""
    return (
        config.random_loss_probability == 0.0
        and config.burst_start_probability == 0.0
        and config.jitter_std_seconds == 0.0
        and config.jitter_std_fraction_of_median_interval == 0.0
        and config.stale_start_probability == 0.0
    )


def recover_regular_grid(
    view: QualityView, target_timestamps: Any, fill_value: float = 0.0
) -> Any:
    """Forward-fill loss and resample an impaired stream to a regular grid."""
    import numpy as np

    filled = np.asarray(forward_fill(view, fill_value=fill_value))
    target = np.asarray(target_timestamps, dtype=np.float64)
    source = np.asarray(view.timestamps, dtype=np.float64)
    flattened = filled.reshape(filled.shape[0], -1)
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 1, source.size - 1)
    left = right - 1
    denominator = source[right] - source[left]
    weight = ((target - source[left]) / denominator)[:, None]
    recovered = flattened[left] * (1.0 - weight) + flattened[right] * weight
    recovered[target <= source[0]] = flattened[0]
    recovered[target >= source[-1]] = flattened[-1]
    return recovered.reshape((target.size, *filled.shape[1:])).astype(
        filled.dtype, copy=False
    )


def impair_regular_tensor_with_quality(
    values: Any,
    config: ImpairmentConfig,
    sample_rate_hz: float = 100.0,
    fill_value: float = 0.0,
) -> tuple[Any, Any, dict[str, float]]:
    """Impair a tensor and return recovered values plus aligned quality channels."""
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected [time, feature] values, got {array.shape}")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if is_clean(config):
        quality = np.column_stack(
            (
                np.ones(array.shape[0], dtype=np.float32),
                np.zeros(array.shape[0], dtype=np.float32),
                np.ones(array.shape[0], dtype=np.float32),
                np.zeros(array.shape[0], dtype=np.float32),
            )
        )
        return np.array(array, copy=True), quality, {
            "observed_fraction": 1.0,
            "stale_fraction": 0.0,
            "mean_abs_timing_offset_seconds": 0.0,
            "max_abs_timing_offset_seconds": 0.0,
        }
    nominal_timestamps = np.arange(array.shape[0], dtype=np.float64) / sample_rate_hz
    sample = CSISample(array[:, :, None], nominal_timestamps, {})
    view = simulate_impairments(sample, config)
    recovered = recover_regular_grid(view, nominal_timestamps, fill_value=fill_value)
    previous_event = np.searchsorted(
        view.timestamps, nominal_timestamps, side="right"
    ) - 1
    previous_event = np.clip(previous_event, 0, view.timestamps.size - 1)
    effective_source = np.asarray(view.source_index, dtype=np.int64).copy()
    last_source = -1
    for index, source_index in enumerate(effective_source):
        if source_index >= 0:
            last_source = int(source_index)
        else:
            effective_source[index] = last_source
    effective_age = np.zeros(view.timestamps.size, dtype=np.float64)
    valid_source = effective_source >= 0
    effective_age[valid_source] = np.maximum(
        0.0,
        view.timestamps[valid_source] - view.timestamps[effective_source[valid_source]],
    )
    nominal_interval = 1.0 / sample_rate_hz
    delta = np.nan_to_num(
        view.delta_t_seconds, nan=nominal_interval, posinf=nominal_interval
    )
    quality = np.column_stack(
        (
            view.observed[previous_event].astype(np.float32),
            view.stale[previous_event].astype(np.float32),
            np.clip(delta[previous_event] / nominal_interval, 0.0, 10.0).astype(
                np.float32
            ),
            np.clip(
                effective_age[previous_event] / nominal_interval, 0.0, 10.0
            ).astype(np.float32),
        )
    )
    timing_offset = np.abs(view.timestamps - nominal_timestamps)
    diagnostics = {
        "observed_fraction": float(view.observed.mean()),
        "stale_fraction": float(view.stale.mean()),
        "mean_abs_timing_offset_seconds": float(timing_offset.mean()),
        "max_abs_timing_offset_seconds": float(timing_offset.max()),
    }
    return recovered[:, :, 0].astype(array.dtype, copy=False), quality, diagnostics


def impair_regular_tensor(
    values: Any,
    config: ImpairmentConfig,
    sample_rate_hz: float = 100.0,
    fill_value: float = 0.0,
) -> tuple[Any, dict[str, float]]:
    """Impair a ``[time, feature]`` tensor and return it on its original grid."""
    recovered, _, diagnostics = impair_regular_tensor_with_quality(
        values,
        config,
        sample_rate_hz=sample_rate_hz,
        fill_value=fill_value,
    )
    return recovered, diagnostics


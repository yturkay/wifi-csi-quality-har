"""Dataset readers that map heterogeneous CSI files to one canonical layout.

Canonical CSI shape is ``[time, subcarrier, link]``. Timestamps are relative
seconds and start at zero. Readers never modify their source files.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO


@dataclass(frozen=True)
class CSISample:
    """One CSI sequence in a dataset-independent representation."""

    values: Any
    timestamps: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if getattr(self.values, "ndim", None) != 3:
            raise ValueError("CSI values must have shape [time, subcarrier, link]")
        if getattr(self.timestamps, "ndim", None) != 1:
            raise ValueError("timestamps must be one-dimensional")
        if self.values.shape[0] != self.timestamps.shape[0]:
            raise ValueError("time dimension and timestamp count do not match")

    @property
    def amplitude(self) -> Any:
        import numpy as np

        return np.abs(self.values)

    @property
    def phase(self) -> Any:
        import numpy as np

        if not np.iscomplexobj(self.values):
            raise ValueError("phase is unavailable for amplitude-only CSI")
        return np.angle(self.values)


def _relative_seconds(raw: Any, scale: float = 1e-6, wrap: int | None = None) -> Any:
    import numpy as np

    original = np.asarray(raw).reshape(-1)
    if original.size == 0:
        return np.empty(0, dtype=np.float64)
    if wrap is not None:
        unsigned = original.astype(np.uint32).astype(np.uint64)
        jumps = np.diff(unsigned.astype(np.int64)) < 0
        offsets = np.concatenate(([0], np.cumsum(jumps, dtype=np.uint64))) * np.uint64(wrap)
        unwrapped = unsigned + offsets
        return (unwrapped - unwrapped[0]).astype(np.float64) * scale
    values = original.astype(np.float64)
    return (values - values[0]) * scale


def read_csi_bench_h5(
    path: str | Path, dataset: str = "CSI_amps", sample_rate_hz: float = 100.0
) -> CSISample:
    """Read a processed CSI-Bench H5 sample (amplitude only)."""
    import h5py
    import numpy as np

    path = Path(path)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"H5 dataset {dataset!r} not found in {path}")
        raw = np.asarray(handle[dataset])
    if raw.ndim == 2:
        raw = raw[:, :, None]
    if raw.ndim != 3:
        raise ValueError(f"Unexpected CSI-Bench shape {raw.shape} in {path}")
    # Published files use [subcarrier, time, link].
    values = np.transpose(raw, (1, 0, 2))
    # H5 files do not retain packet timestamps. The released 500-point HAR
    # windows are sampled at 100 Hz; callers can override this when needed.
    timestamps = np.arange(values.shape[0], dtype=np.float64) / sample_rate_hz
    return CSISample(
        values=values,
        timestamps=timestamps,
        metadata={
            "dataset": "CSI-Bench",
            "format": "h5-amplitude",
            "path": str(path),
            "sample_rate_hz": sample_rate_hz,
            "timestamps_inferred": True,
        },
    )


def read_csi_bench_mat(path: str | Path, timer_scale: float = 1e-6) -> CSISample:
    """Read a raw CSI-Bench MATLAB recording with packet timestamps."""
    import numpy as np
    from scipy.io import loadmat

    path = Path(path)
    trace = loadmat(path, squeeze_me=True, struct_as_record=False).get("csi_trace")
    if trace is None or not hasattr(trace, "csi") or not hasattr(trace, "mactimer"):
        raise ValueError(f"Expected csi_trace.csi and csi_trace.mactimer in {path}")
    csi = np.asarray(trace.csi)
    timer = np.asarray(trace.mactimer).reshape(-1)
    axes = [axis for axis, size in enumerate(csi.shape) if size == timer.size]
    if not axes:
        raise ValueError(f"No CSI axis matches {timer.size} timestamps in shape {csi.shape}")
    time_first = np.moveaxis(csi, axes[-1], 0)
    values = time_first.reshape(time_first.shape[0], -1, 1)
    timestamps = _relative_seconds(timer, scale=timer_scale, wrap=2**32)
    return CSISample(
        values=values,
        timestamps=timestamps,
        metadata={"dataset": "CSI-Bench", "format": "mat-raw", "path": str(path)},
    )


def _mat_struct_items(data: Any) -> list[Any]:
    import numpy as np

    array = np.asarray(data, dtype=object).reshape(-1)
    return list(array)


def read_xrf55_mat(path: str | Path, timer_scale: float = 1e-6) -> CSISample:
    """Read an XRF55 MATLAB sample containing Intel 5300 packet structs."""
    import numpy as np
    from scipy.io import loadmat

    path = Path(path)
    data = loadmat(path, squeeze_me=True, struct_as_record=False).get("data")
    if data is None:
        raise ValueError(f"Expected variable 'data' in {path}")
    packets = _mat_struct_items(data)
    if not packets:
        raise ValueError(f"No packets in {path}")
    csi = np.stack([np.asarray(packet.csi).T for packet in packets], axis=0)
    raw_timestamps = np.asarray([packet.timestamp_low for packet in packets])
    timestamps = _relative_seconds(raw_timestamps, scale=timer_scale, wrap=2**32)
    return CSISample(
        values=csi,
        timestamps=timestamps,
        metadata={
            "dataset": "XRF55",
            "format": "mat-intel5300",
            "path": str(path),
            "rssi_a": np.asarray([packet.rssi_a for packet in packets]),
            "rssi_b": np.asarray([packet.rssi_b for packet in packets]),
            "rssi_c": np.asarray([packet.rssi_c for packet in packets]),
        },
    )


def _signed_byte(value: int) -> int:
    value &= 0xFF
    return value - 256 if value >= 128 else value


def _decode_intel5300_payload(payload: bytes) -> dict[str, Any]:
    """Decode one Intel 5300 beamforming feedback payload."""
    import numpy as np

    if len(payload) < 20:
        raise ValueError("Intel 5300 payload is shorter than 20-byte header")
    timestamp_low, bfee_count = struct.unpack_from("<IH", payload, 0)
    nrx, ntx = payload[8], payload[9]
    if nrx < 1 or ntx < 1:
        raise ValueError(f"Invalid antenna counts Nrx={nrx}, Ntx={ntx}")
    antenna_sel = payload[15]
    csi_len, rate = struct.unpack_from("<HH", payload, 16)
    expected_len = (30 * (nrx * ntx * 8 * 2 + 3) + 7) // 8
    if csi_len != expected_len or len(payload) < 20 + csi_len:
        raise ValueError(
            f"CSI payload length mismatch: declared={csi_len}, expected={expected_len}"
        )
    data = payload[20 : 20 + csi_len]
    csi = np.empty((30, nrx * ntx), dtype=np.complex64)
    bit_index = 0
    for subcarrier in range(30):
        bit_index += 3
        for stream in range(nrx * ntx):
            byte_index, remainder = divmod(bit_index, 8)
            real_raw = (data[byte_index] >> remainder) | (
                data[byte_index + 1] << (8 - remainder)
            )
            bit_index += 8
            byte_index, remainder = divmod(bit_index, 8)
            imag_raw = (data[byte_index] >> remainder) | (
                data[byte_index + 1] << (8 - remainder)
            )
            bit_index += 8
            csi[subcarrier, stream] = complex(
                _signed_byte(real_raw), _signed_byte(imag_raw)
            )
    return {
        "timestamp_low": timestamp_low,
        "bfee_count": bfee_count,
        "nrx": nrx,
        "ntx": ntx,
        "rssi_a": payload[10],
        "rssi_b": payload[11],
        "rssi_c": payload[12],
        "noise": _signed_byte(payload[13]),
        "agc": payload[14],
        "perm": (
            (antenna_sel & 0x3) + 1,
            ((antenna_sel >> 2) & 0x3) + 1,
            ((antenna_sel >> 4) & 0x3) + 1,
        ),
        "rate": rate,
        "csi": csi,
    }


def _read_intel5300_records(handle: BinaryIO) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    while True:
        length_bytes = handle.read(2)
        if not length_bytes:
            break
        if len(length_bytes) != 2:
            raise ValueError("Truncated record length at end of file")
        field_len = int.from_bytes(length_bytes, byteorder="big")
        if field_len < 1:
            raise ValueError(f"Invalid Intel 5300 field length {field_len}")
        code = handle.read(1)
        payload = handle.read(field_len - 1)
        if len(code) != 1 or len(payload) != field_len - 1:
            raise ValueError("Truncated Intel 5300 record")
        if code[0] == 0xBB:
            records.append(_decode_intel5300_payload(payload))
    return records


def read_xrf55_dat(path: str | Path, timer_scale: float = 1e-6) -> CSISample:
    """Read an XRF55 Intel 5300 ``.dat`` sample."""
    import numpy as np

    path = Path(path)
    with path.open("rb") as handle:
        packets = _read_intel5300_records(handle)
    if not packets:
        raise ValueError(f"No Intel 5300 CSI records found in {path}")
    shapes = {packet["csi"].shape for packet in packets}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent CSI shapes in {path}: {sorted(shapes)}")
    values = np.stack([packet["csi"] for packet in packets], axis=0)
    timestamps = _relative_seconds(
        [packet["timestamp_low"] for packet in packets], scale=timer_scale, wrap=2**32
    )
    return CSISample(
        values=values,
        timestamps=timestamps,
        metadata={
            "dataset": "XRF55",
            "format": "dat-intel5300",
            "path": str(path),
            "rssi_a": np.asarray([packet["rssi_a"] for packet in packets]),
            "rssi_b": np.asarray([packet["rssi_b"] for packet in packets]),
            "rssi_c": np.asarray([packet["rssi_c"] for packet in packets]),
        },
    )


def read_xrf55(path: str | Path) -> CSISample:
    """Dispatch to the XRF55 reader matching the file extension."""
    path = Path(path)
    if path.suffix.lower() == ".dat":
        return read_xrf55_dat(path)
    if path.suffix.lower() == ".mat":
        return read_xrf55_mat(path)
    raise ValueError(f"Unsupported XRF55 extension: {path.suffix}")


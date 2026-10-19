
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOUR_NS = 3_600 * 10**9
PARTITIONS = ("train", "validation", "test")


@dataclass
class WindowSet:
    partition: str
    X: np.ndarray  
    Y_scaled: np.ndarray  
    Y: np.ndarray  
    origin_rows: np.ndarray  
    origin_times: np.ndarray  
    candidate_origins: int  

    @property
    def n(self) -> int:
        return int(self.origin_rows.size)

    def target_times(self) -> np.ndarray:
        return self.origin_times[:, None] + np.arange(self.Y.shape[1]) * np.timedelta64(1, "h")


def timestamps_ns(frame: pd.DataFrame) -> np.ndarray:
    return frame["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)


def consecutive_run_lengths(ts_int: np.ndarray) -> np.ndarray:
    """run[i] = number of consecutive hourly timestamps ending at row i."""
    n = ts_int.size
    breaks = np.ones(n, dtype=bool)
    breaks[1:] = np.diff(ts_int) != HOUR_NS
    start = np.maximum.accumulate(np.where(breaks, np.arange(n), 0))
    return np.arange(n) - start + 1


def select_origin_rows(ts_int: np.ndarray, run_len: np.ndarray, first_row: int, end_row: int, input_hours: int,
                       horizon: int, stride_hours: int, block_start_hour: int,
                       inputs_from_row: int = 0) -> tuple[np.ndarray, int]:
    """Origins s with targets in [first_row, end_row) and inputs starting at or after inputs_from_row."""
    low = max(first_row, inputs_from_row + input_hours)
    high = end_row - horizon
    if high < low:
        return np.empty(0, dtype=np.int64), 0
    rows = np.arange(low, high + 1)
    hour_of_day = (ts_int[rows] // HOUR_NS) % 24
    aligned = ((hour_of_day - block_start_hour) % stride_hours) == 0
    consecutive = run_len[rows + horizon - 1] >= input_hours + horizon
    return rows[aligned & consecutive].astype(np.int64), int(aligned.sum())


def build_window_sets(clean: pd.DataFrame, features: list[str], target: str, feature_scaler, target_scaler,
                      train_end: int, val_end: int, input_hours: int, horizon: int, train_stride_hours: int,
                      block_start_hour: int) -> dict[str, WindowSet]:
    x_scaled = feature_scaler.transform(clean[features].to_numpy(dtype=float)).astype(np.float32)
    y_phys = clean[target].to_numpy(dtype=float)
    y_scaled = target_scaler.transform(y_phys.reshape(-1, 1)).ravel()
    ts_int = timestamps_ns(clean)
    run_len = consecutive_run_lengths(ts_int)
    
    specs = {"train": (0, train_end, train_stride_hours),
             "validation": (train_end, val_end, 24),
             "test": (val_end, len(clean), 24)}
    sets = {}
    for partition, (start, end, stride) in specs.items():
        rows, candidates = select_origin_rows(ts_int, run_len, start, end, input_hours, horizon, stride,
                                              block_start_hour)
        if rows.size:
            in_idx = rows[:, None] + np.arange(-input_hours, 0)
            out_idx = rows[:, None] + np.arange(horizon)
            X, Ys, Y = x_scaled[in_idx], y_scaled[out_idx].astype(np.float32), y_phys[out_idx]
        else:
            X = np.empty((0, input_hours, len(features)), np.float32)
            Ys, Y = np.empty((0, horizon), np.float32), np.empty((0, horizon))
        sets[partition] = WindowSet(partition, X, Ys, Y, rows, ts_int[rows].astype("datetime64[ns]"), candidates)
    return sets


def verify_window_sets(sets: dict[str, WindowSet], clean: pd.DataFrame, train_end: int, val_end: int,
                       input_hours: int, horizon: int, block_start_hour: int) -> list[dict]:
    """Independent checks; raises AssertionError on the first failure, returns the check log."""
    ts_int = timestamps_ns(clean)
    bounds = {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, len(clean))}
    log = []

    def check(name, ok, partition, detail=""):
        log.append({"check": name, "partition": partition, "status": "ok" if ok else "FAILED", "detail": detail})
        if not ok:
            raise AssertionError(f"Window check failed: {name} ({partition}). {detail}")

    for partition, ws in sets.items():
        start, end = bounds[partition]
        check("windows_exist", ws.n > 0, partition,
              "no valid windows; inspect unresolved gaps in the preprocessing report")
        check("input_shape_168_hours", ws.X.shape[1] == input_hours, partition, str(ws.X.shape))
        check("exactly_24_targets", ws.Y.shape == (ws.n, horizon) and ws.Y_scaled.shape == (ws.n, horizon),
              partition, str(ws.Y.shape))
        rows = ws.origin_rows[:, None] + np.arange(-input_hours, horizon)
        check("192_consecutive_hours", bool(np.all(np.diff(ts_int[rows], axis=1) == HOUR_NS)), partition,
              "history ends exactly 1 h before the first target hour")
        check("targets_inside_partition", bool(np.all(ws.origin_rows >= start) and
                                               np.all(ws.origin_rows + horizon <= end)), partition)
        if partition == "train":
            check("training_inputs_inside_training_partition", bool(np.all(ws.origin_rows - input_hours >= 0)),
                  partition)
        else:
            hours = (ts_int[ws.origin_rows] // HOUR_NS) % 24
            check("daily_block_alignment", bool(np.all(hours == block_start_hour)), partition,
                  f"first target hour {block_start_hour:02d}:00")
            check("non_overlapping_chronological_blocks",
                  bool(np.all(np.diff(ws.origin_times.astype(np.int64)) >= horizon * HOUR_NS)), partition)
    return log


def window_index_frame(sets: dict[str, WindowSet], input_hours: int) -> pd.DataFrame:
    frames = []
    for partition, ws in sets.items():
        origin = pd.to_datetime(ws.origin_times)
        frames.append(pd.DataFrame({
            "partition": partition, "origin_row": ws.origin_rows,
            "input_first_hour": origin - pd.Timedelta(hours=input_hours),
            "input_last_hour": origin - pd.Timedelta(hours=1),
            "target_first_hour": origin, "target_last_hour": origin + pd.Timedelta(hours=ws.Y.shape[1] - 1),
        }))
    return pd.concat(frames, ignore_index=True)


def window_summary(sets: dict[str, WindowSet], train_end: int, val_end: int, input_hours: int) -> dict:
    out = {}
    for partition, ws in sets.items():
        first = pd.Timestamp(ws.origin_times[0]) if ws.n else None
        out[partition] = {"windows": ws.n, "candidate_origins": ws.candidate_origins,
                          "skipped_because_not_192_consecutive_hours": ws.candidate_origins - ws.n,
                          "first_origin": str(first) if first is not None else None,
                          "last_origin": str(pd.Timestamp(ws.origin_times[-1])) if ws.n else None}
    for partition, start, source in (("validation", train_end, "training"), ("test", val_end, "validation")):
        ws = sets[partition]
        if ws.n:
            from_source = max(0, input_hours - int(ws.origin_rows[0] - start))
            out[partition]["first_window_context"] = (
                f"first origin {pd.Timestamp(ws.origin_times[0])}: {from_source} of its {input_hours} input hours "
                f"come from the {source} partition; all inputs precede the origin and all targets lie in the "
                f"{partition} partition")
    return out


def save_window_sets(sets: dict[str, WindowSet], path: Path) -> None:
    payload = {}
    for partition, ws in sets.items():
        payload.update({f"{partition}_X": ws.X, f"{partition}_Y_scaled": ws.Y_scaled, f"{partition}_Y": ws.Y,
                        f"{partition}_origin_rows": ws.origin_rows,
                        f"{partition}_origin_ns": ws.origin_times.astype(np.int64),
                        f"{partition}_candidates": np.asarray(ws.candidate_origins)})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_window_sets(path: Path) -> dict[str, WindowSet]:
    with np.load(path) as data:
        return {p: WindowSet(p, data[f"{p}_X"], data[f"{p}_Y_scaled"], data[f"{p}_Y"], data[f"{p}_origin_rows"],
                             data[f"{p}_origin_ns"].astype("datetime64[ns]"), int(data[f"{p}_candidates"]))
                for p in PARTITIONS}

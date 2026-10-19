

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

METRIC_DEFINITIONS = {
    "MAE": "mean absolute error, (1/n) sum |y - y_hat|",
    "MSE": "mean squared error, (1/n) sum (y - y_hat)^2",
    "Improvement_pct": "(E_baseline - E_DALF) / E_baseline x 100",
}


def compute_metrics(actual, predicted) -> dict:
    a = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if a.shape != p.shape:
        raise ValueError(f"actual {a.shape} and predicted {p.shape} are not aligned")
    if a.size == 0:
        raise ValueError("metrics need at least one value")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p))):
        raise ValueError("metric inputs must be finite")
    error = a - p
    mse = float(np.mean(error ** 2))
    return {"MAE": float(np.mean(np.abs(error))), "MSE": mse }


def metric_unit(metric: str, unit: str) -> str:
    if unit == "dimensionless":
        return unit
    return f"({unit})^2" if metric == "MSE" else unit


def timestamp_signature(times) -> str:
    """Short hash of the evaluated timestamps; equal signatures mean identical evaluation sets."""
    values = np.asarray(pd.to_datetime(np.asarray(times).reshape(-1)), dtype="datetime64[ns]").astype(np.int64)
    return hashlib.sha1(values.tobytes()).hexdigest()[:16]


def aggregate_daily_energy(hourly_blocks, power_measurement: str, horizon_hours: int = 24) -> np.ndarray:
    """Daily energy from one calendar day of hourly values per row.

    hourly_mean_power (kW): E = sum_h P_h x 1 h  -> kWh
    hourly_energy (kWh):   E = sum_h E_h
    instantaneous readings are not integrated (no valid conversion), so they raise.
    """
    blocks = np.asarray(hourly_blocks, dtype=float)
    if blocks.ndim != 2 or blocks.shape[1] != horizon_hours or horizon_hours != 24:
        raise ValueError(f"daily energy needs complete 24-hour blocks; got shape {blocks.shape}")
    if power_measurement not in ("hourly_mean_power", "hourly_energy"):
        raise ValueError("daily energy cannot be computed from instantaneous power readings")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("daily energy inputs must be finite")
    return blocks.sum(axis=1) * 1.0  # x 1 h per value for mean power; plain sum for hourly energy


def metric_rows(identity: dict, actual, predicted, unit: str, scale: str, resolution: str, times) -> list[dict]:
    metrics = compute_metrics(actual, predicted)
    signature = timestamp_signature(times)
    n = int(np.asarray(actual).size)
    return [{**identity, "resolution": resolution, "scale": scale, "metric": name, "value": value,
             "unit": metric_unit(name, unit), "n_values": n, "timestamp_signature": signature}
            for name, value in metrics.items()]


def relative_improvement(e_baseline: float, e_candidate: float) -> float:
    if not (np.isfinite(e_baseline) and np.isfinite(e_candidate)):
        raise ValueError("improvement needs finite errors")
    if e_baseline == 0:
        return float("nan")
    return (e_baseline - e_candidate) / e_baseline * 100.0


def improvement_table(metrics: pd.DataFrame, candidate: str = "DALF", baselines=None) -> pd.DataFrame:
    """Improvement of the candidate over each baseline, validated for identical evaluation sets."""
    keys = ["plant", "target", "resolution", "scale", "metric"]
    rows = []
    cand = metrics[metrics["model"] == candidate]
    for _, c in cand.iterrows():
        mask = np.ones(len(metrics), dtype=bool)
        for key in keys:
            mask &= (metrics[key] == c[key]).to_numpy()
        others = metrics[mask & (metrics["model"] != candidate).to_numpy()]
        if baselines is not None:
            others = others[others["model"].isin(baselines)]
        for _, b in others.iterrows():
            if b["timestamp_signature"] != c["timestamp_signature"] or b["n_values"] != c["n_values"] \
                    or b["unit"] != c["unit"]:
                raise ValueError(f"cannot compare {candidate} with {b['model']}: different evaluation sets or units "
                                 f"({c['plant']}, {c['target']}, {c['resolution']}, {c['metric']})")
            rows.append({**{k: c[k] for k in keys}, "unit": c["unit"], "baseline": b["model"],
                         "baseline_value": b["value"], "candidate": candidate, "candidate_value": c["value"],
                         "improvement_pct": relative_improvement(b["value"], c["value"]),
                         "timestamp_signature": c["timestamp_signature"], "n_values": int(c["n_values"])})
    return pd.DataFrame(rows)

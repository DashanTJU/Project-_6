"""Fusion of the HCRM, HCGM and HCLM forecasts: DALF, SCAF and UEBF."""


from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize

HOUR = np.timedelta64(1, "h")


def uniform_weights(k: int) -> np.ndarray:
    return np.full(k, 1.0 / k)


def weights_are_valid(weights, k: int, tolerance: float = 1e-8) -> bool:
    if weights is None:
        return False
    w = np.asarray(weights, dtype=float)
    return (w.shape == (k,) and bool(np.all(np.isfinite(w))) and w.min() >= -tolerance
            and abs(w.sum() - 1.0) <= tolerance)


def _normalise(w) -> np.ndarray | None:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    return w / total if np.isfinite(total) and total > 0 else None


def objective_value(weights: np.ndarray, actual: np.ndarray, predictions: np.ndarray, objective: str) -> float:
    residual = actual - predictions @ weights
    return float(np.mean(np.abs(residual)) if objective == "mae" else np.mean(residual ** 2))


def solve_mae_lp(actual: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray | None, str]:
    """min (1/n) sum t_i  s.t.  -t <= y - P w <= t,  w >= 0,  sum(w) = 1."""
    n, k = predictions.shape
    identity = sparse.identity(n, format="csr")
    P = sparse.csr_matrix(predictions)
    A_ub = sparse.vstack([sparse.hstack([P, -identity]), sparse.hstack([-P, -identity])], format="csr")
    b_ub = np.concatenate([actual, -actual])
    c = np.concatenate([np.zeros(k), np.full(n, 1.0 / n)])
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(k), np.zeros(n)])[None, :])
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                     bounds=[(0.0, 1.0)] * k + [(0.0, None)] * n, method="highs")
    if result.status != 0 or result.x is None:
        return None, f"linprog status {result.status}: {result.message}"
    return _normalise(result.x[:k]), f"linprog: {result.message}"


def solve_slsqp(actual: np.ndarray, predictions: np.ndarray, objective: str) -> tuple[np.ndarray | None, str]:
    """SLSQP from the uniform start and each vertex; the lowest objective wins."""
    k = predictions.shape[1]
    starts = [uniform_weights(k)] + list(np.eye(k))
    best, messages = None, []
    for start in starts:
        result = minimize(objective_value, start, args=(actual, predictions, objective), method="SLSQP",
                          bounds=[(0.0, 1.0)] * k, constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                          options={"maxiter": 500, "ftol": 1e-12})
        messages.append(str(result.message))
        w = _normalise(result.x) if result.success else None
        if w is not None:
            value = objective_value(w, actual, predictions, objective)
            if best is None or value < best[0] - 1e-15:
                best = (value, w)
    return (best[1] if best else None), "slsqp: " + " / ".join(messages)


def optimize_weights(actual, predictions, objective: str = "mae", previous=None, tolerance: float = 1e-8) -> dict:
    """Constrained weights with a recorded solver path."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    P = np.asarray(predictions, dtype=float)
    if P.ndim != 2 or P.shape[0] != y.size:
        raise ValueError(f"predictions must have shape (n, k) with n = {y.size}; got {P.shape}")
    k = P.shape[1]
    keep = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    y, P = y[keep], P[keep]
    started = time.perf_counter()
    if y.size == 0:
        w = np.asarray(previous, dtype=float) if weights_are_valid(previous, k, tolerance) else uniform_weights(k)
        solver = "previous_weights_no_finite_history" if weights_are_valid(previous, k, tolerance) else "uniform_no_finite_history"
        return {"weights": w, "objective_value": float("nan"), "solver": solver, "message": "no finite history",
                "n_values": 0, "seconds": time.perf_counter() - started}
    if objective == "mae":
        w, message = solve_mae_lp(y, P)
        solver = "linprog_highs"
        if not weights_are_valid(w, k, tolerance):
            w, message2 = solve_slsqp(y, P, "mae")
            solver, message = "slsqp_fallback", f"{message} | {message2}"
    else:
        w, message = solve_slsqp(y, P, objective)
        solver = "slsqp"
    if not weights_are_valid(w, k, tolerance):
        if weights_are_valid(previous, k, tolerance):
            w, solver = np.asarray(previous, dtype=float), "previous_weights_after_solver_failure"
        else:
            w, solver = uniform_weights(k), "uniform_after_solver_failure"
    return {"weights": w, "objective_value": objective_value(w, y, P, objective), "solver": solver,
            "message": message, "n_values": int(y.size), "seconds": time.perf_counter() - started}


def fuse(predictions, weights) -> np.ndarray:
    """Weighted combination along the last axis (base models)."""
    p, w = np.asarray(predictions, dtype=float), np.asarray(weights, dtype=float)
    if p.shape[-1] != w.size or not weights_are_valid(w, w.size, 1e-6):
        raise ValueError("fuse() needs one non-negative weight per base model, summing to one")
    out = p @ w
    if not np.all(np.isfinite(out)):
        raise ValueError("fused forecasts are not finite")
    return out


def block_values(actual_blocks: np.ndarray, base_blocks: np.ndarray, resolution: str, energy_fn=None):
    """Per-day objective data: hourly -> (n, 24) and (n, 24, k); daily_energy -> (n, 1) and (n, 1, k)."""
    if resolution == "hourly":
        return actual_blocks, base_blocks
    if resolution == "daily_energy":
        if energy_fn is None:
            raise ValueError("daily_energy resolution requires an energy aggregation function")
        daily_actual = energy_fn(actual_blocks)[:, None]
        daily_base = np.stack([energy_fn(base_blocks[..., m]) for m in range(base_blocks.shape[-1])], axis=-1)[:, None, :]
        return daily_actual, daily_base
    raise ValueError(f"unknown resolution {resolution}")


def rolling_dalf(actual_blocks: np.ndarray, base_blocks: np.ndarray, origin_times: np.ndarray,
                 partitions: np.ndarray, *, window_days: int = 30, resolution: str = "hourly", energy_fn=None,
                 objective: str = "mae", window_definition: str = "last_n_completed_days",
                 min_history_days: int = 7, tolerance: float = 1e-8,
                 model_names: tuple[str, ...] = ("HCRM", "HCGM", "HCLM")) -> tuple[np.ndarray, pd.DataFrame]:
    """Chronological, leakage-safe daily re-estimation of the fusion weights.

    actual_blocks (n, 24) and base_blocks (n, 24, k) are in physical units and ordered by
    origin; partitions labels each day ('validation' / 'test'). Returns hourly fused
    forecasts (n, 24) and one record per day.
    """
    n, horizon = actual_blocks.shape
    if base_blocks.shape[:2] != (n, horizon) or base_blocks.shape[2] != len(model_names):
        raise ValueError(f"base_blocks shape {base_blocks.shape} does not match (n, {horizon}, {len(model_names)})")
    origins = np.asarray(origin_times, dtype="datetime64[ns]")
    if n > 1 and not np.all(np.diff(origins.astype(np.int64)) >= horizon * 3_600 * 10**9):
        raise ValueError("forecast days must be chronological and non-overlapping")
    target_end = origins + (horizon - 1) * HOUR
    obj_actual, obj_base = block_values(actual_blocks, base_blocks, resolution, energy_fn)
    k = base_blocks.shape[2]
    fused = np.empty((n, horizon))
    records, previous = [], None
    for i in range(n):
        completed = np.flatnonzero(target_end[:i] < origins[i])
        if window_definition == "calendar_days":
            window = completed[origins[completed] >= origins[i] - np.timedelta64(window_days, "D")]
        else:
            window = completed[-window_days:]
        if window.size >= window_days:
            mode = "full_window"
        elif window.size >= min_history_days:
            mode = "partial_window_insufficient_history"
        else:
            mode = "insufficient_history"
        if mode == "insufficient_history":
            if weights_are_valid(previous, k, tolerance):
                result = {"weights": previous, "objective_value": float("nan"),
                          "solver": "previous_weights_insufficient_history", "message": "", "n_values": 0,
                          "seconds": 0.0}
            else:
                result = {"weights": uniform_weights(k), "objective_value": float("nan"),
                          "solver": "uniform_insufficient_history", "message": "", "n_values": 0, "seconds": 0.0}
        else:
            y = obj_actual[window].reshape(-1)
            P = obj_base[window].reshape(-1, k)
            result = optimize_weights(y, P, objective, previous, tolerance)
        w = np.asarray(result["weights"], dtype=float)
        if not weights_are_valid(w, k, tolerance):
            raise AssertionError(f"invalid DALF weights {w} on {origins[i]}")
        if window.size and not target_end[window[-1]] < origins[i]:
            raise AssertionError("a weight-estimation window contains values that were not yet observed")
        fused[i] = fuse(base_blocks[i], w)
        span = (float((target_end[window[-1]] - origins[window[0]]).astype("timedelta64[h]").astype(int) + 1) / 24
                if window.size else 0.0)
        records.append({
            "partition": partitions[i], "forecast_origin": pd.Timestamp(origins[i]),
            "forecast_last_hour": pd.Timestamp(target_end[i]), "resolution": resolution, "objective": objective,
            "window_mode": mode, "window_days_used": int(window.size), "values_in_objective": int(result["n_values"]),
            "window_first_origin": pd.Timestamp(origins[window[0]]) if window.size else pd.NaT,
            "window_last_hour": pd.Timestamp(target_end[window[-1]]) if window.size else pd.NaT,
            "window_calendar_days": span, "window_partitions": ",".join(sorted(set(partitions[window].tolist()))),
            "objective_value": result["objective_value"], "solver": result["solver"],
            "solver_message": result["message"], "optimization_seconds": result["seconds"],
            **{name: float(w[j]) for j, name in enumerate(model_names)}, "weight_sum": float(w.sum()),
        })
        previous = w
    return fused, pd.DataFrame(records)


def static_weights(actual_blocks: np.ndarray, base_blocks: np.ndarray, resolution: str, energy_fn=None,
                   objective: str = "mae", tolerance: float = 1e-8) -> dict:
    """SCAF: one weight vector estimated on all supplied (validation) days."""
    a, b = block_values(actual_blocks, base_blocks, resolution, energy_fn)
    return optimize_weights(a.reshape(-1), b.reshape(-1, base_blocks.shape[2]), objective, None, tolerance)

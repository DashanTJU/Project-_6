"""Diebold-Mariano"""


from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_variance(d, lag: int) -> float:
    """Long-run variance of d with Bartlett weights (lag 0 gives the sample variance with 1/T)."""
    x = np.asarray(d, dtype=float)
    x = x - x.mean()
    T = x.size
    lrv = float(x @ x / T)
    for k in range(1, min(lag, T - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(x[k:] @ x[:-k] / T)
    return lrv


def diebold_mariano(loss_baseline, loss_candidate, horizon: int = 1, lag: int | None = None,
                    harvey_correction: bool = True) -> dict:
    lb = np.asarray(loss_baseline, dtype=float).reshape(-1)
    lc = np.asarray(loss_candidate, dtype=float).reshape(-1)
    if lb.shape != lc.shape:
        raise ValueError("loss series must be aligned and of equal length")
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(lc))):
        raise ValueError("loss series must be finite")
    d = lb - lc
    T = d.size
    if T < 3:
        raise ValueError("the Diebold-Mariano test needs at least 3 paired observations")
    lag = max(horizon - 1, int(math.floor(T ** (1.0 / 3.0)))) if lag is None else int(lag)
    mean_d = float(d.mean())
    acf1 = float(np.corrcoef(d[1:], d[:-1])[0, 1]) if T > 2 and np.std(d[1:]) > 0 and np.std(d[:-1]) > 0 else float("nan")
    out = {"n_days": T, "mean_loss_differential": mean_d, "hac_lag": lag, "lag1_autocorrelation_of_d": acf1,
           "harvey_correction": harvey_correction, "note": ""}
    if np.allclose(d, 0.0):
        return {**out, "dm_statistic": 0.0, "p_value": 1.0, "note": "identical losses"}
    lrv = newey_west_variance(d, lag)
    if lrv <= 0:
        lrv = newey_west_variance(d, 0)
        out["note"] = "non-positive HAC variance; lag-0 variance used"
    dm = mean_d / math.sqrt(lrv / T)
    if harvey_correction:
        factor = math.sqrt(max((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T, 0.0))
        dm *= factor
        p = 2.0 * stats.t.sf(abs(dm), df=T - 1)
    else:
        p = 2.0 * stats.norm.sf(abs(dm))
    if T < 30:
        out["note"] = (out["note"] + "; " if out["note"] else "") + "fewer than 30 days: low power"
    return {**out, "dm_statistic": float(dm), "p_value": float(p)}


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def daily_losses(actual_blocks: np.ndarray, forecast_blocks: np.ndarray, resolution: str, loss: str,
                 energy_fn=None) -> np.ndarray:
    """One loss per forecast day at the resolution of the scientific target."""
    if resolution == "daily_energy":
        error = energy_fn(actual_blocks) - energy_fn(forecast_blocks)
        return np.abs(error) if loss == "absolute" else error ** 2
    error = np.asarray(actual_blocks, dtype=float) - np.asarray(forecast_blocks, dtype=float)
    per_hour = np.abs(error) if loss == "absolute" else error ** 2
    return per_hour.mean(axis=1)


def wilcoxon_paired(loss_baseline, loss_candidate) -> dict:
    d = np.asarray(loss_baseline, dtype=float) - np.asarray(loss_candidate, dtype=float)
    if np.allclose(d, 0):
        return {"wilcoxon_statistic": float("nan"), "wilcoxon_p_value": 1.0}
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"wilcoxon_statistic": float(res.statistic), "wilcoxon_p_value": float(res.pvalue)}


def significance_label(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "*** (p < 0.001)"
    if p < 0.01:
        return "** (p < 0.01)"
    if p < alpha:
        return f"* (p < {alpha})"
    return "not significant"


def apply_holm(table: pd.DataFrame, alpha: float, family_columns=("loss",)) -> pd.DataFrame:
    """Holm-adjust DM (and Wilcoxon) p-values within each family."""
    out = table.copy()
    out["dm_p_value_holm"] = np.nan
    if "wilcoxon_p_value" in out:
        out["wilcoxon_p_value_holm"] = np.nan
    for _, idx in out.groupby(list(family_columns)).groups.items():
        out.loc[idx, "dm_p_value_holm"] = holm_adjust(out.loc[idx, "p_value"].to_numpy())
        if "wilcoxon_p_value" in out:
            out.loc[idx, "wilcoxon_p_value_holm"] = holm_adjust(out.loc[idx, "wilcoxon_p_value"].to_numpy())
    out["family_size"] = out.groupby(list(family_columns))["p_value"].transform("size")
    out["significance_holm"] = [significance_label(p, alpha) for p in out["dm_p_value_holm"]]
    out["direction"] = np.where(out["mean_loss_differential"] > 0, "candidate lower loss",
                                np.where(out["mean_loss_differential"] < 0, "baseline lower loss", "equal"))
    return out

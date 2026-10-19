

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import ExperimentConfig, PlantConfig, boundary, capacity_in_power_unit
from data_loader import require_columns, required_columns

TARGET_VARIABLES = {"power": ["power", "irradiance", "temperature"], "irradiance": ["irradiance", "temperature"]}
MAX_LISTED_GAPS = 500


def variables_for_target(target: str) -> list[str]:
    return list(TARGET_VARIABLES[target])


def parse_timestamps(series: pd.Series, timestamp_format: str | None, timezone: str | None) -> tuple[pd.Series, dict]:
    """Parse to timezone-naive local wall-clock time with nanosecond resolution."""
    notes = {"format": timestamp_format or "automatic", "timezone": timezone}
    try:
        parsed = pd.to_datetime(series, format=timestamp_format, errors="coerce")
    except (ValueError, TypeError) as exc:
        raise ValueError("Timestamps could not be parsed consistently (mixed formats or time-zone offsets). "
                         "Set data.timestamp_format and data.timezone.") from exc
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError("Timestamps parsed to mixed types; set data.timestamp_format and data.timezone.")
    ambiguous = 0
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        if timezone:
            parsed = parsed.dt.tz_convert(timezone)
        parsed = parsed.dt.tz_localize(None)
        notes["handling"] = "time-zone-aware timestamps converted to local wall-clock time"
    elif timezone:
        localized = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        ambiguous = int((localized.isna() & parsed.notna()).sum())
        parsed = localized.dt.tz_localize(None)
        notes["handling"] = f"naive timestamps interpreted in {timezone}"
    else:
        notes["handling"] = "naive timestamps used as recorded"
    notes["ambiguous_or_nonexistent_local_times"] = ambiguous
    return parsed.astype("datetime64[ns]"), notes


def physical_limits(plant: PlantConfig, cfg: ExperimentConfig) -> dict[str, dict]:
    c = cfg.cleaning
    capacity = capacity_in_power_unit(plant, cfg)
    return {
        "power": {"negative_tolerance": c.power_negative_tolerance_fraction_of_capacity * capacity,
                  "max": c.power_max_fraction_of_capacity * capacity, "min": None},
        "irradiance": {"negative_tolerance": c.irradiance_negative_tolerance, "max": c.irradiance_max, "min": None},
        "temperature": {"negative_tolerance": None, "max": c.temperature_max, "min": c.temperature_min},
    }


def isolated_gap_mask(is_missing: np.ndarray, max_gap: int) -> np.ndarray:
    """True for missing positions in runs of length <= max_gap with valid values on both sides."""
    is_missing = np.asarray(is_missing, dtype=bool)
    out = np.zeros(is_missing.size, dtype=bool)
    if max_gap <= 0 or not is_missing.any():
        return out
    edges = np.flatnonzero(np.diff(np.r_[0, is_missing.astype(np.int8), 0]))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start <= max_gap and start > 0 and end < is_missing.size:
            out[start:end] = True
    return out


def missing_runs(timestamps: pd.DatetimeIndex, is_missing: np.ndarray) -> list[dict]:
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(is_missing, dtype=np.int8), 0]))
    return [{"first_missing_hour": str(timestamps[s]), "last_missing_hour": str(timestamps[e - 1]),
             "hours": int(e - s)} for s, e in zip(edges[::2], edges[1::2])]


def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def clean_target_series(raw: pd.DataFrame, plant: PlantConfig, target: str, cfg: ExperimentConfig,
                        max_gap_hours: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean one plant/target series; returns (hourly cleaned data, report)."""
    roles = required_columns(cfg.data, target)
    require_columns(raw, roles, f"{plant.name} ({plant.file_path})")
    variables = variables_for_target(target)
    max_gap = cfg.cleaning.max_interpolation_gap_hours if max_gap_hours is None else max_gap_hours
    rename = {roles[role]: role for role in roles}
    df = raw[list(rename)].rename(columns=rename).copy()
    df.insert(0, "source_row", np.arange(len(df), dtype=np.int64))
    n_raw = len(df)
    steps = []

    df["timestamp"], ts_notes = parse_timestamps(df["timestamp"], cfg.data.timestamp_format, cfg.data.timezone)
    invalid_ts = df["timestamp"].isna()
    df = df.loc[~invalid_ts].copy()
    steps.append({"step": "remove unparseable timestamps", "rows_removed": int(invalid_ts.sum()),
                  "rows_after": len(df)})

    value_issues = {}
    for var in variables:
        original = df[var]
        numeric = pd.to_numeric(original, errors="coerce").astype(float)
        blank = original.isna()
        non_numeric = numeric.isna() & ~blank
        coded = numeric.isin(cfg.data.missing_value_codes)
        non_finite = numeric.notna() & ~np.isfinite(numeric)
        df[var] = numeric.mask(coded | non_finite)
        value_issues[var] = {"blank": int(blank.sum()), "non_numeric": int(non_numeric.sum()),
                             "missing_value_code": int(coded.sum()), "infinite": int(non_finite.sum())}
    steps.append({"step": "non-numeric, coded and infinite values set to missing", "values": value_issues,
                  "rows_after": len(df)})

    df = df.sort_values(["timestamp", "source_row"], kind="mergesort")
    dup_any = df.duplicated("timestamp", keep=False)
    conflicting = 0
    if dup_any.any():
        conflicting = int((df.loc[dup_any].groupby("timestamp")[variables].nunique(dropna=False) > 1).any(axis=1).sum())
    dup = df.duplicated("timestamp", keep="first")
    df = df.loc[~dup].copy()
    steps.append({"step": "duplicate timestamps: keep first occurrence in file", "rows_removed": int(dup.sum()),
                  "timestamps_with_conflicting_values": conflicting, "rows_after": len(df)})

    off_hour = df["timestamp"] != df["timestamp"].dt.floor("h")
    df = df.loc[~off_hour].copy()
    steps.append({"step": "remove timestamps not on the hour", "rows_removed": int(off_hour.sum()),
                  "rows_after": len(df)})
    if df.empty:
        raise ValueError(f"{plant.name}/{target}: no valid hourly rows remain after timestamp checks.")

    limits = physical_limits(plant, cfg)
    plausibility = {}
    for var in variables:
        lim, x = limits[var], df[var]
        clipped = np.zeros(len(df), dtype=bool)
        invalid = np.zeros(len(df), dtype=bool)
        if lim["negative_tolerance"] is not None:
            clipped = ((x < 0) & (x >= -lim["negative_tolerance"])).to_numpy()
            invalid |= (x < -lim["negative_tolerance"]).to_numpy()
        if lim["min"] is not None:
            invalid |= (x < lim["min"]).to_numpy()
        if lim["max"] is not None:
            invalid |= (x > lim["max"]).to_numpy()
        df.loc[clipped, var] = 0.0
        df.loc[invalid, var] = np.nan
        df[f"{var}_negative_set_to_zero"] = clipped
        plausibility[var] = {"limits": lim, "small_negatives_set_to_zero": int(clipped.sum()),
                             "implausible_set_to_missing": int(invalid.sum())}
    steps.append({"step": "physical plausibility (configured site limits)", "variables": plausibility,
                  "rows_after": len(df)})

    grid_index = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=pd.Timedelta(hours=1))
    grid = df.set_index("timestamp").reindex(grid_index)
    grid.index.name = "timestamp"
    n_grid = len(grid)
    inserted = grid["source_row"].isna().to_numpy()
    for var in variables:
        flag = f"{var}_negative_set_to_zero"
        grid[flag] = grid[flag].astype("boolean").fillna(False).astype(bool)
    steps.append({"step": "place on complete hourly grid", "grid_hours": n_grid,
                  "missing_timestamps_inserted": int(inserted.sum()),
                  "missing_timestamps_pct": _pct(int(inserted.sum()), n_grid)})

    interpolation = {}
    for var in variables:
        series = grid[var]
        mask = isolated_gap_mask(series.isna().to_numpy(), max_gap)
        filled = series.interpolate(method="linear", limit_area="inside")
        grid[var] = series.where(~mask, filled)
        grid[f"{var}_interpolated"] = mask
        interpolation[var] = {"values_interpolated": int(mask.sum()), "pct_of_grid": _pct(int(mask.sum()), n_grid)}
    steps.append({"step": f"linear interpolation of isolated gaps (<= {max_gap} h, valid neighbours on both sides)",
                  "variables": interpolation})

    unresolved = grid[variables].isna().any(axis=1).to_numpy()
    runs = missing_runs(grid.index, unresolved)
    clean = grid.loc[~unresolved].reset_index()
    clean["timestamp_inserted_on_grid"] = inserted[~unresolved]
    clean["source_row"] = clean["source_row"].astype("Int64")
    steps.append({"step": "remove hours still missing (windows cannot cross them)",
                  "hours_removed": int(unresolved.sum()), "pct_of_grid": _pct(int(unresolved.sum()), n_grid),
                  "gap_runs": len(runs), "rows_after": len(clean)})

    observed = int((~clean["timestamp_inserted_on_grid"]).sum())
    report = {
        "plant": plant.name, "target": target, "variables": variables, "raw_rows": n_raw,
        "timestamps": ts_notes, "steps": steps,
        "grid_first_hour": str(grid.index[0]), "grid_last_hour": str(grid.index[-1]), "grid_hours": n_grid,
        "cleaned_rows": int(len(clean)), "cleaned_rows_observed": observed,
        "cleaned_rows_with_interpolated_timestamp": int(len(clean) - observed),
        "unresolved_hours": int(unresolved.sum()), "unresolved_hours_pct_of_grid": _pct(int(unresolved.sum()), n_grid),
        "unresolved_gap_runs": runs[:MAX_LISTED_GAPS], "gap_list_truncated_at": MAX_LISTED_GAPS,
        "zero_values_preserved": {v: int((clean[v] == 0).sum()) for v in variables},
        "reference_cleaned_rows": plant.reference_cleaned_rows,
        "matches_reference_cleaned_rows": (None if plant.reference_cleaned_rows is None
                                           else int(len(clean)) == plant.reference_cleaned_rows),
        "raw_rows_vs_reference": {"raw_rows": n_raw, "reference_raw_rows": plant.reference_raw_rows},
        "max_interpolation_gap_hours": max_gap,
    }
    return clean, report


def quality_summary_row(report: dict) -> dict:
    """Flat data-quality row for the summary table."""
    by_step = {s["step"]: s for s in report["steps"]}
    plaus = next(s for s in report["steps"] if s["step"].startswith("physical"))["variables"]
    interp = next(s for s in report["steps"] if s["step"].startswith("linear"))["variables"]
    values = next(s for s in report["steps"] if s["step"].startswith("non-numeric"))["values"]
    row = {
        "plant": report["plant"], "target": report["target"], "raw_rows": report["raw_rows"],
        "unparseable_timestamps_removed": by_step["remove unparseable timestamps"]["rows_removed"],
        "duplicate_timestamps_removed": by_step["duplicate timestamps: keep first occurrence in file"]["rows_removed"],
        "off_hour_timestamps_removed": by_step["remove timestamps not on the hour"]["rows_removed"],
        "grid_hours": report["grid_hours"],
        "missing_timestamps": by_step["place on complete hourly grid"]["missing_timestamps_inserted"],
        "unresolved_hours_removed": report["unresolved_hours"],
        "unresolved_pct_of_grid": report["unresolved_hours_pct_of_grid"],
        "unresolved_gap_runs": len(report["unresolved_gap_runs"]),
        "cleaned_rows": report["cleaned_rows"], "reference_cleaned_rows": report["reference_cleaned_rows"],
        "matches_reference": report["matches_reference_cleaned_rows"],
    }
    for var in report["variables"]:
        row[f"{var}_missing_or_coded"] = sum(values[var].values())
        row[f"{var}_small_negatives_to_zero"] = plaus[var]["small_negatives_set_to_zero"]
        row[f"{var}_implausible_to_missing"] = plaus[var]["implausible_set_to_missing"]
        row[f"{var}_interpolated"] = interp[var]["values_interpolated"]
        row[f"{var}_zeros_preserved"] = report["zero_values_preserved"][var]
    return row


#  splitting
def chronological_boundaries(n_rows: int, cfg: ExperimentConfig) -> tuple[int, int]:
    """Half-open boundaries: train [0, a), validation [a, b), test [b, n)."""
    s = cfg.split
    train_end = boundary(n_rows, s.train_fraction, s.boundary_rounding)
    val_end = boundary(n_rows, s.train_fraction + s.validation_fraction, s.boundary_rounding)
    if not 0 < train_end < val_end < n_rows:
        raise ValueError(f"{n_rows} cleaned rows are too few for the configured split.")
    return train_end, val_end


def split_report(clean: pd.DataFrame, train_end: int, val_end: int, plant: PlantConfig, cfg: ExperimentConfig) -> dict:
    n = len(clean)
    ranges = {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, n)}
    out = {"cleaned_rows": n, "boundary_rounding": cfg.split.boundary_rounding,
           "fractions": {"train": cfg.split.train_fraction, "validation": cfg.split.validation_fraction,
                         "test": cfg.split.test_fraction}, "partitions": {}}
    for name, (a, b) in ranges.items():
        out["partitions"][name] = {"rows": b - a, "index_first": a, "index_last": b - 1,
                                   "first_timestamp": str(clean["timestamp"].iloc[a]),
                                   "last_timestamp": str(clean["timestamp"].iloc[b - 1])}
    ref = plant.reference_split_indices
    if not ref or plant.reference_cleaned_rows is None:
        out["reference_check"] = "no reference indices configured"
    elif n != plant.reference_cleaned_rows:
        out["reference_check"] = (f"not applicable: {n} cleaned rows differ from the reference "
                                  f"{plant.reference_cleaned_rows}; boundaries recalculated from the actual series")
    else:
        mismatches = {k: {"computed": [out["partitions"][k]["index_first"], out["partitions"][k]["index_last"]],
                          "reference": list(ref[k])}
                      for k in ranges if [out["partitions"][k]["index_first"], out["partitions"][k]["index_last"]]
                      != list(ref[k])}
        out["reference_check"] = "matches reference indices" if not mismatches else {"mismatches": mismatches}
    return out


#  scaling
def fit_scalers(train_rows: pd.DataFrame, features: list[str], target: str) -> tuple[MinMaxScaler, MinMaxScaler]:
    """Feature scaler and target scaler, both fitted on training rows only."""
    if train_rows.empty:
        raise ValueError("The training partition is empty; scalers cannot be fitted.")
    feature_scaler = MinMaxScaler().fit(train_rows[features].to_numpy(dtype=float))
    target_scaler = MinMaxScaler().fit(train_rows[[target]].to_numpy(dtype=float))
    return feature_scaler, target_scaler


def inverse_target(values, target_scaler: MinMaxScaler) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return (arr - target_scaler.min_[0]) / target_scaler.scale_[0]


def scale_target(values, target_scaler: MinMaxScaler) -> np.ndarray:
    return np.asarray(values, dtype=float) * target_scaler.scale_[0] + target_scaler.min_[0]


def save_scalers(feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler, features: list[str], target: str,
                 train_rows: int, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, folder / "feature_scaler.joblib")
    joblib.dump(target_scaler, folder / "target_scaler.joblib")
    payload = {"type": "sklearn MinMaxScaler", "fitted_on": f"training rows 0..{train_rows - 1}",
               "features": features, "feature_min": feature_scaler.data_min_.tolist(),
               "feature_max": feature_scaler.data_max_.tolist(), "target": target,
               "target_min": float(target_scaler.data_min_[0]), "target_max": float(target_scaler.data_max_[0])}
    (folder / "scalers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scalers(folder: Path) -> tuple[MinMaxScaler, MinMaxScaler]:
    return joblib.load(folder / "feature_scaler.joblib"), joblib.load(folder / "target_scaler.joblib")

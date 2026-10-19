

from __future__ import annotations

import csv
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd 
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  

COLORS = {"Actual": "#111111", "DALF": "#d62728", "SCAF": "#ff7f0e", "UEBF": "#8c564b",
          "HCRM": "#1f77b4", "HCGM": "#2ca02c", "HCLM": "#9467bd",
          "RNN": "#7fb3d5", "GRU": "#82c785", "LSTM": "#c3a6d8"}
FUSION = ("DALF", "SCAF", "UEBF")
HOUR_NS = 3_600 * 10**9


ORIGIN_CSV = True  
ORIGIN_HEADER_STYLES = ("three_rows", "single_row")

ORIGIN_HEADER_STYLE = "three_rows"
ORIGIN_FOLDER = "origin_csv"
ORIGIN_DATE_FORMAT = "%Y-%m-%d %H:%M"  
ORIGIN_ENCODING = "utf-8-sig"  
ORIGIN_NUMBER_FORMAT = ".10g"  


@dataclass
class OriginColumn:
    """One worksheet column: Long Name, values, Units and Comments."""

    long_name: str
    values: object
    unit: str = ""
    comment: str = ""


@dataclass
class OriginTable:
    """One CSV file for Origin (one figure or one panel)."""

    part: str  
    content: str  
    columns: list[OriginColumn]
    plot: str  
    notes: str = ""
    y_columns: list[str] | None = None  
    x_column: str | None = None  

    def x_name(self) -> str:
        return self.x_column or self.columns[0].long_name

    def y_names(self) -> list[str]:
        if self.y_columns is not None:
            return list(self.y_columns)
        return [c.long_name for c in self.columns[1:] if c.long_name != "Elapsed time"]


def _clean_text(value) -> str:
    """Text cell without commas or line breaks (keeps every CSV reader on one field per cell)."""
    return str(value).replace("\r", " ").replace("\n", " ").replace(",", ";").strip()


def _cells(values, n_rows: int) -> list[str]:
    """Format one column for the CSV: numbers, dates or text; missing values stay empty."""
    series = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_bool_dtype(series):
        out = ["1" if bool(v) else "0" for v in series]
    elif pd.api.types.is_datetime64_any_dtype(series):
        out = ["" if pd.isna(v) else pd.Timestamp(v).strftime(ORIGIN_DATE_FORMAT) for v in series]
    elif pd.api.types.is_numeric_dtype(series):
        out = [format(float(v), ORIGIN_NUMBER_FORMAT) if np.isfinite(v) else "" for v in series.astype(float)]
    else:
        out = ["" if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT else _clean_text(v)
               for v in series]
    return out + [""] * (n_rows - len(out))


def write_origin_csv(path: Path, columns: list[OriginColumn], header_style: str = ORIGIN_HEADER_STYLE) -> Path:
    """Write one Origin-ready CSV file (columns of unequal length are padded with empty cells)."""
    if header_style not in ORIGIN_HEADER_STYLES:
        raise ValueError(f"header_style must be one of {ORIGIN_HEADER_STYLES}, not {header_style!r}")
    if not columns:
        raise ValueError(f"{path.name}: an Origin table needs at least one column")
    names = [c.long_name for c in columns]
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: column Long Names must be unique, got {names}")
    n_rows = max(len(pd.Series(c.values)) for c in columns)
    body = [_cells(c.values, n_rows) for c in columns]
    if header_style == "three_rows":
        header = [[_clean_text(c.long_name) for c in columns], [_clean_text(c.unit) for c in columns],
                  [_clean_text(c.comment) for c in columns]]
    else:
        header = [[_clean_text(f"{c.long_name} ({c.unit})" if c.unit else c.long_name) for c in columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=ORIGIN_ENCODING, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(header)
        writer.writerows(zip(*body))
    return path


def read_origin_csv(path: Path, header_style: str = ORIGIN_HEADER_STYLE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read an Origin CSV back into (data, labels) - used to check the files in Python."""
    rows = 3 if header_style == "three_rows" else 1
    labels = pd.read_csv(path, nrows=rows, header=None, encoding=ORIGIN_ENCODING, keep_default_na=False)
    data = pd.read_csv(path, skiprows=rows, header=None, encoding=ORIGIN_ENCODING)
    data.columns = labels.iloc[0].tolist()
    labels.index = ["long_name", "unit", "comment"][:rows]
    labels.columns = data.columns
    return data, labels


def split_axis_label(label: str) -> tuple[str, str]:
    """'Power (MW)' -> ('Power', 'MW'); a label without a final '(unit)' -> (label, '')."""
    match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", str(label))
    return (match.group(1), match.group(2)) if match else (str(label), "")


def _datetimes(values) -> np.ndarray:
    return np.asarray(pd.to_datetime(np.asarray(values).reshape(-1)), dtype="datetime64[ns]")


def _with_gap_rows(times, series: dict[str, object], step_ns: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Insert one row with empty values at every gap; its time is the first missing hour or day."""
    t = _datetimes(times)
    values = {key: np.asarray(v, dtype=float) for key, v in series.items()}
    if t.size < 2:
        return t, values
    cut = np.flatnonzero(np.diff(t.astype(np.int64)) > step_ns) + 1
    if not cut.size:
        return t, values
    t_out = np.insert(t, cut, t[cut - 1] + np.timedelta64(int(step_ns), "ns"))
    return t_out, {key: np.insert(v, cut, np.nan) for key, v in values.items()}


def _elapsed_column(times: np.ndarray) -> OriginColumn:
    t = np.asarray(times, dtype="datetime64[ns]")
    if t.size == 0:
        return OriginColumn("Elapsed time", np.empty(0), "d", "numeric alternative X")
    start = pd.Timestamp(t[0])
    return OriginColumn("Elapsed time", (t - t[0]) / np.timedelta64(1, "D"), "d",
                        f"days since {start:%Y-%m-%d %H:%M}; numeric alternative X")


def _mean_and_count(stack: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds that reached each epoch (NaN where none did) and the number of seeds used."""
    matrix = np.vstack(stack)
    counts = np.sum(np.isfinite(matrix), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(matrix, axis=0)
    return means, counts


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100, "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


class FigureSaver:
    """Saves figures in all formats with a caption, the plotted data, a caption index and Origin CSV files."""

    def __init__(self, formats, dpi: int, synthetic: bool, origin_csv: bool = ORIGIN_CSV,
                 origin_header: str = ORIGIN_HEADER_STYLE):
        if origin_header not in ORIGIN_HEADER_STYLES:
            raise ValueError(f"origin_header must be one of {ORIGIN_HEADER_STYLES}")
        self.formats, self.dpi, self.synthetic = list(formats), dpi, synthetic
        self.origin_csv, self.origin_header = origin_csv, origin_header
        self._caption_files: set[Path] = set()
        self._origin_index: dict[Path, list[dict]] = {}

    def save(self, fig, folder: Path, name: str, caption: str, data: pd.DataFrame | None,
             origin: list[OriginTable] | None = None) -> list[Path]:
        folder.mkdir(parents=True, exist_ok=True)
        if self.synthetic:
            caption = "SYNTHETIC DATA - pipeline check only, not research results. " + caption
        wrapped = "\n".join(textwrap.wrap(caption, width=int(fig.get_figwidth() * 17)))
        n_lines = wrapped.count("\n") + 1
        bottom = min(0.3, (0.16 * n_lines + 0.1) / fig.get_figheight())
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.text(0.01, 0.01, wrapped, fontsize=7, va="bottom", ha="left", color="0.25")
        paths = []
        for fmt in self.formats:
            path = folder / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.dpi if fmt == "png" else None)
            paths.append(path)
        plt.close(fig)
        if data is not None:
            data.to_csv(folder / f"{name}_data.csv", index=False)
        captions = folder / "captions.md"
        mode = "a" if captions in self._caption_files else "w"  
        self._caption_files.add(captions)
        with open(captions, mode, encoding="utf-8") as handle:
            handle.write(f"**{name}** - {caption}\n\n")
        if self.origin_csv and origin:
            paths += self._write_origin(folder, name, caption, origin)
        return paths

    def _write_origin(self, folder: Path, name: str, caption: str, tables: list[OriginTable]) -> list[Path]:
        target = folder / ORIGIN_FOLDER
        target.mkdir(parents=True, exist_ok=True)
        if target not in self._origin_index:
            self._origin_index[target] = []
        rows = [r for r in self._origin_index[target] if r["figure"] != name]  
        written = []
        for table in tables:
            columns = list(table.columns)
            if self.synthetic:
                first = columns[0]
                columns[0] = OriginColumn(first.long_name, first.values, first.unit,
                                          ("SYNTHETIC DATA - not research results. " + first.comment).strip())
            file = target / f"{name}__{table.part}.csv"
            write_origin_csv(file, columns, self.origin_header)
            written.append(file)
            rows.append({"figure": name, "csv_file": file.name, "content": table.content,
                         "x_column": table.x_name(), "y_columns": "; ".join(table.y_names()),
                         "suggested_origin_plot": table.plot, "notes": table.notes,
                         "rows": int(max(len(pd.Series(c.values)) for c in columns)),
                         "header_style": self.origin_header,
                         "data_source": "SYNTHETIC smoke test" if self.synthetic else "experiment outputs",
                         "figure_caption": caption})
        self._origin_index[target] = rows
        pd.DataFrame(rows).to_csv(target / "origin_index.csv", index=False, encoding=ORIGIN_ENCODING)
        return written


def _time_axis(ax) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def with_breaks(times, values, step_ns: int = HOUR_NS):
    """Insert NaN where consecutive points are further apart than one step."""
    t = np.asarray(pd.to_datetime(times), dtype="datetime64[ns]")
    v = np.asarray(values, dtype=float)
    if t.size < 2:
        return t, v
    cut = np.flatnonzero(np.diff(t.astype(np.int64)) > step_ns) + 1
    if not cut.size:
        return t, v
    return np.insert(t, cut, t[cut - 1] + np.timedelta64(1, "h")), np.insert(v, cut, np.nan)


#
def plot_daily_energy(saver: FigureSaver, folder: Path, daily: pd.DataFrame, title: str, unit: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    days = pd.to_datetime(daily["forecast_day"]).to_numpy(dtype="datetime64[ns]")
    ax = axes[0]
    t, v = with_breaks(days, daily["actual"], 24 * HOUR_NS)
    ax.plot(t, v, color=COLORS["Actual"], marker="o", ms=2.5, lw=1.2, label="Actual")
    for method in reversed(FUSION):  # DALF drawn last, on top
        t, v = with_breaks(days, daily[method], 24 * HOUR_NS)
        ax.plot(t, v, color=COLORS[method], lw=1.3 if method == "DALF" else 1.0, marker=".", ms=3, label=method)
    ax.set_ylabel(f"Daily energy ({unit})")
    ax.set_title(title)
    ax.legend(ncol=4)
    err = daily["actual"] - daily["DALF"]
    axes[1].bar(days, err, width=0.8, color=np.where(err >= 0, "#4a6a8a", "#d62728"))
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_ylabel(f"Actual - DALF ({unit})")
    _time_axis(axes[1])

    series = {"Actual": daily["actual"], **{m: daily[m] for m in FUSION}, "error": err}
    t_gap, values = _with_gap_rows(days, series, 24 * HOUR_NS)
    columns = [OriginColumn("Forecast day", t_gap, "", "first hour of the 24-hour forecast day (local time)"),
               OriginColumn("Actual", values["Actual"], unit, "observed daily energy = sum of 24 hourly values x 1 h")]
    columns += [OriginColumn(m, values[m], unit, f"{m} forecast daily energy") for m in FUSION]
    columns += [OriginColumn("Actual - DALF", values["error"], unit, "panel b: daily error of DALF"),
                _elapsed_column(t_gap)]
    origin = [OriginTable(
        "daily_energy", "Observed and forecast daily energy, test period (panel a) and DALF error (panel b)", columns,
        plot="Panel a: Line + Symbol (Y = Actual; DALF; SCAF; UEBF). Panel b: Column (Y = Actual - DALF)",
        notes="Empty rows mark missing forecast days.",
        y_columns=["Actual", *FUSION, "Actual - DALF"])]
    saver.save(fig, folder, "fig03_daily_energy_actual_vs_predicted",
               f"Figure 1. {title}: observed daily PV energy and the fused forecasts over the test period; lower "
               f"panel: daily error of DALF. Daily energy is the sum of the 24 hourly values x 1 h ({unit}); "
               "missing days are shown as breaks.", daily, origin)


#
def plot_hourly_series(saver: FigureSaver, folder: Path, frame: pd.DataFrame, name: str, title: str, ylabel: str,
                       zoom_days: int, figure_number: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    last = frame["timestamp"].max()
    zoom = frame[frame["timestamp"] > last - pd.Timedelta(days=zoom_days)]
    for ax, data, sub in ((axes[0], frame, "Full test period"), (axes[1], zoom, f"Final {zoom_days} test days")):
        for col, lw in (("actual", 1.2), ("UEBF", 0.8), ("SCAF", 0.8), ("DALF", 1.0)):  
            t, v = with_breaks(data["timestamp"], data[col])
            ax.plot(t, v, color=COLORS["Actual" if col == "actual" else col], lw=lw,
                    label="Actual" if col == "actual" else col)
        ax.set_ylabel(ylabel)
        ax.set_title(sub)
        _time_axis(ax)
        ax.legend(ncol=4, loc="upper right")
    fig.suptitle(title)

    quantity, unit = split_axis_label(ylabel)
    t_gap, values = _with_gap_rows(frame["timestamp"], {"Actual": frame["actual"], **{m: frame[m] for m in FUSION}},
                                   HOUR_NS)
    columns = [OriginColumn("Time", t_gap, "", "target hour (local time; hour-start label)"),
               OriginColumn("Actual", values["Actual"], unit, f"observed {quantity.lower()}")]
    columns += [OriginColumn(m, values[m], unit, f"{m} hourly forecast") for m in FUSION]
    columns.append(_elapsed_column(t_gap))
    zoom_start = pd.Timestamp(zoom["timestamp"].min()) if len(zoom) else None
    origin = [OriginTable(
        "hourly", f"Observed and fused hourly {quantity.lower()}, test period", columns,
        plot="Line (Y = Actual; DALF; SCAF; UEBF); two layers for full period and zoom",
        notes=(f"Panel b (zoom) uses the same columns with the X axis from {zoom_start:%Y-%m-%d %H:%M} to "
               f"{pd.Timestamp(last):%Y-%m-%d %H:%M}. Empty rows mark missing hours.") if zoom_start is not None
        else "Empty rows mark missing hours.",
        y_columns=["Actual", *FUSION])]
    saver.save(fig, folder, name,
               f"Figure 2. {title}: observed hourly values and the fused forecasts (DALF, SCAF, UEBF) "
               f"for the test period, full period and final {zoom_days} days. Missing hours are shown as breaks.",
               frame[["timestamp", "actual", "DALF", "SCAF", "UEBF"]], origin)


#
def plot_loss_curves(saver: FigureSaver, folder: Path, histories: pd.DataFrame, title: str) -> None:
    models = list(dict.fromkeys(histories["model"]))
    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.3 * rows), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        data = histories[histories["model"] == model]
        for seed, run in data.groupby("seed"):
            ax.plot(run["epoch"], run["loss"], color="#1f77b4", lw=0.8, alpha=0.6)
            ax.plot(run["epoch"], run["val_loss"], color="#d62728", lw=0.8, alpha=0.6)
        ax.plot([], [], color="#1f77b4", label="training")
        ax.plot([], [], color="#d62728", label="validation")
        ax.set_title(f"{model} ({data['seed'].nunique()} seeds)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (normalized target)")
        ax.set_yscale("log")
        ax.legend()
    for ax in axes.ravel()[len(models):]:
        ax.set_axis_off()
    fig.suptitle(title)

    loss_comment = "loss on the training-fitted min-max scale of the target"
    all_epochs = np.arange(1, int(histories["epoch"].max()) + 1)
    origin, mean_columns = [], [OriginColumn("Epoch", all_epochs, "", "training epoch")]
    for model in models:
        data = histories[histories["model"] == model]
        epochs = np.arange(1, int(data["epoch"].max()) + 1)
        columns = [OriginColumn("Epoch", epochs, "", "training epoch")]
        train_runs, val_runs, train_all, val_all = [], [], [], []
        for seed, run in data.groupby("seed"):
            run = run.drop_duplicates("epoch", keep="last").set_index("epoch")
            train, val = (run[c].reindex(epochs).to_numpy(dtype=float) for c in ("loss", "val_loss"))
            columns += [OriginColumn(f"Training loss seed {seed}", train, "dimensionless", loss_comment),
                        OriginColumn(f"Validation loss seed {seed}", val, "dimensionless", loss_comment)]
            train_runs.append(train)
            val_runs.append(val)
            train_all.append(run["loss"].reindex(all_epochs).to_numpy(dtype=float))
            val_all.append(run["val_loss"].reindex(all_epochs).to_numpy(dtype=float))
        train_mean, n_seeds = _mean_and_count(train_runs)
        val_mean, _ = _mean_and_count(val_runs)
        columns += [OriginColumn("Mean training loss", train_mean, "dimensionless",
                                 "mean over the seeds that reached this epoch"),
                    OriginColumn("Mean validation loss", val_mean, "dimensionless",
                                 "mean over the seeds that reached this epoch"),
                    OriginColumn("Seeds in mean", n_seeds, "count", "early stopping ends seeds at different epochs")]
        origin.append(OriginTable(
            model, f"{model}: training and validation loss per epoch for every seed", columns,
            plot="Line, log10 Y axis (blue: training; red: validation)",
            notes="Empty cells: the seed had already stopped (early stopping).",
            y_columns=[c.long_name for c in columns[1:-1]]))
        mean_columns += [OriginColumn(f"{model} mean training loss", _mean_and_count(train_all)[0], "dimensionless",
                                      "mean over seeds"),
                         OriginColumn(f"{model} mean validation loss", _mean_and_count(val_all)[0], "dimensionless",
                                      "mean over seeds")]
    origin.append(OriginTable("mean_all_models", "Seed-mean training and validation loss of every model",
                              mean_columns, plot="Line, log10 Y axis (one curve per model and loss)",
                              notes="Not drawn in the PNG; a compact alternative to the per-seed panels."))
    saver.save(fig, folder, "fig06_loss_curves",
               f"Figure 3. {title}: training and validation loss per epoch for every seed (one line each); the "
               "loss is computed on the training-fitted min-max scale of the target (dimensionless, log axis).",
               histories, origin)


#
def plot_weights(saver: FigureSaver, folder: Path, weights: pd.DataFrame, models: list[str], title: str) -> None:
    data = weights[weights["partition"] == "test"]
    origins = pd.to_datetime(data["forecast_origin"]).to_numpy(dtype="datetime64[ns]")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for m in models:
        t, v = with_breaks(origins, data[m], 24 * HOUR_NS)
        ax.plot(t, v, color=COLORS[m], lw=1.1, marker="o", ms=2, label=m)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("DALF weight (dimensionless)")
    ax.set_xlabel("Forecast day (test period)")
    ax.set_title(title)
    _time_axis(ax)
    ax.legend(ncol=3)
    res = data["resolution"].iloc[0] if len(data) else ""

    series = {m: data[m] for m in models}
    if "window_days_used" in data:
        series["window"] = data["window_days_used"]
    t_gap, values = _with_gap_rows(origins, series, 24 * HOUR_NS)
    columns = [OriginColumn("Forecast day", t_gap, "", "first hour of the forecast day (local time)")]
    columns += [OriginColumn(m, values[m], "dimensionless", "DALF weight; the weights of one day sum to 1")
                for m in models]
    if "window" in values:
        columns.append(OriginColumn("Days in optimization window", values["window"], "days",
                                    "completed days used to estimate the weights"))
    columns.append(_elapsed_column(t_gap))
    origin = [OriginTable("weights", f"Daily DALF weights, test period (objective MAE, {res} resolution)", columns,
                          plot="Line + Symbol (or stacked area, since the weights sum to 1)",
                          notes="Empty rows mark missing forecast days.", y_columns=list(models))]
    saver.save(fig, folder, "fig07_dalf_weights",
               f"Figure 4. {title}: DALF weights re-estimated each day from the preceding completed days "
               f"(objective MAE at {res} resolution); weights are non-negative and sum to one.", data, origin)


#
def plot_fusion_comparison(saver: FigureSaver, folder: Path, metrics: pd.DataFrame, improvements: pd.DataFrame) -> None:
    data = metrics[metrics["model"].isin(FUSION) & (metrics["scale"] == "physical") & metrics["primary"]]
    panels = list(data.groupby(["target", "metric"]).groups)
    panels = [p for p in panels if p[1] in ("MAE", "RMSE")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.2), squeeze=False)
    origin = []
    for ax, (target, metric) in zip(axes.ravel(), panels):
        sub = data[(data["target"] == target) & (data["metric"] == metric)]
        plants = list(dict.fromkeys(sub["plant"]))
        x = np.arange(len(plants))
        bars = {}
        for j, method in enumerate(FUSION):
            vals = [sub[(sub["plant"] == p) & (sub["model"] == method)]["value"].mean() for p in plants]
            bars[method] = vals
            ax.bar(x + (j - 1) * 0.26, vals, width=0.26, color=COLORS[method], label=method)
        gains = {"SCAF": [], "UEBF": []}
        for i, p in enumerate(plants):
            for base in ("SCAF", "UEBF"):
                imp = improvements[(improvements["plant"] == p) & (improvements["target"] == target)
                                   & (improvements["metric"] == metric) & (improvements["baseline"] == base)
                                   & improvements["primary"]]
                gains[base].append(float(imp["improvement_pct"].iloc[0]) if len(imp) else np.nan)
                if len(imp):
                    ax.annotate(f"vs {base}: {imp['improvement_pct'].iloc[0]:+.1f}%", (i, 0), xytext=(0, -24 - 9 * (base == "UEBF")),
                                textcoords="offset points", ha="center", fontsize=6.5)
        ax.set_xticks(x, plants)
        unit = sub["unit"].iloc[0]
        resolution = sub["resolution"].iloc[0]
        ax.set_ylabel(f"{metric} ({unit})")
        ax.set_title(f"{target}, {resolution.replace('_', ' ')}: {metric}")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)  # headroom so the legend never covers a bar
        ax.legend(loc="upper center", ncol=3, fontsize=7)
        columns = [OriginColumn("Plant", plants, "", "X as categories")]
        columns += [OriginColumn(m, bars[m], unit, f"test-period {metric} of {m} ({resolution.replace('_', ' ')})")
                    for m in FUSION]
        columns += [OriginColumn(f"Improvement vs {b}", gains[b], "%",
                                 f"(E_{b} - E_DALF) / E_{b} x 100; positive = DALF better") for b in ("SCAF", "UEBF")]
        origin.append(OriginTable(
            f"{target}_{metric}", f"{target}: test-period {metric} of DALF, SCAF and UEBF per plant", columns,
            plot="Grouped column (X = Plant as categories; Y = DALF; SCAF; UEBF); improvements as labels",
            y_columns=list(FUSION)))
    saver.save(fig, folder, "fig08_fusion_comparison",
               "Figure 5. Test-period errors of DALF, SCAF and UEBF at the resolution of each scientific target "
               "(daily energy for power, hourly values for irradiance). Annotations: relative improvement of DALF "
               "over SCAF and UEBF, (E_baseline - E_DALF) / E_baseline x 100.", data, origin)


# 
def plot_scatter(saver: FigureSaver, folder: Path, frame: pd.DataFrame, methods: list[str], title: str, unit: str,
                 value_label: str) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.8), squeeze=False)
    lim = [min(frame["actual"].min(), frame[methods].min().min()), max(frame["actual"].max(), frame[methods].max().max())]
    for ax, m in zip(axes.ravel(), methods):
        ax.scatter(frame["actual"], frame[m], s=4 if len(frame) > 500 else 12, alpha=0.35, color=COLORS[m],
                   edgecolors="none")
        ax.plot(lim, lim, color="0.3", lw=0.8, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"Actual ({unit})")
        ax.set_ylabel(f"{m} forecast ({unit})")
        ax.set_title(m)
    fig.suptitle(title)

    points = [OriginColumn("Actual", frame["actual"].to_numpy(dtype=float), unit, f"observed value per {value_label[:-1]}")]
    points += [OriginColumn(m, frame[m].to_numpy(dtype=float), unit, f"{m} forecast") for m in methods]
    origin = [
        OriginTable("points", f"Forecast against observation for each test-period {value_label[:-1]}", points,
                    plot="Scatter, one panel per method (X = Actual; Y = method)", y_columns=list(methods)),
        OriginTable("identity_line", "End points of the dashed 1:1 line and the common axis range",
                    [OriginColumn("Actual", lim, unit, "axis range of every panel"),
                     OriginColumn("Forecast", lim, unit, "1:1 line (forecast = actual)")],
                    plot="Line (dashed) added to every scatter panel"),
    ]
    saver.save(fig, folder, "fig10_scatter_actual_vs_predicted",
               f"Figure 6. {title}: forecast against observation for each test-period {value_label[:-1]}; the "
               "dashed line is the 1:1 line.", frame[["actual"] + methods], origin)



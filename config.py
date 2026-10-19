"""Experiment configuration for PV power and solar-irradiance forecasting with DALF."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TARGETS = ("power", "irradiance")
TARGET_FOLDERS = {"power": "power_production", "irradiance": "solar_irradiance"}
STANDALONE_MODELS = ("RNN", "GRU", "LSTM")
HYBRID_MODELS = ("HCRM", "HCGM", "HCLM")
ALL_MODELS = STANDALONE_MODELS + HYBRID_MODELS
FUSION_METHODS = ("UEBF", "SCAF", "DALF")
KNOWN_OPTIMIZERS = ("adam", "nadam", "rmsprop", "sgd", "adamw")
KNOWN_ACTIVATIONS = ("leaky_relu", "relu", "elu", "selu", "tanh", "gelu", "swish", "linear")
POWER_UNITS = ("MW", "kW", "MWh", "kWh")
POWER_MEASUREMENTS = ("hourly_mean_power", "hourly_energy", "instantaneous")
FUSION_RESOLUTIONS = ("hourly", "daily_energy")


class ConfigError(ValueError):
    """Raised for an invalid or incomplete configuration."""


@dataclass
class PlantConfig:
    """One PV plant. Reference values are checks only, never targets for cleaning."""

    name: str
    capacity_mw: float
    site: str
    file_path: str
    sheet_name: str | int | None = None  
    reference_raw_rows: int | None = None
    reference_cleaned_rows: int | None = None
    reference_split_indices: dict | None = None  


def default_plants() -> list[PlantConfig]:
    return [
        PlantConfig(
            name="plant_50mw", capacity_mw=50.0,
            site="Nizam Energy Sindh PV Phase I Project No. 3, Sukkur, Pakistan",
            file_path="data/plant_50mw.csv", reference_raw_rows=17520, reference_cleaned_rows=17270,
            reference_split_indices={"train": [0, 12088], "validation": [12089, 13815], "test": [13816, 17269]},
        ),
        PlantConfig(
            name="plant_100mw", capacity_mw=100.0, site="Sapphire Solar, Chakwal, Pakistan",
            file_path="data/plant_100mw.csv", reference_raw_rows=17520, reference_cleaned_rows=17357,
            reference_split_indices={"train": [0, 12149], "validation": [12150, 13885], "test": [13886, 17356]},
        ),
    ]


@dataclass
class DataConfig:
    """Column mapping, timestamp convention and units (SITE SETTINGS)."""

    timestamp_col: str = "timestamp"
    power_col: str = "power"
    irradiance_col: str = "irradiance"
    temperature_col: str = "temperature"
    timestamp_format: str | None = None  
    timezone: str | None = None  
    timestamp_label: str = "period_start"  
    power_unit: str = "MW"  
    power_measurement: str = "hourly_mean_power"  
    irradiance_unit: str = "W/m²"
    temperature_unit: str = "°C"
    missing_value_codes: list[float] = field(default_factory=lambda: [-9999.0, -999.0])
    csv_separator: str = ","
    csv_decimal: str = "."
    csv_encoding: str | None = None


@dataclass
class CleaningConfig:
    """Physical-plausibility limits (SITE SETTINGS; confirm for each plant)."""

    power_negative_tolerance_fraction_of_capacity: float = 0.01  
    power_max_fraction_of_capacity: float = 1.10  
    irradiance_negative_tolerance: float = 10.0  
    irradiance_max: float = 1500.0  
    temperature_min: float = -20.0  
    temperature_max: float = 60.0  
    max_interpolation_gap_hours: int = 1  


@dataclass
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.10
    test_fraction: float = 0.20
    boundary_rounding: str = "round"  


@dataclass
class WindowConfig:
    input_hours: int = 168
    horizon_hours: int = 24
    training_origin_stride_hours: int = 1  


@dataclass
class ModelConfig:
    """IMPLEMENTATION DEFAULTS taken from the earlier study specification; align with the manuscript."""

    models: list[str] = field(default_factory=lambda: list(ALL_MODELS))
    conv_filters: int = 32
    kernel_size: int = 3
    conv_padding: str = "causal"
    recurrent_units: int = 64
    recurrent_layers: int = 1
    dense_units: list[int] = field(default_factory=lambda: [32])
    dropout: float = 0.0
    activation: str = "leaky_relu"
    leaky_relu_slope: float = 0.1
    loss: str = "mse"


@dataclass
class TrainingConfig:
    optimizer: str = "nadam"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    shuffle: bool = False
    seeds: list[int] = field(default_factory=lambda: [42, 1, 7, 13, 21, 33, 55, 77, 99, 123])
    deterministic_ops: bool = True
    lr_schedule: str | None = None  
    lr_schedule_factor: float = 0.5
    lr_schedule_patience: int = 5
    lr_schedule_min_lr: float = 1e-5
    prediction_batch_size: int = 256
    verbose: int = 2


@dataclass
class FusionConfig:
    base_models: list[str] = field(default_factory=lambda: list(HYBRID_MODELS))
    window_days: int = 30
    
    resolution: dict = field(default_factory=lambda: {"power": "daily_energy", "irradiance": "hourly"})
    objective: str = "mae"  
    window_definition: str = "last_n_completed_days"  
    min_history_days: int = 7  
    weight_tolerance: float = 1e-8
    scaf_mode: str = "estimate_on_validation"  
    scaf_reported_coefficients: dict = field(default_factory=lambda: {
        "plant_50mw": {"HCRM": 0.12, "HCGM": 0.27, "HCLM": 0.61},
        "plant_100mw": {"HCRM": 0.11, "HCGM": 0.24, "HCLM": 0.65},
    })
    scaf_reported_target: str | None = None  
    scaf_reported_confirmed: bool = False


@dataclass
class StatsConfig:
    enabled: bool = True
    alpha: float = 0.05
    loss_functions: list[str] = field(default_factory=lambda: ["absolute", "squared"])
    hac_lag: int | None = None  
    harvey_correction: bool = True
    wilcoxon: bool = True
    candidate: str = "DALF"
    baselines: list[str] = field(default_factory=lambda: ["UEBF", "SCAF", "HCRM", "HCGM", "HCLM", "RNN", "GRU", "LSTM"])


@dataclass
class AblationConfig:
    dalf_window_days: list[int] = field(default_factory=lambda: [7, 14, 30, 60])
    mse_dalf: bool = False  
    interpolation_limits_hours: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    input_masking_fractions: list[float] = field(default_factory=lambda: [0.05, 0.10])
    masking_seed: int = 2024
    activations: list[str] = field(default_factory=list)  
    optimizers: list[str] = field(default_factory=list)  
    variant_seeds: list[int] = field(default_factory=lambda: [42])


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    figure_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    excel: bool = True
    zoom_days: int = 14


@dataclass
class ExperimentConfig:
    plants: list[PlantConfig] = field(default_factory=default_plants)
    targets: list[str] = field(default_factory=lambda: list(TARGETS))
    data: DataConfig = field(default_factory=DataConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    site_settings_confirmed: bool = False
    synthetic_data: bool = False



def config_to_dict(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)


def _merge_dataclass(instance, values: dict, path: str):
    if not isinstance(values, dict):
        raise ConfigError(f"'{path}' must be a JSON object.")
    known = {f.name: f for f in dataclasses.fields(instance)}
    for key, value in values.items():
        if key.startswith("_"):  
            continue
        if key not in known:
            raise ConfigError(f"Unknown configuration key '{path}.{key}'. Known keys: {sorted(known)}")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            _merge_dataclass(current, value, f"{path}.{key}")
        elif key == "plants":
            setattr(instance, key, [p if isinstance(p, PlantConfig) else PlantConfig(**p) for p in value])
        else:
            setattr(instance, key, copy.deepcopy(value))
    return instance


def config_from_dict(values: dict) -> ExperimentConfig:
    """Apply a (partial) dictionary on top of the defaults."""
    try:
        return _merge_dataclass(ExperimentConfig(), values, "config")
    except TypeError as exc:  # wrong field names inside a plant entry
        raise ConfigError(f"Invalid plant entry: {exc}") from exc


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Configuration file not found: {source.resolve()}")
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{source} is not valid JSON: {exc}") from exc
    return config_from_dict(values)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config_to_dict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")



def energy_unit(cfg: ExperimentConfig) -> str | None:
    """Unit of daily energy, or None when daily aggregation is not valid for the data."""
    d = cfg.data
    if d.power_measurement == "hourly_mean_power" and d.power_unit in ("MW", "kW"):
        return f"{d.power_unit}h"
    if d.power_measurement == "hourly_energy" and d.power_unit in ("MWh", "kWh"):
        return d.power_unit
    return None


def capacity_in_power_unit(plant: PlantConfig, cfg: ExperimentConfig) -> float:
    """Plant capacity expressed in the power column's unit (per hour for energy units)."""
    return plant.capacity_mw * (1000.0 if cfg.data.power_unit in ("kW", "kWh") else 1.0)


def target_unit(cfg: ExperimentConfig, target: str) -> str:
    return cfg.data.power_unit if target == "power" else cfg.data.irradiance_unit


def daily_block_start_hour(cfg: ExperimentConfig) -> int:
    """First hour label of a calendar-day block (0 for period-start labels, 1 for period-end labels)."""
    return 0 if cfg.data.timestamp_label == "period_start" else 1


def site_settings_checklist(cfg: ExperimentConfig) -> list[str]:
    d, c = cfg.data, cfg.cleaning
    return [
        f"columns: timestamp='{d.timestamp_col}', power='{d.power_col}', irradiance='{d.irradiance_col}', "
        f"temperature='{d.temperature_col}'",
        f"timestamp format={d.timestamp_format!r}, time zone={d.timezone!r}, label convention='{d.timestamp_label}'",
        f"power unit='{d.power_unit}', power measurement='{d.power_measurement}', irradiance unit='{d.irradiance_unit}'",
        f"missing-value codes={d.missing_value_codes}",
        f"negative tolerance: power {c.power_negative_tolerance_fraction_of_capacity:.3f} x capacity, "
        f"irradiance {c.irradiance_negative_tolerance} {d.irradiance_unit}",
        f"upper limits: power {c.power_max_fraction_of_capacity} x capacity, irradiance {c.irradiance_max} "
        f"{d.irradiance_unit}; temperature range [{c.temperature_min}, {c.temperature_max}] {d.temperature_unit}",
        f"interpolation of isolated gaps up to {c.max_interpolation_gap_hours} h",
    ]


def boundary(n_rows: int, fraction: float, rounding: str) -> int:
    value = n_rows * fraction
    return int(math.floor(value + 0.5)) if rounding == "round" else int(math.floor(value + 1e-9))



def validate_config(cfg: ExperimentConfig) -> list[str]:
    """Return a list of problems (empty when the configuration is consistent)."""
    problems: list[str] = []
    s = cfg.split
    fractions = (s.train_fraction, s.validation_fraction, s.test_fraction)
    if any(f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        problems.append(f"split fractions must be positive and sum to 1 (got {fractions})")
    if s.boundary_rounding not in ("round", "floor"):
        problems.append("split.boundary_rounding must be 'round' or 'floor'")
    if not cfg.plants:
        problems.append("at least one plant is required")
    names = [p.name for p in cfg.plants]
    if len(set(names)) != len(names):
        problems.append("plant names must be unique")
    for p in cfg.plants:
        if p.capacity_mw <= 0:
            problems.append(f"plant {p.name}: capacity_mw must be positive")
    bad_targets = [t for t in cfg.targets if t not in TARGETS]
    if bad_targets or not cfg.targets:
        problems.append(f"targets must be a non-empty subset of {TARGETS} (got {cfg.targets})")
    d = cfg.data
    if d.timestamp_label not in ("period_start", "period_end"):
        problems.append("data.timestamp_label must be 'period_start' or 'period_end'")
    if d.power_unit not in POWER_UNITS:
        problems.append(f"data.power_unit must be one of {POWER_UNITS}")
    if d.power_measurement not in POWER_MEASUREMENTS:
        problems.append(f"data.power_measurement must be one of {POWER_MEASUREMENTS}")
    if d.power_measurement == "hourly_energy" and d.power_unit not in ("MWh", "kWh"):
        problems.append("hourly_energy measurements need an energy unit (MWh or kWh)")
    if d.power_measurement != "hourly_energy" and d.power_unit in ("MWh", "kWh"):
        problems.append("an energy unit (MWh/kWh) requires data.power_measurement='hourly_energy'")
    w = cfg.window
    if w.input_hours < 1 or w.horizon_hours < 1:
        problems.append("window lengths must be positive")
    if w.training_origin_stride_hours < 1 or 24 % w.training_origin_stride_hours != 0:
        problems.append("window.training_origin_stride_hours must divide 24")
    m = cfg.model
    unknown_models = [x for x in m.models if x not in ALL_MODELS]
    if unknown_models:
        problems.append(f"unknown models {unknown_models}; choose from {ALL_MODELS}")
    if m.activation not in KNOWN_ACTIVATIONS:
        problems.append(f"model.activation must be one of {KNOWN_ACTIVATIONS}")
    if not 0.0 <= m.dropout < 1.0:
        problems.append("model.dropout must be in [0, 1)")
    if m.recurrent_layers < 1 or m.recurrent_units < 1 or m.conv_filters < 1 or m.kernel_size < 1:
        problems.append("model sizes must be positive")
    if m.loss not in ("mse", "mae", "huber"):
        problems.append("model.loss must be 'mse', 'mae' or 'huber'")
    t = cfg.training
    if t.optimizer not in KNOWN_OPTIMIZERS:
        problems.append(f"training.optimizer must be one of {KNOWN_OPTIMIZERS}")
    if not t.seeds or len(set(t.seeds)) != len(t.seeds):
        problems.append("training.seeds must be a non-empty list without duplicates")
    if t.epochs < 1 or t.batch_size < 1 or t.learning_rate <= 0 or t.early_stopping_patience < 0:
        problems.append("training.epochs, batch_size, learning_rate and patience must be positive")
    if t.lr_schedule not in (None, "reduce_on_plateau"):
        problems.append("training.lr_schedule must be null or 'reduce_on_plateau'")
    f = cfg.fusion
    if len(f.base_models) < 2 or any(b not in m.models for b in f.base_models):
        problems.append("fusion.base_models must contain at least two trained models")
    if f.objective != "mae":
        problems.append("fusion.objective must be 'mae' for the main results (use ablation.mse_dalf for MSE)")
    if f.window_days < 1 or not 1 <= f.min_history_days <= f.window_days:
        problems.append("fusion.window_days >= 1 and 1 <= fusion.min_history_days <= window_days are required")
    if f.window_definition not in ("last_n_completed_days", "calendar_days"):
        problems.append("fusion.window_definition must be 'last_n_completed_days' or 'calendar_days'")
    for target in cfg.targets:
        res = f.resolution.get(target)
        if res not in FUSION_RESOLUTIONS:
            problems.append(f"fusion.resolution['{target}'] must be one of {FUSION_RESOLUTIONS}")
        if res == "daily_energy" and target != "power":
            problems.append("daily_energy fusion resolution is defined only for the power target")
        if res == "daily_energy" and energy_unit(cfg) is None:
            problems.append("daily_energy resolution needs hourly mean power (MW/kW) or hourly energy (MWh/kWh); "
                            "instantaneous readings cannot be summed into energy")
        if res == "daily_energy" and w.horizon_hours != 24:
            problems.append("daily_energy resolution requires horizon_hours = 24")
    if f.scaf_mode not in ("estimate_on_validation", "reported"):
        problems.append("fusion.scaf_mode must be 'estimate_on_validation' or 'reported'")
    if f.scaf_mode == "reported":
        if f.scaf_reported_target not in TARGETS or not f.scaf_reported_confirmed:
            problems.append("reported SCAF coefficients need fusion.scaf_reported_target set to 'power' or "
                            "'irradiance' and fusion.scaf_reported_confirmed = true")
        for plant in cfg.plants:
            coeffs = f.scaf_reported_coefficients.get(plant.name)
            if coeffs is None or sorted(coeffs) != sorted(f.base_models) \
                    or not math.isclose(sum(coeffs.values()), 1.0, abs_tol=0.011):
                problems.append(f"reported SCAF coefficients for {plant.name} must cover {f.base_models} and sum to 1")
    st = cfg.stats
    if not 0 < st.alpha < 1:
        problems.append("stats.alpha must be between 0 and 1")
    if any(x not in ("absolute", "squared") for x in st.loss_functions):
        problems.append("stats.loss_functions must contain 'absolute' and/or 'squared'")
    a = cfg.ablation
    if any(x < 1 for x in a.dalf_window_days) or any(x < 0 for x in a.interpolation_limits_hours):
        problems.append("ablation window lengths must be >= 1 and interpolation limits >= 0")
    if any(not 0 < x < 0.5 for x in a.input_masking_fractions):
        problems.append("ablation.input_masking_fractions must lie in (0, 0.5)")
    if any(x not in KNOWN_ACTIVATIONS for x in a.activations) or any(x not in KNOWN_OPTIMIZERS for x in a.optimizers):
        problems.append("ablation activations/optimizers must be known names")
    if cfg.cleaning.max_interpolation_gap_hours < 0:
        problems.append("cleaning.max_interpolation_gap_hours must be >= 0")
    o = cfg.output
    if any(x not in ("png", "pdf", "svg") for x in o.figure_formats) or "png" not in o.figure_formats:
        problems.append("output.figure_formats must include 'png' and may add 'pdf' or 'svg'")
    if o.dpi < 72:
        problems.append("output.dpi must be at least 72")
    return problems


def require_valid(cfg: ExperimentConfig) -> None:
    problems = validate_config(cfg)
    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def require_confirmed_site_settings(cfg: ExperimentConfig) -> None:
    """Real-data training and evaluation require explicitly confirmed site settings."""
    if cfg.synthetic_data or cfg.site_settings_confirmed:
        return
    raise ConfigError(
        "Site settings have not been confirmed. Check these values against the data documentation for "
        "each plant, then set site_settings_confirmed=true in the JSON configuration or pass "
        "--confirm-site-settings:\n  - " + "\n  - ".join(site_settings_checklist(cfg))
    )


def summary_json(cfg: ExperimentConfig) -> dict[str, Any]:
    return config_to_dict(cfg)

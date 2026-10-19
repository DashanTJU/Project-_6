

from __future__ import annotations

import difflib
from pathlib import Path

import pandas as pd

from config import DataConfig, PlantConfig

CSV_SUFFIXES = {".csv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


class DataValidationError(ValueError):
    """Raised when an input file is missing, unreadable or lacks required columns."""


def load_table(plant: PlantConfig, data_cfg: DataConfig) -> tuple[pd.DataFrame, dict]:
    """Read one plant file; returns the raw table and a description of what was read."""
    source = Path(plant.file_path)
    if not source.is_file():
        raise DataValidationError(
            f"Data file for {plant.name} not found: {source.resolve()}\n"
            "Set PlantConfig.file_path in your configuration (or place the file at that path)."
        )
    suffix = source.suffix.lower()
    info = {"plant": plant.name, "file": str(source), "format": suffix.lstrip("."), "sheet": None}
    if suffix in CSV_SUFFIXES:
        table = pd.read_csv(source, sep=data_cfg.csv_separator, decimal=data_cfg.csv_decimal,
                            encoding=data_cfg.csv_encoding)
    elif suffix in EXCEL_SUFFIXES:
        try:
            workbook = pd.ExcelFile(source)
        except ImportError as exc:
            engine = "xlrd" if suffix == ".xls" else "openpyxl"
            raise DataValidationError(f"Reading {suffix} files needs the '{engine}' package: pip install {engine}") from exc
        sheets = workbook.sheet_names
        sheet = plant.sheet_name if plant.sheet_name is not None else sheets[0]
        if isinstance(sheet, str) and sheet not in sheets:
            raise DataValidationError(f"Sheet '{sheet}' not found in {source.name}; available sheets: {sheets}")
        if isinstance(sheet, int) and not 0 <= sheet < len(sheets):
            raise DataValidationError(f"Sheet index {sheet} is out of range for {source.name}; sheets: {sheets}")
        table = workbook.parse(sheet)
        info.update(sheet=sheet if isinstance(sheet, str) else sheets[sheet], available_sheets=sheets,
                    sheet_was_configured=plant.sheet_name is not None)
    else:
        raise DataValidationError(f"Unsupported file type '{suffix}' for {source.name}; use CSV or Excel.")
    info["raw_rows"] = int(len(table))
    info["columns"] = [str(c) for c in table.columns]
    return table, info


def required_columns(data_cfg: DataConfig, target: str | None = None) -> dict[str, str]:
    """Role -> configured column name. With a target, only the columns that target needs."""
    roles = {"timestamp": data_cfg.timestamp_col, "power": data_cfg.power_col,
             "irradiance": data_cfg.irradiance_col, "temperature": data_cfg.temperature_col}
    if target == "irradiance":
        roles.pop("power")
    return roles


def column_diagnostics(table: pd.DataFrame, roles: dict[str, str]) -> dict:
    """Which configured columns exist, and close matches for those that do not."""
    available = [str(c) for c in table.columns]
    rows = []
    for role, column in roles.items():
        found = column in available
        suggestions = [] if found else difflib.get_close_matches(column, available, n=3, cutoff=0.4)
        if not found:
            suggestions += [c for c in available if c.lower() == column.lower() and c not in suggestions]
        rows.append({"role": role, "configured_column": column, "found": found, "suggestions": suggestions})
    return {"available_columns": available, "required": rows, "missing": [r for r in rows if not r["found"]]}


def require_columns(table: pd.DataFrame, roles: dict[str, str], source: str) -> None:
    diagnostics = column_diagnostics(table, roles)
    if not diagnostics["missing"]:
        return
    lines = [f"Required columns are missing in {source}.", f"Available columns: {diagnostics['available_columns']}"]
    for row in diagnostics["missing"]:
        hint = f" (similar: {row['suggestions']})" if row["suggestions"] else ""
        lines.append(f"  - {row['role']}: configured as '{row['configured_column']}'{hint}")
    lines.append("Edit the column names in the 'data' section of the configuration.")
    raise DataValidationError("\n".join(lines))

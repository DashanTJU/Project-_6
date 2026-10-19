

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, ModelConfig, TrainingConfig
from models import _tf, build_model, model_family, model_summary_text, parameter_counts
from preprocessing import inverse_target
from window_generator import WindowSet


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _tf().keras.utils.set_random_seed(seed)


def configure_determinism(enabled: bool) -> dict:
    """Request deterministic TensorFlow kernels where supported."""
    if not enabled:
        return {"deterministic_ops": False, "note": "disabled in configuration"}
    try:
        _tf().config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as exc:
        return {"deterministic_ops": False, "note": f"not supported: {exc}"}
    return {"deterministic_ops": True, "note": "tf.config.experimental.enable_op_determinism()"}


def mask_input_windows(X: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Hide a fraction of interior input hours per window and refill them by linear interpolation.

    Used only for the inference-time missing-data sensitivity experiment.
    """
    masked = np.array(X, dtype=np.float32, copy=True)
    n_windows, n_hours, n_features = masked.shape
    positions = np.arange(n_hours)
    n_hidden = max(1, int(round(fraction * (n_hours - 2))))
    for i in range(n_windows):
        hidden = rng.choice(np.arange(1, n_hours - 1), size=n_hidden, replace=False)
        keep = np.setdiff1d(positions, hidden)
        for f in range(n_features):
            masked[i, hidden, f] = np.interp(hidden, keep, masked[i, keep, f])
    return masked


def run_fingerprint(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _callbacks(checkpoint: Path, history_csv: Path, tcfg: TrainingConfig):
    tf = _tf()
    cb = tf.keras.callbacks
    callbacks = [
        cb.EarlyStopping(monitor="val_loss", patience=tcfg.early_stopping_patience, restore_best_weights=True,
                         verbose=1),
        cb.ModelCheckpoint(filepath=str(checkpoint), monitor="val_loss", save_best_only=True, verbose=0),
        cb.CSVLogger(str(history_csv)),
        cb.TerminateOnNaN(),
    ]
    if tcfg.lr_schedule == "reduce_on_plateau":
        callbacks.append(cb.ReduceLROnPlateau(monitor="val_loss", factor=tcfg.lr_schedule_factor,
                                              patience=tcfg.lr_schedule_patience, min_lr=tcfg.lr_schedule_min_lr,
                                              verbose=1))
    return callbacks


def train_single_run(*, plant: str, target: str, model_type: str, seed: int, windows: dict[str, WindowSet],
                     target_scaler, cfg: ExperimentConfig, run_dirs: dict[str, Path], variant: str = "main",
                     model_cfg: ModelConfig | None = None, training_cfg: TrainingConfig | None = None,
                     masking_fractions: tuple[float, ...] = (), resume: bool = False, logger=None) -> dict:
    """Train (or reload) one run. Returns {'record', 'history', 'predictions'} with physical-unit predictions."""
    mcfg = model_cfg or cfg.model
    tcfg = training_cfg or cfg.training
    run_name = f"{model_type}_{variant}_seed{seed}"
    checkpoint = run_dirs["models"] / f"{run_name}.keras"
    history_csv = run_dirs["histories"] / f"{run_name}_history.csv"
    record_json = run_dirs["histories"] / f"{run_name}_record.json"
    predictions_npz = run_dirs["predictions"] / f"{run_name}.npz"
    fingerprint = run_fingerprint({
        "plant": plant, "target": target, "model": model_type, "seed": seed, "variant": variant,
        "model_cfg": dataclasses.asdict(mcfg),
        "training_cfg": {k: v for k, v in dataclasses.asdict(tcfg).items() if k not in ("seeds", "verbose")},
        "window": dataclasses.asdict(cfg.window), "masking": list(masking_fractions),
        "train_windows": windows["train"].n, "origins": [int(windows[p].origin_times.astype(np.int64).sum())
                                                         for p in ("train", "validation", "test")],
    })
    if resume and checkpoint.exists() and history_csv.exists() and record_json.exists() and predictions_npz.exists():
        record = json.loads(record_json.read_text(encoding="utf-8"))
        if record.get("fingerprint") == fingerprint:
            with np.load(predictions_npz) as data:
                preds = {"validation": data["validation"], "test": data["test"],
                         "test_masked": {float(f): data[f"test_masked_{f}"] for f in masking_fractions}}
            if logger:
                logger.info("  reused finished run %s", run_name)
            return {"record": {**record, "reused": True}, "history": pd.read_csv(history_csv), "predictions": preds}

    tf = _tf()
    tf.keras.backend.clear_session()
    set_global_seed(seed)
    for path in (checkpoint, history_csv):
        if path.exists():
            path.unlink()
    model = build_model(model_type, windows["train"].X.shape[2], mcfg, tcfg, cfg.window.input_hours,
                        cfg.window.horizon_hours)
    run_dirs["summaries"].mkdir(parents=True, exist_ok=True)
    summary_file = run_dirs["summaries"] / f"{model_type}_{variant}_summary.txt"
    if not summary_file.exists():
        summary_file.write_text(model_summary_text(model), encoding="utf-8")
    params = parameter_counts(model)
    started = time.perf_counter()
    try:
        fit = model.fit(windows["train"].X, windows["train"].Y_scaled,
                        validation_data=(windows["validation"].X, windows["validation"].Y_scaled),
                        epochs=tcfg.epochs, batch_size=tcfg.batch_size, shuffle=tcfg.shuffle,
                        callbacks=_callbacks(checkpoint, history_csv, tcfg), verbose=tcfg.verbose)
    except Exception as exc:
        raise RuntimeError(f"Training failed for {plant}/{target}/{run_name}: {exc}") from exc
    training_seconds = time.perf_counter() - started
    if not checkpoint.exists():
        raise RuntimeError(f"No checkpoint saved for {run_name}: the validation loss never became finite.")
    best = tf.keras.models.load_model(checkpoint)
    history = pd.DataFrame(fit.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    history.to_csv(history_csv, index=False)

    def predict(X):
        out = np.asarray(best.predict(X, batch_size=tcfg.prediction_batch_size, verbose=0), dtype=float)
        if out.shape != (X.shape[0], cfg.window.horizon_hours) or not np.all(np.isfinite(out)):
            raise ValueError(f"{run_name}: invalid predictions with shape {out.shape}")
        return out

    t0 = time.perf_counter()
    pred_val = predict(windows["validation"].X)
    t1 = time.perf_counter()
    pred_test = predict(windows["test"].X)
    t2 = time.perf_counter()
    rng = np.random.default_rng(cfg.ablation.masking_seed)
    masked = {f: inverse_target(predict(mask_input_windows(windows["test"].X, f, rng)), target_scaler)
              for f in masking_fractions}
    predictions = {"validation": inverse_target(pred_val, target_scaler),
                   "test": inverse_target(pred_test, target_scaler), "test_masked": masked}
    np.savez_compressed(predictions_npz, validation=predictions["validation"], test=predictions["test"],
                        validation_origin_ns=windows["validation"].origin_times.astype(np.int64),
                        test_origin_ns=windows["test"].origin_times.astype(np.int64),
                        **{f"test_masked_{f}": v for f, v in masked.items()})
    val_loss = history["val_loss"].to_numpy(dtype=float)
    best_epoch = int(np.nanargmin(val_loss)) + 1 if np.isfinite(val_loss).any() else None
    record = {
        "plant": plant, "target": target, "model": model_type, "family": model_family(model_type),
        "variant": variant, "seed": seed, "fingerprint": fingerprint, "reused": False,
        "epochs_run": int(len(history)), "best_epoch": best_epoch,
        "best_val_loss": float(np.nanmin(val_loss)) if best_epoch else None,
        "training_seconds": training_seconds, **params,
        "model_file": str(checkpoint), "model_file_bytes": int(checkpoint.stat().st_size),
        "inference_seconds_validation": t1 - t0, "inference_seconds_test": t2 - t1,
        "inference_ms_per_test_window": 1000.0 * (t2 - t1) / max(windows["test"].n, 1),
        "train_windows": windows["train"].n, "validation_windows": windows["validation"].n,
        "test_windows": windows["test"].n, "activation": mcfg.activation, "optimizer": tcfg.optimizer,
        "learning_rate": tcfg.learning_rate, "batch_size": tcfg.batch_size, "max_epochs": tcfg.epochs,
        "model_config": dataclasses.asdict(mcfg),
    }
    record_json.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    del model, best
    tf.keras.backend.clear_session()
    return {"record": record, "history": history, "predictions": predictions}



from __future__ import annotations

import inspect

import numpy as np

from config import HYBRID_MODELS, STANDALONE_MODELS, ModelConfig, TrainingConfig

RECURRENT_LAYER = {"RNN": "SimpleRNN", "GRU": "GRU", "LSTM": "LSTM",
                   "HCRM": "SimpleRNN", "HCGM": "GRU", "HCLM": "LSTM"}


def _tf():
    import tensorflow as tf 
    return tf


def model_family(model_type: str) -> str:
    return "hybrid CNN-recurrent" if model_type in HYBRID_MODELS else "standalone recurrent"


def architecture_spec(model_type: str, mcfg: ModelConfig, n_features: int, input_hours: int,
                      horizon: int) -> list[dict]:
    """Layer list used for documentation tables and the architecture figure (no TensorFlow needed)."""
    if model_type not in RECURRENT_LAYER:
        raise ValueError(f"Unknown model type {model_type}")
    act = mcfg.activation if mcfg.activation != "leaky_relu" else f"LeakyReLU({mcfg.leaky_relu_slope})"
    layers = [{"layer": "Input", "details": f"{input_hours} h x {n_features} features"}]
    if model_type in HYBRID_MODELS:
        layers.append({"layer": "Conv1D", "details": f"{mcfg.conv_filters} filters, kernel {mcfg.kernel_size}, "
                                                     f"{mcfg.conv_padding} padding, {act}"})
        if mcfg.dropout > 0:
            layers.append({"layer": "Dropout", "details": f"rate {mcfg.dropout}"})
    for i in range(mcfg.recurrent_layers):
        layers.append({"layer": RECURRENT_LAYER[model_type],
                       "details": f"{mcfg.recurrent_units} units" + (" (returns sequence)" if i < mcfg.recurrent_layers - 1 else "")})
    for units in mcfg.dense_units:
        layers.append({"layer": "Dense", "details": f"{units} units, {act}"})
        if mcfg.dropout > 0:
            layers.append({"layer": "Dropout", "details": f"rate {mcfg.dropout}"})
    layers.append({"layer": "Dense (output)", "details": f"{horizon} units, linear (24 hourly forecasts)"})
    return layers


def activation_layer(name: str, slope: float, layer_name: str):
    tf = _tf()
    if name == "leaky_relu":
        params = inspect.signature(tf.keras.layers.LeakyReLU.__init__).parameters
        key = "negative_slope" if "negative_slope" in params else "alpha"  # Keras 3 / Keras 2
        return tf.keras.layers.LeakyReLU(**{key: slope}, name=layer_name)
    return tf.keras.layers.Activation(name, name=layer_name)


def make_optimizer(name: str, learning_rate: float):
    tf = _tf()
    opts = tf.keras.optimizers
    factories = {"adam": lambda: opts.Adam(learning_rate=learning_rate),
                 "nadam": lambda: opts.Nadam(learning_rate=learning_rate),
                 "rmsprop": lambda: opts.RMSprop(learning_rate=learning_rate),
                 "sgd": lambda: opts.SGD(learning_rate=learning_rate, momentum=0.9),
                 "adamw": lambda: opts.AdamW(learning_rate=learning_rate)}
    if name not in factories:
        raise ValueError(f"Unknown optimizer {name}")
    return factories[name]()


def build_model(model_type: str, n_features: int, mcfg: ModelConfig, tcfg: TrainingConfig, input_hours: int = 168,
                horizon: int = 24):
    """Build and compile one model with the shared configuration."""
    tf = _tf()
    layers = tf.keras.layers
    if model_type not in RECURRENT_LAYER:
        raise ValueError(f"Unknown model type {model_type}; choose from {STANDALONE_MODELS + HYBRID_MODELS}")
    inputs = tf.keras.Input(shape=(input_hours, n_features), name="history")
    x = inputs
    if model_type in HYBRID_MODELS:
        x = layers.Conv1D(mcfg.conv_filters, mcfg.kernel_size, padding=mcfg.conv_padding,
                          kernel_initializer="glorot_uniform", name="temporal_conv1d")(x)
        x = activation_layer(mcfg.activation, mcfg.leaky_relu_slope, "conv_activation")(x)
        if mcfg.dropout > 0:
            x = layers.Dropout(mcfg.dropout, name="conv_dropout")(x)
    recurrent = getattr(layers, RECURRENT_LAYER[model_type])
    for i in range(mcfg.recurrent_layers):
        x = recurrent(mcfg.recurrent_units, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
                      return_sequences=i < mcfg.recurrent_layers - 1, name=f"recurrent_{i + 1}")(x)
    for j, units in enumerate(mcfg.dense_units):
        x = layers.Dense(units, kernel_initializer="glorot_uniform", name=f"dense_{j + 1}")(x)
        x = activation_layer(mcfg.activation, mcfg.leaky_relu_slope, f"dense_{j + 1}_activation")(x)
        if mcfg.dropout > 0:
            x = layers.Dropout(mcfg.dropout, name=f"dense_{j + 1}_dropout")(x)
    outputs = layers.Dense(horizon, activation="linear", kernel_initializer="glorot_uniform", name="forecast_24h")(x)
    model = tf.keras.Model(inputs, outputs, name=model_type)
    model.compile(optimizer=make_optimizer(tcfg.optimizer, tcfg.learning_rate), loss=mcfg.loss,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    if tuple(model.output_shape) != (None, horizon):
        raise ValueError(f"{model_type} output shape {model.output_shape} is not (None, {horizon})")
    return model


def model_summary_text(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lambda line, *args, **kwargs: lines.append(line))
    return "\n".join(lines)


def parameter_counts(model) -> dict:
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    non_trainable = int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights))
    return {"trainable_parameters": trainable, "non_trainable_parameters": non_trainable,
            "total_parameters": trainable + non_trainable}

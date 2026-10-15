"""Train and select the requested TensorFlow multilayer perceptron."""

from __future__ import annotations

import json

import joblib
import numpy as np

from src.config import MODEL_DIR, SEED, ensure_project_dirs
from src.data_types import PreparedData

SETTINGS = [
    {"learning_rate": 0.001, "first_layer_units": 64},
    {"learning_rate": 0.0005, "first_layer_units": 64},
    {"learning_rate": 0.001, "first_layer_units": 32},
]


def _load_tensorflow():
    try:
        import tensorflow as tf
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TensorFlow could not be imported. Use Python 3.10-3.12 (recommended: "
            "`py -3.11 -m venv .venv`), activate `.venv`, and install requirements.txt."
        ) from error
    return tf


def train_model(prepared: PreparedData | None = None) -> dict:
    """Compare three small configurations using validation loss only."""
    ensure_project_dirs()
    tf = _load_tensorflow()
    if prepared is None:
        prepared = joblib.load(MODEL_DIR / "prepared_data.joblib")
    tf.keras.utils.set_random_seed(SEED)
    ratio = max(np.bincount(prepared.y_train.astype(int))) / min(np.bincount(prepared.y_train.astype(int)))
    class_weight = None
    if ratio >= 2.0:
        counts = np.bincount(prepared.y_train.astype(int))
        class_weight = {index: float(len(prepared.y_train) / (2 * count)) for index, count in enumerate(counts)}

    history_by_setting: dict[str, dict] = {}
    candidates = []
    for index, setting in enumerate(SETTINGS, start=1):
        tf.keras.utils.set_random_seed(SEED)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(prepared.X_train.shape[1],)),
                tf.keras.layers.Dense(setting["first_layer_units"], activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=setting["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        candidate_path = MODEL_DIR / f"candidate_{index}.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(candidate_path), monitor="val_loss", save_best_only=True
            ),
        ]
        history = model.fit(
            prepared.X_train,
            prepared.y_train,
            validation_data=(prepared.X_val, prepared.y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0,
        )
        best_epoch = int(np.argmin(history.history["val_loss"]))
        best_val_loss = float(history.history["val_loss"][best_epoch])
        setting_key = f"lr={setting['learning_rate']},units={setting['first_layer_units']}"
        history_by_setting[setting_key] = {
            key: [float(value) for value in values] for key, values in history.history.items()
        }
        candidates.append((best_val_loss, candidate_path, setting, best_epoch + 1, setting_key))
        print(
            f"Setting {index}/{len(SETTINGS)}: {setting_key}; "
            f"best validation loss={best_val_loss:.4f} at epoch {best_epoch + 1}"
        )

    best_val_loss, best_path, _, best_epoch, best_key = min(candidates, key=lambda item: item[0])
    selected_model = tf.keras.models.load_model(best_path)
    selected_model.save(MODEL_DIR / "best_model.keras")
    for _, candidate_path, _, _, _ in candidates:
        candidate_path.unlink(missing_ok=True)
    history_output = {
        "selected_setting": best_key,
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "settings": history_by_setting,
        "selected_history": history_by_setting[best_key],
        "class_weights": class_weight,
    }
    (MODEL_DIR / "training_history.json").write_text(json.dumps(history_output, indent=2), encoding="utf-8")
    print(f"Selected configuration by validation loss: {best_key} (loss={best_val_loss:.4f})")
    return history_output


if __name__ == "__main__":
    train_model()

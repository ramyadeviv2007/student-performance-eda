"""Evaluate the selected network and write test metrics and plots."""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import MODEL_DIR, ensure_project_dirs
from src.train_model import _load_tensorflow


def evaluate_model() -> dict:
    """Evaluate exactly once on the held-out test set and calculate validation importance."""
    ensure_project_dirs()
    tf = _load_tensorflow()
    prepared = joblib.load(MODEL_DIR / "prepared_data.joblib")
    model = tf.keras.models.load_model(MODEL_DIR / "best_model.keras")
    probabilities = model.predict(prepared.X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype("int64")
    y_test = prepared.y_test.to_numpy(dtype="int64")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=["fail", "pass"], output_dict=True, zero_division=0
        ),
        "decision_threshold": 0.5,
        "test_set_note": "The test set is small (about 60 students); metrics may be noisy.",
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    history = json.loads((MODEL_DIR / "training_history.json").read_text(encoding="utf-8"))["selected_history"]
    sns.set_theme(style="whitegrid")
    for metric, title, filename in (
        ("loss", "Training and Validation Loss", "loss_curves.png"),
        ("accuracy", "Training and Validation Accuracy", "accuracy_curves.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history[metric], label="Training")
        ax.plot(history[f"val_{metric}"], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=metric.title())
        ax.legend()
        fig.tight_layout()
        fig.savefig(MODEL_DIR / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
    ax.set(title="Test Confusion Matrix", xlabel="Predicted class", ylabel="Actual class")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    false_positive, true_positive, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_positive, true_positive, label=f"Test ROC-AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(title="Receiver Operating Characteristic", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

    baseline = float(
        np.mean(
            (model.predict(prepared.X_val, verbose=0).ravel() >= 0.5)
            == prepared.y_val.to_numpy(dtype="int64")
        )
    )
    rng = np.random.default_rng(42)
    importances = []
    for column in prepared.feature_names:
        differences = []
        for _ in range(10):
            permuted = prepared.raw_X_val.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            transformed = prepared.transformer.transform(permuted).astype("float32")
            score = np.mean(
                (model.predict(transformed, verbose=0).ravel() >= 0.5)
                == prepared.y_val.to_numpy(dtype="int64")
            )
            differences.append(baseline - float(score))
        importances.append({"feature": column, "mean_accuracy_decrease": float(np.mean(differences))})
    importance_frame = pd.DataFrame(importances).sort_values("mean_accuracy_decrease", ascending=True)
    importance_frame.to_csv(MODEL_DIR / "permutation_importance.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, max(5, len(importance_frame) * 0.28)))
    ax.barh(importance_frame["feature"], importance_frame["mean_accuracy_decrease"], color="#4c72b0")
    ax.set(
        title="Validation Permutation Importance (Not Causal)",
        xlabel="Decrease in validation accuracy after permutation",
        ylabel="Original input feature",
    )
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "permutation_importance.png", dpi=200)
    plt.close(fig)

    print(
        f"Test results ({metrics['test_samples']} students): accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}"
    )
    print("Confusion matrix (actual rows: fail, pass; predicted columns: fail, pass):")
    print(matrix)
    return metrics


if __name__ == "__main__":
    evaluate_model()

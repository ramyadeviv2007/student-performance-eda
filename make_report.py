#!/usr/bin/env python3
"""
Student Performance Analysis and Prediction (EDA + Deep Learning)
One-command project: downloads data, runs EDA, trains an MLP, evaluates it,
saves all figures, and writes report/Project_Report.docx.

Usage (from the project folder):
    python make_report.py
Optional:
    python make_report.py --data path/to/student-mat.csv --out some_folder

Every number in the report is computed here. If TensorFlow is missing, the
model sections of the report are marked PENDING (nothing is invented).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------------------------------------------------------
# Student details (edit here if needed)
# ----------------------------------------------------------------------------
STUDENT_NAME = "RAMYADEVI V"
ROLL_NO = "24AD086"
COLLEGE = "[College Name]"
DEPARTMENT = "[Department]"
ACADEMIC_YEAR = "[Academic Year]"
SUBMISSION_DATE = "[Submission Date]"
GITHUB_LINK = "[GitHub repository link]"

SEED = 42
PASS_THRESHOLD = 10
LEAK_COLS = ["G1", "G2", "G3"]
DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
STUDY_LABELS = {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}
ROOT = Path(__file__).resolve().parent
sns.set_theme(style="whitegrid", font_scale=1.05)


# ----------------------------------------------------------------------------
# Stage 1: acquire + parse
# ----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load student-mat.csv (semicolon separated); download it if missing."""
    if not path.exists():
        print("[1] Downloading dataset from UCI ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(DATA_URL, timeout=60) as r:
                zf = zipfile.ZipFile(io.BytesIO(r.read()))
            path.write_bytes(zf.read("student-mat.csv"))
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Download failed ({exc}).\nManually download the zip from "
                     "https://archive.ics.uci.edu/dataset/320/student+performance, "
                     f"extract student-mat.csv and save it as {path}")
    df = pd.read_csv(path, sep=";")
    if df.shape[1] != 33:
        sys.exit(f"Unexpected shape {df.shape}; expected 33 columns. Check delimiter/file.")
    return df


def data_summary(df: pd.DataFrame, out: Path) -> dict:
    """Print and save the data-understanding outputs."""
    num = df.select_dtypes("number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    info = {
        "rows": int(df.shape[0]), "cols": int(df.shape[1]),
        "numeric_cols": len(num), "categorical_cols": len(cat),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "grades_out_of_range": int(((df[LEAK_COLS] < 0) | (df[LEAK_COLS] > 20)).sum().sum()),
    }
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([df.head(5), df.tail(5)]).to_csv(out / "head_tail.csv", index=False)
    pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum()}).to_csv(
        out / "columns_dtypes_missing.csv")
    df.describe().T.assign(skew=df[num].skew()).to_csv(out / "numeric_summary.csv")
    (out / "data_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("[2] Data summary:", info)
    return info


# ----------------------------------------------------------------------------
# Stage 2: filter / preprocess
# ----------------------------------------------------------------------------
def make_target(df: pd.DataFrame) -> pd.Series:
    """Pass (1) if G3 >= 10 else Fail (0). The project's chosen definition."""
    return (df["G3"] >= PASS_THRESHOLD).astype(int)


def outlier_counts(df: pd.DataFrame) -> pd.DataFrame:
    """IQR outlier counts for numeric non-grade columns (observations are kept)."""
    rows = []
    for c in df.select_dtypes("number").columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        rows.append({"column": c, "iqr_outliers": n})
    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = make_target(df)
    X = df.drop(columns=LEAK_COLS)
    assert not set(LEAK_COLS) & set(X.columns), "Leakage: G1/G2/G3 in features"
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 70/15/15 split."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp,
                                          random_state=SEED)
    return Xtr, Xva, Xte, ytr, yva, yte


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


# ----------------------------------------------------------------------------
# Stage 3/4/5: mine + represent + refine (EDA figures)
# ----------------------------------------------------------------------------
def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_eda(df: pd.DataFrame, figdir: Path, outdir: Path) -> dict:
    """Create the 5 required figures + class distribution; return computed facts."""
    facts: dict = {}
    g3 = df["G3"]
    df = df.copy()
    df["pass_fail"] = make_target(df)
    df["study_label"] = df["studytime"].map(STUDY_LABELS)
    order = list(STUDY_LABELS.values())

    # 1. histogram + KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(g3, bins=21, kde=True, color="#4C72B0", ax=ax)
    ax.axvline(PASS_THRESHOLD, color="red", ls="--", label=f"Pass mark (G3 = {PASS_THRESHOLD})")
    ax.set(title="Distribution of Final Mathematics Grade (G3)", xlabel="Final grade G3 (0-20)",
           ylabel="Number of students")
    ax.legend()
    save_fig(fig, figdir / "fig1_g3_histogram_kde.png")
    facts["fig1"] = (f"G3 has mean {g3.mean():.2f}, median {g3.median():.1f}, standard deviation "
                     f"{g3.std():.2f} and skewness {g3.skew():.2f}. {int((g3 == 0).sum())} "
                     f"students ({(g3 == 0).mean():.1%}) scored 0, which creates a second small "
                     "peak at the left of the distribution.")

    # 2. boxplot G3 by studytime
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="study_label", y="G3", order=order, hue="study_label", legend=False,
                palette="Blues", ax=ax)
    ax.set(title="Final Grade (G3) by Weekly Study Time", xlabel="Weekly study time",
           ylabel="Final grade G3")
    save_fig(fig, figdir / "fig2_g3_by_studytime_boxplot.png")
    med = df.groupby("study_label")["G3"].median().reindex(order)
    groups = [df.loc[df.studytime == k, "G3"] for k in STUDY_LABELS]
    kw_stat, kw_p = stats.kruskal(*groups)
    facts["kruskal"] = {"H": float(kw_stat), "p": float(kw_p)}
    facts["fig2"] = ("Median G3 by study time: " + ", ".join(f"{k} = {v:.1f}" for k, v in med.items())
                     + f". A Kruskal-Wallis test (non-parametric, used because G3 is not normally "
                       f"distributed) gave H = {kw_stat:.2f}, p = {kw_p:.4f}.")

    # 3. correlation heatmap
    num = df.select_dtypes("number").drop(columns=["pass_fail"])
    corr = num.corr(method="pearson")
    corr.to_csv(outdir / "pearson_correlation.csv")
    num.corr(method="spearman").to_csv(outdir / "spearman_correlation.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".1f", annot_kws={"size": 6},
                square=True, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set(title="Correlation Heatmap of Numeric Features (Pearson)")
    save_fig(fig, figdir / "fig3_correlation_heatmap.png")
    c = corr["G3"].drop("G3")
    c_model = c.drop(labels=[x for x in LEAK_COLS if x in c.index])
    top = c_model.reindex(c_model.abs().sort_values(ascending=False).index)[:3]
    facts["fig3"] = (f"G1 (r = {c['G1']:.2f}) and G2 (r = {c['G2']:.2f}) are almost the same as G3, "
                     "which is why they are excluded from the model. Among the allowed features the "
                     "strongest Pearson correlations with G3 are "
                     + ", ".join(f"{k} (r = {v:.2f})" for k, v in top.items()) + ".")

    # 4. scatter absences vs G3
    rho, p = stats.spearmanr(df["absences"], g3)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x="absences", y="G3", scatter_kws={"alpha": 0.5, "s": 25},
                line_kws={"color": "red", "label": f"Trend line (Spearman rho = {rho:.2f})"}, ax=ax)
    ax.set(title="Absences versus Final Grade (G3)", xlabel="Number of school absences",
           ylabel="Final grade G3")
    ax.legend()
    save_fig(fig, figdir / "fig4_absences_vs_g3_scatter.png")
    facts["fig4"] = (f"Spearman rho = {rho:.2f} (p = {p:.4f}) between absences and G3. "
                     f"The maximum number of absences is {int(df.absences.max())}; "
                     f"{int((df.absences == 0).sum())} students had none.")
    facts["spearman_absences"] = {"rho": float(rho), "p": float(p)}

    # 5. bar chart mean G3 by studytime
    agg = df.groupby("study_label")["G3"].agg(["mean", "std", "count"]).reindex(order)
    agg.to_csv(outdir / "g3_by_studytime.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agg.index, agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]), capsize=5,
           color="#55A868", label="Mean G3 (error bar = standard error)")
    for i, (m, n) in enumerate(zip(agg["mean"], agg["count"])):
        ax.text(i, m + 0.3, f"{m:.1f}\n(n={n})", ha="center", fontsize=9)
    ax.set(title="Mean Final Grade (G3) by Weekly Study Time", xlabel="Weekly study time",
           ylabel="Mean G3", ylim=(0, agg["mean"].max() + 3))
    ax.legend(loc="lower right")
    save_fig(fig, figdir / "fig5_mean_g3_by_studytime_bar.png")
    facts["fig5"] = ("Mean G3 by study time: " + ", ".join(
        f"{k} = {r['mean']:.2f} (n = {int(r['count'])})" for k, r in agg.iterrows())
        + ". Group sizes differ a lot, so the highest-study group is based on few students.")

    # extra: class distribution
    counts = df["pass_fail"].map({1: "Pass", 0: "Fail"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#55A868", "#C44E52"],
                legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, f"{v} ({v / len(df):.1%})", ha="center")
    ax.set(title="Pass/Fail Class Distribution", xlabel="Outcome (G3 >= 10 is Pass)",
           ylabel="Number of students")
    save_fig(fig, figdir / "fig6_class_distribution.png")
    facts["class_counts"] = {k: int(v) for k, v in counts.items()}

    # normality check
    sw = stats.shapiro(g3)
    facts["shapiro_g3"] = {"W": float(sw.statistic), "p": float(sw.pvalue)}
    out = {}
    for col in ["sex", "school", "higher", "failures"]:
        out[col] = df.groupby(col)["G3"].mean().round(2).to_dict()
    facts["group_means"] = out
    outlier_counts(df.drop(columns=["pass_fail"])).to_csv(outdir / "outlier_counts_iqr.csv",
                                                         index=False)
    facts["pass_by_studytime"] = df.groupby("study_label")["pass_fail"].mean().reindex(order).round(3).to_dict()
    return facts


def interactive_chart(df: pd.DataFrame, path: Path) -> bool:
    """Plotly chart with hover + dropdown to change the grouping variable."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("    plotly not installed; skipping interactive chart")
        return False
    vars_ = ["studytime", "sex", "school", "failures", "higher", "internet", "romantic"]
    fig = go.Figure()
    vis_map = []
    for v in vars_:
        n0 = len(fig.data)
        for cat in sorted(df[v].unique(), key=str):
            fig.add_trace(go.Box(y=df.loc[df[v] == cat, "G3"], name=str(cat), boxpoints="all",
                                 jitter=0.4, pointpos=0, visible=(v == vars_[0])))
        vis_map.append((n0, len(fig.data)))
    total = len(fig.data)
    buttons = []
    for v, (a, b) in zip(vars_, vis_map):
        vis = [a <= i < b for i in range(total)]
        buttons.append(dict(label=v, method="update",
                            args=[{"visible": vis}, {"xaxis.title": v}]))
    fig.update_layout(title="Interactive: Final Grade (G3) by chosen variable",
                      xaxis_title=vars_[0], yaxis_title="Final grade G3",
                      updatemenus=[dict(buttons=buttons, x=0.0, y=1.15, xanchor="left")])
    fig.write_html(str(path), include_plotlyjs=True)
    return True


# ----------------------------------------------------------------------------
# Stage 6: deep learning
# ----------------------------------------------------------------------------
def build_mlp(n_in: int, h1: int, h2: int, lr: float):
    from tensorflow import keras
    m = keras.Sequential([
        keras.layers.Input(shape=(n_in,)),
        keras.layers.Dense(h1, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(h2, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def run_model(split, pre, modeldir: Path) -> dict | None:
    """Train 3 configs, select by VALIDATION loss, evaluate once on the test set."""
    try:
        from tensorflow import keras
    except ImportError:
        print("\n!! TensorFlow is not installed (needs Python 3.9-3.12).\n"
              "   Install with:  pip install tensorflow\n"
              "   Model sections of the report will be marked PENDING.\n")
        return None

    Xtr, Xva, Xte, ytr, yva, yte = split
    modeldir.mkdir(parents=True, exist_ok=True)
    Ptr = pre.fit_transform(Xtr)          # fit ONLY on training data
    Pva, Pte = pre.transform(Xva), pre.transform(Xte)
    feature_names = pre.get_feature_names_out().tolist()
    import joblib
    joblib.dump(pre, modeldir / "preprocessor.joblib")

    share_min = min(ytr.mean(), 1 - ytr.mean())
    cw = None
    if share_min < 0.35:
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
        cw = {0: float(w[0]), 1: float(w[1])}
    configs = [dict(lr=1e-3, h1=64, h2=32), dict(lr=5e-4, h1=64, h2=32),
               dict(lr=1e-3, h1=32, h2=16)]
    rows, best = [], None
    for i, cfg in enumerate(configs):
        keras.utils.set_random_seed(SEED)
        model = build_mlp(Ptr.shape[1], cfg["h1"], cfg["h2"], cfg["lr"])
        ckpt = modeldir / f"ckpt_{i}.keras"
        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                             restore_best_weights=True),
               keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True)]
        h = model.fit(Ptr, ytr, validation_data=(Pva, yva), epochs=100, batch_size=32,
                      class_weight=cw, callbacks=cbs, verbose=0)
        vloss = float(min(h.history["val_loss"]))
        rows.append({**cfg, "best_val_loss": round(vloss, 4),
                     "best_val_accuracy": round(float(max(h.history["val_accuracy"])), 4),
                     "epochs_run": len(h.history["loss"])})
        print(f"    config {cfg}: best val_loss = {vloss:.4f}")
        if best is None or vloss < best["vloss"]:
            best = dict(vloss=vloss, cfg=cfg, hist=h.history, idx=i)
    comp = pd.DataFrame(rows)
    comp.to_csv(modeldir / "hyperparameter_comparison.csv", index=False)
    model = keras.models.load_model(modeldir / f"ckpt_{best['idx']}.keras")
    model.save(modeldir / "best_model.keras")
    for i in range(len(configs)):
        (modeldir / f"ckpt_{i}.keras").unlink(missing_ok=True)
    hist = best["hist"]
    (modeldir / "history.json").write_text(json.dumps(hist), encoding="utf-8")

    # ---- training curves
    ep = range(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, hist["loss"], label="Training loss")
    ax.plot(ep, hist["val_loss"], label="Validation loss")
    ax.set(title="Training and Validation Loss per Epoch", xlabel="Epoch",
           ylabel="Binary cross-entropy loss")
    ax.legend()
    save_fig(fig, modeldir / "eval1_loss_curves.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, hist["accuracy"], label="Training accuracy")
    ax.plot(ep, hist["val_accuracy"], label="Validation accuracy")
    ax.set(title="Training and Validation Accuracy per Epoch", xlabel="Epoch", ylabel="Accuracy")
    ax.legend()
    save_fig(fig, modeldir / "eval2_accuracy_curves.png")

    # ---- single evaluation on the TEST set (selection is frozen)
    p_te = model.predict(Pte, verbose=0).ravel()
    pred = (p_te >= 0.5).astype(int)          # fixed 0.5 threshold, not tuned on test
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(yte, p_te)
    auc = float(roc_auc_score(yte, p_te))
    train_acc = float(accuracy_score(ytr, (model.predict(Ptr, verbose=0).ravel() >= 0.5)))
    val_acc = float(accuracy_score(yva, (model.predict(Pva, verbose=0).ravel() >= 0.5)))
    metrics = {
        "best_config": best["cfg"], "class_weights": cw,
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "n_features_after_encoding": int(Ptr.shape[1]),
        "train_accuracy": train_acc, "val_accuracy": val_acc,
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_precision": float(precision_score(yte, pred, zero_division=0)),
        "test_recall": float(recall_score(yte, pred, zero_division=0)),
        "test_f1": float(f1_score(yte, pred, zero_division=0)),
        "test_roc_auc": auc,
        "majority_baseline_accuracy": float(max(yte.mean(), 1 - yte.mean())),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(yte, pred, target_names=["Fail", "Pass"],
                                                       zero_division=0),
        "epochs_run": len(hist["loss"]),
        "hyperparameter_table": rows,
    }
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16},
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
    ax.set(title="Confusion Matrix (Test Set)", xlabel="Predicted class", ylabel="Actual class")
    save_fig(fig, modeldir / "eval3_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"MLP (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
    ax.set(title="ROC Curve (Test Set)", xlabel="False positive rate",
           ylabel="True positive rate")
    ax.legend(loc="lower right")
    save_fig(fig, modeldir / "eval4_roc_curve.png")

    # ---- permutation importance on VALIDATION set, grouped by original column
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(yva, model.predict(Pva, verbose=0).ravel())
    imp = []
    for col in Xva.columns:
        drops = []
        for _ in range(10):
            Xp = Xva.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - roc_auc_score(yva, model.predict(pre.transform(Xp), verbose=0).ravel()))
        imp.append((col, float(np.mean(drops)), float(np.std(drops))))
    imp_df = pd.DataFrame(imp, columns=["feature", "mean_auc_drop", "std"]).sort_values(
        "mean_auc_drop", ascending=False)
    imp_df.to_csv(modeldir / "permutation_importance.csv", index=False)
    top = imp_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_auc_drop"], xerr=top["std"], color="#8172B3",
            label="Drop in validation ROC-AUC")
    ax.set(title="Top 10 Features by Permutation Importance (Validation Set)",
           xlabel="Mean decrease in ROC-AUC when shuffled", ylabel="Feature")
    ax.legend(loc="lower right")
    save_fig(fig, modeldir / "eval5_permutation_importance.png")
    metrics["top_features"] = imp_df.head(5)["feature"].tolist()
    metrics["validation_auc_baseline"] = float(base)
    (modeldir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("    test metrics:", {k: round(v, 3) for k, v in metrics.items()
                                if k.startswith("test_")})
    return metrics


# ----------------------------------------------------------------------------
# Stage 7: Word report
# ----------------------------------------------------------------------------
def build_report(path: Path, info: dict, facts: dict, metrics: dict | None,
                 figdir: Path, modeldir: Path, interactive_ok: bool) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1)

    def para(text="", bold=False, size=None, align=None, italic=False, space=6):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if align == "c":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "j":
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(space)
        return p

    def body(text):
        return para(text, align="j")

    figno = [0]

    def figure(file: Path, caption: str, explanation: str, width=5.8):
        if not file.exists():
            body(f"[Figure PENDING: {file.name} was not generated]")
            return
        figno[0] += 1
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(file), width=Inches(width))
        para(f"Figure {figno[0]}: {caption}", italic=True, align="c")
        body(explanation)

    # ---------------- title page
    para("")
    para(COLLEGE, bold=True, size=18, align="c", space=4)
    para(DEPARTMENT, bold=True, size=14, align="c", space=40)
    para("PROJECT REPORT", bold=True, size=16, align="c", space=20)
    para("Student Performance Analysis and Prediction Using Exploratory Data Analysis, "
         "Data Visualization, Statistical Analysis, and Deep Learning",
         bold=True, size=18, align="c", space=40)
    para("Submitted by", italic=True, size=13, align="c", space=8)
    para(f"Name: {STUDENT_NAME}", bold=True, size=14, align="c", space=4)
    para(f"Roll Number: {ROLL_NO}", bold=True, size=14, align="c", space=4)
    para(f"Department: {DEPARTMENT}", size=13, align="c", space=4)
    para(f"Academic Year: {ACADEMIC_YEAR}", size=13, align="c", space=4)
    para(f"Date of Submission: {SUBMISSION_DATE}", size=13, align="c", space=4)
    doc.add_page_break()

    # footer page numbers
    fp = doc.sections[0].footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)

    PEND = "PENDING (model has not been trained; run the script with TensorFlow installed)."
    m = metrics
    cc = facts["class_counts"]
    n = info["rows"]

    # ---------------- 1 abstract
    doc.add_heading("Abstract", 1)
    if m:
        res = (f"The final multilayer perceptron achieved a test accuracy of {m['test_accuracy']:.1%} "
               f"and a ROC-AUC of {m['test_roc_auc']:.3f} on {m['n_test']} unseen students, compared "
               f"with {m['majority_baseline_accuracy']:.1%} for always predicting the majority class.")
    else:
        res = "Model results are pending."
    body(f"This project studies the UCI Student Performance dataset, using the mathematics file with "
         f"{n} students and {info['cols']} attributes. Exploratory data analysis, visualization and "
         f"statistical tests were used to examine how study time, absences and background factors relate "
         f"to the final grade G3. A pass/fail target was defined as G3 of at least 10, and the grades G1, "
         f"G2 and G3 were removed from the inputs so the task is an early prediction. A neural network "
         f"with two hidden layers and dropout was trained with stratified train, validation and test "
         f"splits. {res} The findings are associations in one school setting and are not causal.")

    # ---------------- 2 introduction
    doc.add_heading("1. Introduction and Objectives", 1)
    body("Identifying students who are likely to struggle early allows teachers to offer support. "
         "Schools hold data on family background, study habits and attendance, but this data needs to "
         "be explored and modelled carefully before it is useful.")
    body("Objectives:")
    for t in ["Understand the structure and quality of the student performance data.",
              "Discover patterns between study-related factors and final grades using EDA and statistics.",
              "Present the findings with clear, labelled visualizations following the seven-stage "
              "visualization pipeline.",
              "Build a deep learning (MLP) model to predict pass or fail and evaluate it honestly on "
              "unseen data."]:
        doc.add_paragraph(t, style="List Bullet")

    # ---------------- 3 dataset
    doc.add_heading("2. Dataset Description", 1)
    body("Source: UCI Machine Learning Repository, Student Performance dataset "
         "(https://archive.ics.uci.edu/dataset/320/student+performance), collected from two "
         "Portuguese secondary schools by Cortez and Silva (2008). The mathematics file "
         "student-mat.csv (semicolon separated) is used. It is released under the CC BY 4.0 licence.")
    body(f"The file has {n} rows and {info['cols']} columns ({info['numeric_cols']} numeric and "
         f"{info['categorical_cols']} categorical). There are {info['missing_total']} missing values, "
         f"{info['duplicates']} exact duplicate rows and {info['grades_out_of_range']} grade values "
         "outside the valid 0 to 20 range.")
    body("Attributes cover school and demographics (school, sex, age, address, family size), family "
         "(parents' education and jobs, family relationship quality), study habits (study time, "
         "failures, extra support, tuition classes), lifestyle (going out, alcohol use, free time, "
         "health, relationship status, internet access) and attendance (absences). G1, G2 and G3 are "
         "the period 1, period 2 and final grades.")
    body(f"Target definition: Pass = G3 of 10 or above, Fail = G3 below 10. This is the project's "
         f"chosen definition of passing. Class counts: Pass = {cc.get('Pass', 0)}, "
         f"Fail = {cc.get('Fail', 0)}.")

    # ---------------- 4 EDA & preprocessing
    doc.add_heading("3. Exploratory Data Analysis and Preprocessing", 1)
    body("Preprocessing decisions:")
    for t in [f"Missing values: {info['missing_total']} found, so no imputation was needed.",
              f"Duplicates: {info['duplicates']} exact duplicates found. The dataset has no student ID, "
              "so duplicate detection is only approximate.",
              "Outliers: checked with boxplots and the IQR rule (counts saved in outputs/eda). "
              "Observations were kept because high absences and low grades are real students.",
              "Leakage control: G1, G2 and G3 are excluded from the model inputs. G1 and G2 are earlier "
              "grades of the same course and strongly predict G3, so using them would turn the task "
              "into copying a grade. G3 is used only to create the target. An assertion in the code "
              "enforces this.",
              "Encoding and scaling: one-hot encoding for categorical columns and StandardScaler for "
              "numeric columns, fitted on the training set only to prevent data leakage.",
              "Split: stratified 70% training, 15% validation and 15% test with a fixed seed (42)."]:
        doc.add_paragraph(t, style="List Bullet")
    gm = facts["group_means"]
    body(f"Statistical summary: the Shapiro-Wilk test on G3 gave W = {facts['shapiro_g3']['W']:.3f}, "
         f"p = {facts['shapiro_g3']['p']:.4g}, so G3 is not treated as normally distributed and "
         f"rank-based methods (Spearman, Kruskal-Wallis) are used. Mean G3 by school: {gm['school']}; "
         f"by sex: {gm['sex']}; by past failures: {gm['failures']}. These are descriptive differences "
         "and do not show causes. Many comparisons were examined, which can produce false positives.")

    # ---------------- 5 figures
    doc.add_heading("4. EDA Visualizations", 1)
    spec = [
        ("fig1_g3_histogram_kde.png", "Histogram with KDE of final grade G3", "fig1",
         "Shows the shape of the grade distribution and the pass mark. Created to see whether grades "
         "are symmetric and where the pass/fail boundary falls."),
        ("fig2_g3_by_studytime_boxplot.png", "Boxplot of G3 by weekly study time", "fig2",
         "Compares the spread and median grade of each study-time group."),
        ("fig3_correlation_heatmap.png", "Correlation heatmap of numeric features", "fig3",
         "Shows linear relationships between all numeric variables."),
        ("fig4_absences_vs_g3_scatter.png", "Scatterplot of absences versus G3", "fig4",
         "Shows how attendance relates to the final grade, with a fitted trend line."),
        ("fig5_mean_g3_by_studytime_bar.png", "Mean G3 by study-time category", "fig5",
         "Compares average grades across study-time groups, with standard-error bars and group sizes."),
        ("fig6_class_distribution.png", "Pass/fail class distribution", None,
         f"Shows class balance: Pass = {cc.get('Pass', 0)}, Fail = {cc.get('Fail', 0)}."),
    ]
    for fname, cap, key, intro in spec:
        result = facts.get(key, "") if key else ""
        figure(figdir / fname, cap, f"What it shows and why: {intro} Actual result: {result} "
               "Limitation: the data are observational, from self-reported variables and one "
               "educational setting, so patterns should not be read as causes.")
    if interactive_ok:
        body("An interactive Plotly chart with hover information and a dropdown to change the "
             "grouping variable was also created and saved as outputs/figures/interactive_g3.html.")

    # ---------------- 6 seven stages
    doc.add_heading("5. The Seven Stages of Visualization", 1)
    stages = [
        ("Acquire", "The dataset was downloaded programmatically from the UCI repository."),
        ("Parse", f"The semicolon-delimited file was loaded into a DataFrame; {info['cols']} fields "
                  "were classified as numeric or categorical and data types were checked."),
        ("Filter", "Missing values, duplicates and invalid grades were checked; outliers were flagged "
                   "with the IQR rule but retained; G1, G2 and G3 were removed from model inputs."),
        ("Mine", "Descriptive statistics, Pearson and Spearman correlation, a Shapiro-Wilk test and a "
                 "Kruskal-Wallis test were computed."),
        ("Represent", "Histogram, boxplot, heatmap, scatterplot and bar chart were chosen to match "
                      "the type of each variable."),
        ("Refine", "Titles, axis labels, legends, readable study-time labels, consistent fonts and "
                   "200 dpi export were applied."),
        ("Interact", "An interactive Plotly boxplot with hover and a dropdown was created."
         if interactive_ok else "Interactive chart PENDING (Plotly not installed)."),
    ]
    for i, (s, t) in enumerate(stages, 1):
        p = doc.add_paragraph(style="List Number")
        p.add_run(f"{s}: ").bold = True
        p.add_run(t)

    # ---------------- 7 model
    doc.add_heading("6. Deep Learning Architecture and Hyperparameters", 1)
    body("A Multilayer Perceptron (MLP) was built with TensorFlow/Keras: Input, Dense(h1, ReLU), "
         "Dropout(0.3), Dense(h2, ReLU), Dropout(0.2), Dense(1, sigmoid). Loss: binary cross-entropy. "
         "Optimizer: Adam. Batch size 32, at most 100 epochs, EarlyStopping on validation loss "
         "(patience 15, best weights restored) and ModelCheckpoint saving the best model. "
         "Random seed 42.")
    if m:
        body(f"Three settings were compared using validation loss only. Selected setting: "
             f"learning rate {m['best_config']['lr']}, hidden layers {m['best_config']['h1']} and "
             f"{m['best_config']['h2']}. Input size after encoding: {m['n_features_after_encoding']} "
             f"features. Training/validation/test sizes: {m['n_train']}/{m['n_val']}/{m['n_test']}. "
             f"Class weights: {'not used (classes not strongly imbalanced)' if not m['class_weights'] else m['class_weights']}.")
        t = doc.add_table(rows=1, cols=5)
        t.style = "Table Grid"
        for c, h in zip(t.rows[0].cells, ["Learning rate", "Hidden 1", "Hidden 2",
                                          "Best val loss", "Epochs run"]):
            c.text = h
        for r in m["hyperparameter_table"]:
            cells = t.add_row().cells
            for c, v in zip(cells, [r["lr"], r["h1"], r["h2"], r["best_val_loss"], r["epochs_run"]]):
                c.text = str(v)
        para("")
    else:
        body(PEND)

    # ---------------- 8 evaluation
    doc.add_heading("7. Model Evaluation", 1)
    if m:
        cm = m["confusion_matrix"]
        body(f"The test set was used once, after model selection. Test accuracy "
             f"{m['test_accuracy']:.3f}, precision {m['test_precision']:.3f}, recall "
             f"{m['test_recall']:.3f}, F1-score {m['test_f1']:.3f}, ROC-AUC {m['test_roc_auc']:.3f}. "
             f"Always predicting the majority class would give accuracy "
             f"{m['majority_baseline_accuracy']:.3f}. The classification threshold was fixed at 0.5 "
             "and not tuned on the test data.")
        t = doc.add_table(rows=3, cols=3)
        t.style = "Table Grid"
        for i, v in enumerate(["Actual \\ Predicted", "Fail", "Pass"]):
            t.rows[0].cells[i].text = v
        t.rows[1].cells[0].text, t.rows[2].cells[0].text = "Fail", "Pass"
        for i in range(2):
            for j in range(2):
                t.rows[i + 1].cells[j + 1].text = str(cm[i][j])
        para("")
        body("Classification report:\n" + m["classification_report"])
        gap = m["train_accuracy"] - m["val_accuracy"]
        fp, fn = cm[0][1], cm[1][0]
        figure(modeldir / "eval1_loss_curves.png", "Training and validation loss",
               f"Shows whether the model learns and whether it overfits. Training stopped after "
               f"{m['epochs_run']} epochs. Training accuracy is {m['train_accuracy']:.3f} and "
               f"validation accuracy is {m['val_accuracy']:.3f}, a gap of {gap:+.3f}; a large positive "
               "gap would indicate overfitting.")
        figure(modeldir / "eval2_accuracy_curves.png", "Training and validation accuracy",
               "Shows how accuracy changes during training; validation accuracy is the signal used "
               "to judge generalization.")
        figure(modeldir / "eval3_confusion_matrix.png", "Confusion matrix on the test set",
               f"Out of {m['n_test']} test students there were {fp} false positives (failing students "
               f"predicted to pass) and {fn} false negatives (passing students predicted to fail). "
               "Missing a failing student is usually the costlier error in an early-support setting.")
        figure(modeldir / "eval4_roc_curve.png", "ROC curve on the test set",
               f"The test ROC-AUC is {m['test_roc_auc']:.3f}; 0.5 would be random guessing and 1.0 "
               "perfect separation. With only about "
               f"{m['n_test']} test students this estimate is noisy.", width=5.0)
        figure(modeldir / "eval5_permutation_importance.png",
               "Permutation feature importance (validation set)",
               f"The features whose shuffling reduces validation ROC-AUC the most are "
               f"{', '.join(m['top_features'])}. Correlated features share importance, and this "
               "measure describes the model, not causes in real students.")
    else:
        body(PEND)

    # ---------------- 9 conclusion
    doc.add_heading("8. Conclusion and Future Scope", 1)
    sp = facts["spearman_absences"]
    body(f"EDA showed that G3 is spread across the 0 to 20 scale with a "
         f"cluster of zero scores, and that {cc.get('Pass', 0)} of {n} students passed. "
         f"Absences had a Spearman correlation of {sp['rho']:.2f} with G3. The Kruskal-Wallis test "
         f"across study-time groups gave p = {facts['kruskal']['p']:.4f}. "
         + (f"The MLP reached a test ROC-AUC of {m['test_roc_auc']:.3f} and accuracy of "
            f"{m['test_accuracy']:.1%} without using earlier grades." if m else
            "Model conclusions are PENDING."))
    body("Limitations: the data are observational, so all relationships are correlations and not "
         "causal effects. The data come from two Portuguese schools and may not generalize elsewhere. "
         "The test set is small, so metrics are uncertain. Several attributes are self-reported, and "
         "sensitive attributes such as sex and age are present, so a fairness analysis is needed "
         "before any real use.")
    body("Predictions from this model must not be used to make consequential decisions about "
         "students without further validation, fairness analysis and human oversight.")
    body("Future scope: k-fold cross-validation, comparison with other models (logistic regression, "
         "random forest, gradient boosting), more schools and years of data, additional features "
         "such as attendance over time, and calibration and fairness analysis.")

    # ---------------- references
    doc.add_heading("References", 1)
    refs = [
        "Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... Zheng, X. (2016). "
        "TensorFlow: A system for large-scale machine learning. Proceedings of the 12th USENIX "
        "Symposium on Operating Systems Design and Implementation (OSDI '16), 265-283.",
        "Cortez, P. (2014). Student performance [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5TG7T",
        "Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student "
        "performance. In A. Brito & J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness "
        "TEChnology Conference (FUBUTEC 2008) (pp. 5-12). EUROSIS.",
        "Fry, B. (2007). Visualizing data: Exploring and explaining data with the Processing "
        "environment. O'Reilly Media.",
        "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & "
        "Engineering, 9(3), 90-95.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... "
        "Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine "
        "Learning Research, 12, 2825-2830.",
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)

    # ---------------- appendix
    doc.add_heading("Appendix", 1)
    body(f"Complete source code: {GITHUB_LINK}. The whole project is reproduced by running "
         "`python make_report.py`, which downloads the data, creates all figures, trains the model "
         "and regenerates this report. Key steps in code:")
    code = ("df = pd.read_csv('student-mat.csv', sep=';')\n"
            "y = (df['G3'] >= 10).astype(int)\n"
            "X = df.drop(columns=['G1', 'G2', 'G3'])\n"
            "# stratified 70/15/15 split, preprocessor fitted on training data only\n"
            "model = Sequential([Dense(64, 'relu'), Dropout(0.3), Dense(32, 'relu'),\n"
            "                    Dropout(0.2), Dense(1, 'sigmoid')])\n"
            "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])")
    p = doc.add_paragraph()
    r = p.add_run(code)
    r.font.name = "Courier New"
    r.font.size = Pt(9)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))



def export_pdf(docx_path: Path) -> Path | None:
    """Convert the .docx to PDF (Microsoft Word via docx2pdf, else LibreOffice)."""
    import shutil
    import subprocess
    pdf = docx_path.with_suffix(".pdf")
    try:
        from docx2pdf import convert  # works on Windows/macOS with Word installed
        convert(str(docx_path), str(pdf))
        if pdf.exists():
            return pdf
    except Exception:  # noqa: BLE001
        pass
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if exe:
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir",
                        str(docx_path.parent), str(docx_path)], check=False,
                       capture_output=True, timeout=180)
        if pdf.exists():
            return pdf
    return None

# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "student-mat.csv"))
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args()
    out = Path(a.out)
    figdir, edadir, modeldir = out / "outputs" / "figures", out / "outputs" / "eda", out / "outputs" / "model"

    df = load_data(Path(a.data))
    info = data_summary(df, edadir)
    print("[3] Running EDA and saving figures ...")
    facts = run_eda(df, figdir, edadir)
    ok = interactive_chart(df, figdir / "interactive_g3.html")

    print("[4] Preprocessing and splitting ...")
    df_clean = df.drop_duplicates()
    X, y = build_features(df_clean)
    split = split_data(X, y)
    pre = build_preprocessor(X)

    print("[5] Training the MLP ...")
    metrics = run_model(split, pre, modeldir)

    print("[6] Writing the Word report ...")
    report = out / "report" / "Project_Report.docx"
    build_report(report, info, facts, metrics, figdir, modeldir, ok)
    print(f"\nDone. Word report saved to: {report}")
    pdf = export_pdf(report)
    if pdf:
        print(f"PDF report saved to: {pdf}")
    else:
        print("Could not auto-create the PDF. Open the .docx in Word and use "
              "File > Save As > PDF.")


if __name__ == "__main__":
    main()

"""Descriptive analysis, statistical checks, and visualizations."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import kruskal, skew

from src.config import DATA_PATH, FIGURES_DIR, OUTPUTS_DIR, PASS_THRESHOLD, ensure_project_dirs

STUDY_LABELS = {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}


def _save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUTPUTS_DIR / filename, encoding="utf-8")


def run_eda(frame: pd.DataFrame | None = None) -> dict:
    """Write dataset summaries, required plots, interpretations, and HTML chart."""
    ensure_project_dirs()
    if frame is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run `python -m src.download_data` first.")
        frame = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
    data = frame.copy()
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    print("First five rows:")
    print(data.head().to_string(index=False))
    print("\nLast five rows:")
    print(data.tail().to_string(index=False))
    print(f"\nShape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Dtypes:\n{data.dtypes.to_string()}")
    print(f"Numeric columns: {numeric_columns}")
    print(f"Categorical columns: {categorical_columns}")
    print(f"Descriptive statistics:\n{data.describe(include='all').to_string()}")
    print(f"Missing values per column:\n{data.isna().sum().to_string()}")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")
    print(f"Invalid G3 values outside 0-20: {int((~data['G3'].between(0, 20)).sum())}")
    print(
        "Pass/fail counts (pass means G3 >= "
        f"{PASS_THRESHOLD}):\n{data['G3'].ge(PASS_THRESHOLD).map({True: 'pass', False: 'fail'}).value_counts().to_string()}"
    )

    (OUTPUTS_DIR / "first_5_rows.csv").write_text(data.head().to_csv(index=False), encoding="utf-8")
    (OUTPUTS_DIR / "last_5_rows.csv").write_text(data.tail().to_csv(index=False), encoding="utf-8")
    data.dtypes.astype(str).rename("dtype").to_csv(OUTPUTS_DIR / "column_dtypes.csv", encoding="utf-8")
    data.describe(include="all").transpose().to_csv(OUTPUTS_DIR / "describe.csv", encoding="utf-8")
    data.isna().sum().rename("missing_count").to_csv(OUTPUTS_DIR / "missing_values.csv", encoding="utf-8")
    invalid_grades = data.loc[~data["G3"].between(0, 20), ["G3"]]
    pd.DataFrame(
        {
            "shape_rows": [data.shape[0]],
            "shape_columns": [data.shape[1]],
            "duplicate_count": [int(data.duplicated().sum())],
            "invalid_G3_count": [len(invalid_grades)],
            "numeric_columns": [", ".join(numeric_columns)],
            "categorical_columns": [", ".join(categorical_columns)],
        }
    ).to_csv(OUTPUTS_DIR / "dataset_overview.csv", index=False, encoding="utf-8")
    data.loc[data["G3"].between(0, 20), "G3"].ge(PASS_THRESHOLD).map({True: "pass", False: "fail"}).value_counts().rename_axis(
        "class"
    ).rename("count").to_csv(OUTPUTS_DIR / "pass_fail_counts.csv", encoding="utf-8")

    stats = data[numeric_columns].agg(["mean", "median", "std", "min", "max"]).transpose()
    stats["q1"] = data[numeric_columns].quantile(0.25)
    stats["q3"] = data[numeric_columns].quantile(0.75)
    stats["skewness"] = data[numeric_columns].apply(lambda column: skew(column, bias=False))
    _save_csv(stats, "numeric_statistics.csv")
    if categorical_columns:
        counts = pd.concat(
            {column: data[column].value_counts(dropna=False) for column in categorical_columns},
            names=["column", "category"],
        ).rename("count").to_frame()
        counts.to_csv(OUTPUTS_DIR / "categorical_counts.csv", encoding="utf-8")

    correlations = pd.DataFrame(
        {
            "pearson": data[numeric_columns].corr(method="pearson")["G3"],
            "spearman": data[numeric_columns].corr(method="spearman")["G3"],
        }
    )
    correlations.to_csv(OUTPUTS_DIR / "grade_correlations.csv", encoding="utf-8")
    study_groups = [group["G3"].to_numpy() for _, group in data.groupby("studytime", observed=True) if len(group)]
    statistic, p_value = kruskal(*study_groups)
    statistical_test = {
        "test": "Kruskal-Wallis",
        "purpose": "Check whether the G3 distributions differ across study-time groups.",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "interpretation": "This test assesses an association between group distributions; it does not establish causality.",
    }
    (OUTPUTS_DIR / "statistical_test.json").write_text(json.dumps(statistical_test, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid", font_scale=1.0)
    figures: dict[str, dict] = {}
    grade_mean, grade_median = float(data["G3"].mean()), float(data["G3"].median())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="G3", kde=True, discrete=True, ax=ax)
    ax.set(title="Final Mathematics Grade (G3)", xlabel="Final grade (0-20)", ylabel="Student count")
    fig.tight_layout()
    path = FIGURES_DIR / "g3_histogram_kde.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures[path.name] = {
        "what": "Distribution of final grades with a smoothed KDE curve.",
        "why": "Shows the grade range, concentration, and shape before classification.",
        "actual_result": f"Mean G3 was {grade_mean:.2f}; median G3 was {grade_median:.2f}.",
        "limitation": "A univariate distribution does not explain why grades differ.",
    }

    plot_data = data.assign(studytime_label=data["studytime"].map(STUDY_LABELS))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_data, x="studytime_label", y="G3", order=list(STUDY_LABELS.values()), ax=ax)
    ax.set(title="Final Grade by Weekly Study-Time Group", xlabel="Study time per week", ylabel="Final grade (0-20)")
    fig.tight_layout()
    path = FIGURES_DIR / "g3_by_studytime_boxplot.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    means = data.groupby("studytime", observed=True)["G3"].mean()
    figures[path.name] = {
        "what": "Grade distributions, medians, and potential outliers across study-time categories.",
        "why": "Compares the spread of grades among reported study-time groups.",
        "actual_result": f"Mean G3 by group was {', '.join(f'{STUDY_LABELS.get(int(k), k)}: {v:.2f}' for k, v in means.items())}.",
        "limitation": "Study time is self-reported and group differences are observational.",
    }

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(data[numeric_columns].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation Heatmap of Numeric Features")
    fig.tight_layout()
    path = FIGURES_DIR / "numeric_correlation_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    strongest = correlations.drop(index="G3", errors="ignore")["pearson"].abs().idxmax()
    figures[path.name] = {
        "what": "Pairwise Pearson correlations among numeric variables.",
        "why": "Highlights linear relationships that can guide further exploration.",
        "actual_result": f"The numeric feature with the largest absolute Pearson correlation with G3 was {strongest} ({correlations.loc[strongest, 'pearson']:.3f}).",
        "limitation": "Correlation is not causation and Pearson correlation can miss nonlinear relationships.",
    }

    absences_corr = float(data["absences"].corr(data["G3"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=data, x="absences", y="G3", scatter_kws={"alpha": 0.55}, line_kws={"color": "red"}, ax=ax)
    ax.set(title="Absences and Final Grade", xlabel="Number of absences", ylabel="Final grade (0-20)")
    fig.tight_layout()
    path = FIGURES_DIR / "absences_vs_g3.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures[path.name] = {
        "what": "Student absences plotted against final grade with a fitted linear trend line.",
        "why": "Explores a possible linear association between absence counts and final grade.",
        "actual_result": f"Pearson correlation between absences and G3 was {absences_corr:.3f}.",
        "limitation": "The trend line is descriptive; confounders and overlapping observations are not controlled.",
    }

    means = means.reindex(list(STUDY_LABELS.keys()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([STUDY_LABELS[key] for key in means.index], means.values, color=sns.color_palette("deep", len(means)))
    ax.set(title="Mean Final Grade by Study-Time Group", xlabel="Study time per week", ylabel="Mean final grade (0-20)")
    fig.tight_layout()
    path = FIGURES_DIR / "mean_g3_by_studytime.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures[path.name] = {
        "what": "Mean final grade in each of the four study-time categories.",
        "why": "Provides a compact comparison of group averages.",
        "actual_result": f"Highest group mean was {STUDY_LABELS[int(means.idxmax())]} ({means.max():.2f}); lowest was {STUDY_LABELS[int(means.idxmin())]} ({means.min():.2f}).",
        "limitation": "Means hide within-group variation and do not imply that study time caused the differences.",
    }

    class_counts = data["G3"].ge(PASS_THRESHOLD).map({True: "Pass", False: "Fail"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(class_counts.index, class_counts.values, color=["#4c72b0", "#dd8452"])
    ax.set(title=f"Pass/Fail Classes (pass: G3 >= {PASS_THRESHOLD})", xlabel="Class", ylabel="Student count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pass_fail_distribution.png", dpi=200)
    plt.close(fig)

    iqr_rows = []
    for column in numeric_columns:
        q1, q3 = data[column].quantile([0.25, 0.75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        count = int(((data[column] < low) | (data[column] > high)).sum())
        iqr_rows.append({"feature": column, "q1": q1, "q3": q3, "lower_bound": low, "upper_bound": high, "outlier_count": count})
    pd.DataFrame(iqr_rows).to_csv(OUTPUTS_DIR / "iqr_outlier_counts.csv", index=False, encoding="utf-8")

    (FIGURES_DIR / "interpretations.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    interactive_data = data.copy()
    interactive_data["studytime"] = interactive_data["studytime"].map(STUDY_LABELS)
    interactive = px.box(
        interactive_data,
        x="studytime",
        y="G3",
        hover_data=["age", "absences", "studytime"],
        title="Interactive Final Grade Explorer",
        labels={"studytime": "Study-time category", "G3": "Final grade (0-20)"},
    )
    interactive.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": column,
                        "method": "update",
                        "args": [
                            {
                                "x": [
                                    (
                                        data[column].map(STUDY_LABELS).tolist()
                                        if column == "studytime"
                                        else data[column].tolist()
                                    )
                                ]
                            },
                            {"xaxis": {"title": column}},
                        ],
                    }
                    for column in ("studytime", "sex", "age", "schoolsup", "famsup", "activities")
                ],
                "direction": "down",
                "x": 1.02,
                "y": 1,
            }
        ]
    )
    interactive.write_html(FIGURES_DIR / "interactive_g3.html", include_plotlyjs=True)
    result = {"figures": figures, "kruskal_p_value": float(p_value), "class_counts": class_counts.to_dict()}
    print(f"EDA complete. Kruskal-Wallis p-value: {p_value:.6g}; G3 mean: {grade_mean:.2f}.")
    return result


if __name__ == "__main__":
    run_eda()

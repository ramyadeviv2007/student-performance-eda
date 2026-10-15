"""Create an editable Word report and synchronize README metrics."""

from __future__ import annotations

import json
import re

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from src.config import FIGURES_DIR, MODEL_DIR, OUTPUTS_DIR, REPORT_DIR, ensure_project_dirs


def _add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run._r.addnext(field)


def _add_picture_with_caption(document: Document, path, caption: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required report chart is missing: {path}")
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(5.8))
    caption_paragraph = document.add_paragraph(caption, style="Caption")
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _format_number(value: float) -> str:
    return f"{value:.3f}"


def _sync_readme(metrics: dict) -> None:
    readme_path = REPORT_DIR.parent / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    block = (
        f"Test set: **{metrics['test_samples']} students**  \n"
        f"Accuracy: **{_format_number(metrics['accuracy'])}**; "
        f"precision: **{_format_number(metrics['precision'])}**; "
        f"recall: **{_format_number(metrics['recall'])}**; "
        f"F1: **{_format_number(metrics['f1'])}**; "
        f"ROC-AUC: **{_format_number(metrics['roc_auc'])}**  \n"
        f"Confusion matrix (actual fail/pass by predicted fail/pass): "
        f"`{metrics['confusion_matrix']}`"
    )
    text = re.sub(r"<!-- METRICS:START -->.*?<!-- METRICS:END -->", f"<!-- METRICS:START -->\n{block}\n<!-- METRICS:END -->", text, flags=re.S)
    readme_path.write_text(text, encoding="utf-8")


def generate_report() -> None:
    """Build the project report from generated summaries, plots, and metrics."""
    ensure_project_dirs()
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Test metrics are missing. Run `python -m src.evaluate_model` first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    interpretation = json.loads((FIGURES_DIR / "interpretations.json").read_text(encoding="utf-8"))
    overview = pd.read_csv(OUTPUTS_DIR / "dataset_overview.csv").iloc[0]
    stats = pd.read_csv(OUTPUTS_DIR / "numeric_statistics.csv", index_col=0)
    test_result = json.loads((OUTPUTS_DIR / "statistical_test.json").read_text(encoding="utf-8"))
    split = pd.read_csv(OUTPUTS_DIR / "split_summary.csv")
    training = json.loads((MODEL_DIR / "training_history.json").read_text(encoding="utf-8"))
    class_counts = pd.read_csv(OUTPUTS_DIR / "pass_fail_counts.csv").set_index("class")["count"].to_dict()
    missing_values = pd.read_csv(OUTPUTS_DIR / "missing_values.csv", index_col=0)
    history = training["selected_history"]
    importance = pd.read_csv(MODEL_DIR / "permutation_importance.csv")

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    _add_page_number(section)
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(10)

    for _ in range(3):
        document.add_paragraph()
    title = document.add_heading("Student Performance Analysis and Prediction", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph("Using EDA, Data Visualization, and Deep Learning")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph()
    for placeholder in ("[Student Name]", "[Roll Number]", "[Department]", "[Academic Year]", "[Submission Date]"):
        p = document.add_paragraph(placeholder)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_page_break()

    document.add_heading("Abstract", level=1)
    abstract = (
        f"This project analyzes the UCI Student Performance mathematics dataset, containing "
        f"{int(overview['shape_rows'])} students and {int(overview['shape_columns'])} variables. "
        f"Exploratory analysis summarizes grades, attendance, and reported study-time groups; "
        f"a Kruskal-Wallis test compares final-grade distributions across study-time categories "
        f"(p={test_result['p_value']:.4g}), without implying causation. Passing is defined for this "
        f"project as G3 >= 10. To avoid direct use of same-course grades, G1, G2, and G3 are excluded "
        f"from model inputs. A training-only fitted one-hot/scaling transformer prepares a stratified "
        f"train/validation/test split, and a small Keras multilayer perceptron is selected by validation "
        f"loss. On the held-out test sample of {metrics['test_samples']} students, accuracy was "
        f"{metrics['accuracy']:.3f}, F1 was {metrics['f1']:.3f}, and ROC-AUC was {metrics['roc_auc']:.3f}. "
        f"The small test set and single-school observational data make the results noisy and limit "
        f"generalization; these predictions are educationally exploratory, not decision tools."
    )
    document.add_paragraph(abstract)
    if not 100 <= len(abstract.split()) <= 150:
        raise ValueError(f"Generated abstract is {len(abstract.split())} words; required range is 100-150.")

    document.add_heading("1. Introduction and Objectives", level=1)
    document.add_paragraph(
        "The purpose is to demonstrate a complete, reproducible workflow for exploring student "
        "performance and evaluating a simple binary classifier. Objectives are to inspect the data, "
        "visualize grade relationships, prepare leakage-conscious features, train and select an MLP "
        "using validation loss, and report held-out test results with appropriate limitations."
    )
    document.add_heading("2. Dataset Description", level=1)
    document.add_paragraph(
        "The mathematics file student-mat.csv is from the UCI Student Performance dataset. It has "
        f"{int(overview['shape_rows'])} rows and {int(overview['shape_columns'])} columns; the CSV uses "
        "a semicolon delimiter. The dataset is distributed under CC BY 4.0. The target is pass_fail=1 "
        "when final grade G3 is at least 10 and 0 otherwise; this is the project's chosen definition "
        "of passing, not a universal policy."
    )
    document.add_paragraph(
        f"Source inspection found {int(overview['duplicate_count'])} exact duplicates, "
        f"{int(overview['invalid_G3_count'])} grades outside 0-20, and "
        f"{int(missing_values.sum().iloc[0])} total missing values. "
        f"Pass/fail class counts at the chosen threshold were pass={int(class_counts['pass'])} "
        f"and fail={int(class_counts['fail'])}."
    )
    document.add_heading("3. EDA and Preprocessing", level=1)
    document.add_paragraph(
        f"The mean and median G3 were {stats.loc['G3', 'mean']:.2f} and {stats.loc['G3', 'median']:.2f}. "
        f"The Kruskal-Wallis test across study-time groups returned statistic {test_result['statistic']:.3f} "
        f"and p-value {test_result['p_value']:.4g}. This is a group-distribution test and does not "
        "support a causal conclusion. Potential outliers were identified by the 1.5 x IQR rule and "
        "retained because an unusual observation is not necessarily an error."
    )
    document.add_paragraph(
        "G1, G2, and G3 are excluded from model predictors because they are grades from the same "
        "course; including them would leak direct course-grade information and make it possible for "
        "the model to copy the answer. Exact duplicates are removed, and incomplete rows are removed "
        "only if missing values occur. A single ColumnTransformer applies one-hot encoding to "
        "categorical inputs and StandardScaler to numeric inputs. It is fitted only on training data."
    )
    document.add_paragraph(
        "Stratified partitions: " + "; ".join(
            f"{row.split}: {row.rows} rows" for row in split.itertuples(index=False)
        ) + "."
    )
    document.add_heading("4. Exploratory Figures", level=1)
    descriptions = [
        ("g3_histogram_kde.png", "Final grade distribution", "Histogram and KDE show the observed shape; mean and median are given above. A distribution alone cannot explain causes."),
        ("g3_by_studytime_boxplot.png", "Grades by study time", "Boxplots compare medians and spread by reported study-time group; self-report and confounding limit interpretation."),
        ("numeric_correlation_heatmap.png", "Numeric feature correlation", "Pairwise Pearson correlations summarize linear association, not causation or nonlinear dependence."),
        ("absences_vs_g3.png", "Absences and final grade", "Scatterplot and fitted line show the observed relationship; the line is descriptive and omits other variables."),
        ("mean_g3_by_studytime.png", "Mean grades by study time", "Group averages are directly compared; means conceal within-group variation and are not causal effects."),
    ]
    for index, (filename, caption, limitation) in enumerate(descriptions, start=1):
        _add_picture_with_caption(document, FIGURES_DIR / filename, f"Figure {index}. {caption}.")
        document.add_paragraph(interpretation[filename]["actual_result"] + " " + limitation)
    _add_picture_with_caption(document, FIGURES_DIR / "pass_fail_distribution.png", "Supplementary figure. Pass/fail class counts.")

    document.add_heading("5. Seven Visualization Stages", level=1)
    stages = [
        ("Acquire", "Downloaded the source CSV from the UCI archive and validated its expected dimensions."),
        ("Parse", "Read the semicolon-delimited file into pandas and classified columns by data type."),
        ("Filter", "Inspected missing values, duplicates, grade bounds, and IQR outlier counts; retained legitimate observations."),
        ("Mine", "Computed summary statistics, correlations, class counts, and the Kruskal-Wallis group comparison."),
        ("Represent", "Created the five required EDA plots and the pass/fail distribution."),
        ("Refine", "Added labeled axes, readable chart sizes, descriptive titles, and saved PNG images at 200 dpi."),
        ("Interact", "Saved a Plotly HTML chart with hover information and a grouping-variable dropdown."),
    ]
    for name, details in stages:
        document.add_heading(name, level=2)
        document.add_paragraph(details)

    document.add_heading("6. Model Architecture and Selection", level=1)
    document.add_paragraph(
        "The binary Keras MLP uses Dense(64, ReLU), Dropout(0.3), Dense(32, ReLU), "
        "Dropout(0.2), and Dense(1, sigmoid). It is trained with binary cross-entropy, Adam, "
        "accuracy, batch size 32, and a maximum of 100 epochs. EarlyStopping monitors validation "
        "loss and restores the best weights. Three small configurations are compared using validation "
        f"loss only; selected setting: {training['selected_setting']}; best validation loss: "
        f"{training['best_validation_loss']:.4f} at epoch {training['best_epoch']}. The test set is "
        "not used for model selection."
    )
    document.add_heading("7. Evaluation and Interpretation", level=1)
    document.add_paragraph(
        f"On {metrics['test_samples']} held-out students, accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}, and ROC-AUC={metrics['roc_auc']:.3f}. "
        f"Confusion matrix (actual fail/pass by predicted fail/pass): {metrics['confusion_matrix']}."
    )
    document.add_paragraph(
        "The test set is only about 60 students, so all test estimates are noisy. Loss and accuracy "
        f"curves ended at training/validation loss {_format_number(history['loss'][-1])}/"
        f"{_format_number(history['val_loss'][-1])} and training/validation accuracy "
        f"{_format_number(history['accuracy'][-1])}/{_format_number(history['val_accuracy'][-1])}. "
        "A gap may indicate a generalization difference, but the small validation sample makes this "
        "evidence uncertain. The confusion matrix contains "
        f"{metrics['confusion_matrix'][0][0]} true fails, {metrics['confusion_matrix'][0][1]} false passes, "
        f"{metrics['confusion_matrix'][1][0]} false fails, and {metrics['confusion_matrix'][1][1]} true passes. "
        f"The ROC curve's held-out AUC is {metrics['roc_auc']:.3f}. Validation permutation importance "
        f"ranks {importance.iloc[-1]['feature']} highest by observed accuracy decrease "
        f"({_format_number(importance.iloc[-1]['mean_accuracy_decrease'])}); this is predictive association, "
        "not causal importance."
    )
    for filename, caption in (
        ("loss_curves.png", "Training and validation loss"),
        ("accuracy_curves.png", "Training and validation accuracy"),
        ("confusion_matrix.png", "Test confusion matrix (actual rows, predicted columns)"),
        ("roc_curve.png", f"Test ROC curve (AUC = {metrics['roc_auc']:.3f})"),
        ("permutation_importance.png", "Validation permutation importance (not causal)"),
    ):
        _add_picture_with_caption(document, MODEL_DIR / filename, caption)
        if filename == "loss_curves.png":
            document.add_paragraph(
                f"Final recorded training loss was {_format_number(history['loss'][-1])}; validation loss "
                f"was {_format_number(history['val_loss'][-1])}. Any gap may suggest overfitting/generalization "
                "differences but is uncertain with this validation sample."
            )
        elif filename == "accuracy_curves.png":
            document.add_paragraph(
                f"Final recorded training accuracy was {_format_number(history['accuracy'][-1])}; "
                f"validation accuracy was {_format_number(history['val_accuracy'][-1])}."
            )
        elif filename == "confusion_matrix.png":
            document.add_paragraph(
                f"Actual fail predictions: {metrics['confusion_matrix'][0][0]} correct and "
                f"{metrics['confusion_matrix'][0][1]} false passes; actual pass predictions: "
                f"{metrics['confusion_matrix'][1][0]} false fails and "
                f"{metrics['confusion_matrix'][1][1]} correct."
            )
        elif filename == "roc_curve.png":
            document.add_paragraph(f"The held-out test ROC-AUC is {metrics['roc_auc']:.3f}.")
        else:
            document.add_paragraph(
                f"Highest validation permutation importance was {importance.iloc[-1]['feature']} "
                f"({_format_number(importance.iloc[-1]['mean_accuracy_decrease'])} accuracy decrease). "
                "This ranking is not causal."
            )

    document.add_heading("8. Conclusion and Future Scope", level=1)
    document.add_paragraph(
        "This project demonstrates reproducible data inspection, visualization, preprocessing, and "
        "held-out evaluation. Correlation is not causation. The dataset represents one school and may "
        "not generalize to other settings; the sample and test set are small. Predictions must not drive "
        "real educational decisions without external validation, fairness checks, and human oversight. "
        "Future work could use cross-validation, compare other models, and evaluate additional and more "
        "representative data."
    )
    document.add_heading("9. References", level=1)
    references = [
        "Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008), 5-12. Bibliographic pagination/details should be checked against the original paper before formal submission.",
        "UCI Machine Learning Repository. (n.d.). Student Performance. https://archive.ics.uci.edu/dataset/320/student+performance",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. In Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 265-283.",
        "Fry, B. (2007). Visualizing data. O'Reilly Media. Edition/publication details should be checked against the copy used.",
    ]
    for reference in references:
        document.add_paragraph(reference, style="List Bullet")
    document.add_heading("Appendix: Repository and Key Code", level=1)
    document.add_paragraph("GitHub repository: [GitHub link placeholder]")
    document.add_paragraph("Key reproducibility commands: `python main.py`; `pytest -q`.")
    document.add_paragraph(
        "Target construction: pass_fail = 1 when G3 >= 10, otherwise 0. "
        "Feature construction removes G1, G2, G3 before fitting the train-only transformer."
    )
    output_path = REPORT_DIR / "Project_Report.docx"
    document.save(output_path)
    _sync_readme(metrics)
    print(f"Editable report written to {output_path}")


if __name__ == "__main__":
    generate_report()

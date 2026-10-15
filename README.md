# Student Performance Analysis and Prediction

College project demonstrating exploratory data analysis, data visualization, preprocessing, and a small deep-learning classifier on the UCI Student Performance mathematics data.

## Problem and objectives

Explore student grades and related attributes; prepare a reproducible classification dataset; compare a few Keras MLP settings using validation loss; and report held-out metrics with limitations. This is a demonstration, not a system for making educational decisions.

The project defines **pass** as final mathematics grade `G3 >= 10` (inclusive). This is the project's chosen threshold, not a universal grading rule. `G1`, `G2`, and `G3` are excluded from model inputs because they are grades from the same course and would leak direct information about the target.

## Dataset and citation

Download: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), mathematics data file `student-mat.csv`; semicolon-delimited. The source describes the dataset as CC BY 4.0. `src/download_data.py` verifies the expected 395 rows and 33 columns. The project does not modify the source dataset.

Reference: UCI Machine Learning Repository. (n.d.). *Student Performance*. https://archive.ics.uci.edu/dataset/320/student+performance

## Project structure

```text
student-performance/
├── data/student-mat.csv
├── outputs/
│   ├── figures/
│   └── model/
├── src/
│   ├── config.py
│   ├── data_types.py
│   ├── download_data.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── generate_report.py
├── notebooks/Student_Performance.ipynb
├── report/Project_Report.docx
├── tests/test_preprocessing.py
├── main.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## Install and run (Windows PowerShell)

Python 3.11 is recommended; supported project range is Python 3.10-3.12.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.download_data
python -m src.eda
python -m src.preprocessing
python -m src.train_model
python -m src.evaluate_model
python -m src.generate_report
pytest -q
```

Or run the full sequence with `python main.py`. Execute the notebook end-to-end with:

```powershell
jupyter nbconvert --to notebook --execute notebooks\Student_Performance.ipynb --output Student_Performance.executed.ipynb --output-dir outputs
```

The MLP is TensorFlow/Keras; if TensorFlow cannot be installed/imported, use Python 3.10-3.12 and consult the troubleshooting notes below. The project does not substitute a different model.

## Seven visualization stages

1. **Acquire** the UCI archive and validate its data shape.
2. **Parse** the semicolon-delimited CSV and inspect columns and types.
3. **Filter** by checking duplicates, missing values, grade range, and IQR outlier candidates; retain legitimate observations.
4. **Mine** summary statistics, class counts, correlations, and a Kruskal-Wallis test across study-time groups.
5. **Represent** grade distributions and relationships in labeled figures.
6. **Refine** plots with readable labels/titles and save PNGs at 200 dpi.
7. **Interact** with a Plotly HTML chart, hover details, and a grouping-variable dropdown.

## Main figures and outputs

- EDA figures: `outputs/figures/g3_histogram_kde.png`, `g3_by_studytime_boxplot.png`, `numeric_correlation_heatmap.png`, `absences_vs_g3.png`, and `mean_g3_by_studytime.png`.
- Class distribution: `outputs/figures/pass_fail_distribution.png`.
- Interactive chart: `outputs/figures/interactive_g3.html`.
- EDA summaries and IQR counts: CSV/JSON/Markdown files in `outputs/`.
- Model, metrics, curves, confusion matrix, ROC curve, and validation permutation importance: `outputs/model/`.
- Editable report: `report/Project_Report.docx`.
- Notebook: `notebooks/Student_Performance.ipynb`.

### Actual held-out results

Numbers below are synchronized from `outputs/model/metrics.json` by report generation. The test set contains only about 60 students, so results may be noisy.

<!-- METRICS:START -->
Test set: **60 students**  
Accuracy: **0.617**; precision: **0.698**; recall: **0.750**; F1: **0.723**; ROC-AUC: **0.614**  
Confusion matrix (actual fail/pass by predicted fail/pass): `[[7, 13], [10, 30]]`
<!-- METRICS:END -->

## Limitations

The data is observational and represents one school; correlation is not causation and results may not generalize. The test set is small. Permutation importance is predictive association, not causal importance. Do not use predictions to make real educational decisions without validation, fairness checks, and human oversight.

## References

- Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In *Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008)*, 5-12. Verify pagination/details against the original paper before formal submission.
- UCI Machine Learning Repository. (n.d.). *Student Performance*. https://archive.ics.uci.edu/dataset/320/student+performance
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
- Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. In *Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 265-283.
- Fry, B. (2007). *Visualizing data*. O'Reilly Media. Verify edition/publication details against the copy used.

## Short viva questions and answers

1. **Why remove G1, G2, and G3?** G3 defines the target, and G1/G2 are same-course grades that could let the model copy target-related information.
2. **Why use stratified splits?** They approximately preserve pass/fail proportions in train, validation, and test subsets.
3. **Why fit preprocessing only on training data?** It prevents validation/test information from influencing learned scaling or categories.
4. **What does ROC-AUC measure?** How well the model ranks positive cases above negative cases across classification thresholds.
5. **Does a correlation or feature importance show cause?** No. Both are descriptive/predictive associations, not causal evidence.

## GitHub upload (optional)

Review the staged files and supply your own remote URL and commit details:

```powershell
git init
git branch -M main
git add .
git commit -m "Add student performance analysis project"
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

These are instructions only; the project has not been published.

## Troubleshooting

- **Dataset download fails:** use the manual download link above and place `student-mat.csv` in `data\`; then rerun `python -m src.download_data`.
- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell session, then activate the venv.
- **TensorFlow install/import fails:** confirm `python --version` is 3.10, 3.11, or 3.12, recreate `.venv` with the matching `py -3.x` launcher, and reinstall requirements. Do not silently use another model.
- **Notebook command is missing:** install requirements in the active venv and run `python -m jupyter nbconvert ...`.

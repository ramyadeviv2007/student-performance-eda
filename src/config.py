"""Shared project paths and reproducibility settings."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODEL_DIR = OUTPUTS_DIR / "model"
REPORT_DIR = ROOT_DIR / "report"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
DATA_PATH = DATA_DIR / "student-mat.csv"
SEED = 42
PASS_THRESHOLD = 10


def ensure_project_dirs() -> None:
    """Create output directories used by the project."""
    for directory in (DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, MODEL_DIR, REPORT_DIR, NOTEBOOK_DIR):
        directory.mkdir(parents=True, exist_ok=True)

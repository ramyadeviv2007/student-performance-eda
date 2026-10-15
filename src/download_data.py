"""Download and validate the UCI Student Performance mathematics dataset."""

from __future__ import annotations

import sys
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_DIR, DATA_PATH, ensure_project_dirs

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
EXPECTED_SHAPE = (395, 33)


def download_data() -> Path:
    """Download the UCI archive, extract the mathematics CSV, and validate it."""
    ensure_project_dirs()
    if DATA_PATH.exists():
        import pandas as pd

        frame = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
        if frame.shape != EXPECTED_SHAPE:
            raise ValueError(f"Existing dataset has shape {frame.shape}; expected {EXPECTED_SHAPE}.")
        print(f"Validated existing dataset: {DATA_PATH} ({frame.shape[0]} rows, {frame.shape[1]} columns)")
        return DATA_PATH

    archive_path = DATA_DIR / "student-performance.zip"
    try:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "student-performance-project/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as archive:
            archive.write(response.read())
        with zipfile.ZipFile(archive_path) as archive:
            csv_member = next((name for name in archive.namelist() if Path(name).name == "student-mat.csv"), None)
            csv_content = archive.read(csv_member) if csv_member else None
            if csv_content is None:
                nested_name = next(
                    (name for name in archive.namelist() if Path(name).suffix.lower() == ".zip"), None
                )
                if nested_name is not None:
                    with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested_archive:
                        csv_member = next(
                            (name for name in nested_archive.namelist() if Path(name).name == "student-mat.csv"),
                            None,
                        )
                        if csv_member is not None:
                            csv_content = nested_archive.read(csv_member)
            if csv_content is None:
                raise FileNotFoundError("The downloaded UCI archive and its nested archives contain no student-mat.csv.")
            DATA_PATH.write_bytes(csv_content)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, StopIteration) as error:
        DATA_PATH.unlink(missing_ok=True)
        print(f"Could not download or extract the dataset: {error}", file=sys.stderr)
        print(
            "Manual fallback: download the Student Performance ZIP from "
            "https://archive.ics.uci.edu/static/public/320/student+performance.zip, "
            f"then place student-mat.csv in {DATA_DIR}.",
            file=sys.stderr,
        )
        raise RuntimeError("Dataset download failed; see the manual instructions above.") from error
    finally:
        archive_path.unlink(missing_ok=True)

    import pandas as pd

    frame = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
    if frame.shape != EXPECTED_SHAPE:
        DATA_PATH.unlink(missing_ok=True)
        raise ValueError(f"Downloaded dataset has shape {frame.shape}; expected {EXPECTED_SHAPE}.")
    print(f"Downloaded and validated dataset: {DATA_PATH} ({frame.shape[0]} rows, {frame.shape[1]} columns)")
    return DATA_PATH


if __name__ == "__main__":
    try:
        download_data()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

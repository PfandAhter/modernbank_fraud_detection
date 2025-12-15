"""Utilities for downloading and inspecting the raw fraud detection dataset."""

from __future__ import annotations

import subprocess
import zipfile
import shutil
from pathlib import Path

import pandas as pd


DATASET_SLUG = "valakhorasani/bank-transaction-dataset-for-fraud-detection"


def download_dataset(raw_dir: Path) -> Path:
    """Download the Kaggle dataset into ``raw_dir`` and return the zip file path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{DATASET_SLUG.split('/')[-1]}.zip"

    # Prefer the kaggle CLI if it's available on PATH (makes behavior identical on *nix)
    kaggle_cmd = shutil.which("kaggle")
    if kaggle_cmd:
        command = [
            kaggle_cmd,
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(raw_dir),
            "--force",
        ]
        try:
            subprocess.run(command, check=True)
            return zip_path
        except subprocess.CalledProcessError:
            # The kaggle CLI exists but failed (often due to missing credentials).
            # Fall back to the Python Kaggle API below which will give a clearer
            # diagnostic if credentials are not configured.
            pass

    # Fallback: try to use the Python kaggle package (requires ~/.kaggle/kaggle.json)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError as exc:  # package not installed at all
        raise FileNotFoundError(
            "'kaggle' CLI not found on PATH and the Python 'kaggle' package is not installed.\n"
            "Install the package with 'pip install kaggle' and configure API credentials as described at "
            "https://github.com/Kaggle/kaggle-api."
        ) from exc
    except Exception as exc:  # other import-time errors (e.g. top-level auth complaining about missing kaggle.json)
        raise RuntimeError(
            "The Python 'kaggle' package is present but failed to import. This often means the package tried to "
            "authenticate during import and couldn't find credentials. Make sure you have a valid 'kaggle.json' at "
            "%USERPROFILE%\\.kaggle\\kaggle.json (Windows) or set KAGGLE_USERNAME and KAGGLE_KEY environment variables. "
            "See https://github.com/Kaggle/kaggle-api for setup instructions."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - informative error path
        raise RuntimeError(
            "Kaggle API authentication failed. Make sure you have a valid 'kaggle.json' at "
            "%USERPROFILE%\\.kaggle\\kaggle.json (Windows) or set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables. See https://github.com/Kaggle/kaggle-api for details."
        ) from exc

    # This will download a zip named after the dataset (e.g. 'bank-transaction-dataset-for-fraud-detection.zip')
    api.dataset_download_files(DATASET_SLUG, path=str(raw_dir), unzip=False, force=True)
    return zip_path


def extract_zip(zip_path: Path, raw_dir: Path) -> None:
    """Extract the dataset archive into ``raw_dir``."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected dataset archive at {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)


def summarize_dataset(raw_dir: Path) -> None:
    """Load the first CSV in ``raw_dir`` and print basic dataset information."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    dataset_path = csv_files[0]
    df = pd.read_csv(dataset_path)

    print(f"Dataset file: {dataset_path.name}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values per column:")
    print(df.isnull().sum())


def main() -> None:
    """Download, extract, and summarize the fraud detection dataset."""
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    zip_path = download_dataset(raw_dir)
    extract_zip(zip_path, raw_dir)
    summarize_dataset(raw_dir)


if __name__ == "__main__":
    main()

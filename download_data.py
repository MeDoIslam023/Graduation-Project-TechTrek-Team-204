"""
Download & prepare the Home Credit Default Risk dataset from Kaggle.

Prerequisites:
    1. Create a Kaggle account and go to: Account -> API -> "Create New Token"
       This downloads a kaggle.json file.
    2. Place it at: ~/.kaggle/kaggle.json  (on Linux/Mac)
       or: C:\\Users\\<you>\\.kaggle\\kaggle.json  (on Windows)
    3. Accept the competition rules on the competition page (required once):
       https://www.kaggle.com/competitions/home-credit-default-risk/rules
    4. Install the kaggle CLI:
       pip install kaggle --break-system-packages

Usage:
    python download_data.py
"""

import os
import subprocess
import zipfile
from pathlib import Path

COMPETITION = "home-credit-default-risk"
DATA_DIR = Path("data/raw")

EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "HomeCredit_columns_description.csv",
    "sample_submission.csv",
]


def check_kaggle_credentials():
    """Make sure kaggle.json exists before trying to download."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Kaggle API token not found at {kaggle_json}.\n"
            "Go to Kaggle -> Account -> API -> 'Create New Token', "
            "then place the downloaded kaggle.json there."
        )
    # Kaggle CLI requires restrictive permissions on the token file
    os.chmod(kaggle_json, 0o600)


def download_dataset():
    """Download the competition files via the Kaggle CLI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle", "competitions", "download",
        "-c", COMPETITION,
        "-p", str(DATA_DIR),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def unzip_dataset():
    """Unzip any .zip files found in the data directory."""
    for zip_path in DATA_DIR.glob("*.zip"):
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()  # remove the zip after extracting


def verify_files():
    """Check that the expected CSVs are present."""
    missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print("WARNING: the following expected files are missing:")
        for f in missing:
            print(f"  - {f}")
    else:
        print("All expected files are present.")

    print("\nFiles in data/raw:")
    for f in sorted(DATA_DIR.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:<40} {size_mb:>8.2f} MB")


def main():
    check_kaggle_credentials()
    download_dataset()
    unzip_dataset()
    verify_files()


if __name__ == "__main__":
    main()

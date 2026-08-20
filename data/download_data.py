"""
Download the ManyThings.org Anki translation dataset.

Usage:
    python data/download_data.py --language fra

Languages:
    fra = English -> French
    spa = English -> Spanish
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path


DATASET_URLS = {
    "fra": "https://www.manythings.org/anki/fra-eng.zip",
    "spa": "https://www.manythings.org/anki/spa-eng.zip",
}


def download_dataset(language):
    if language not in DATASET_URLS:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    root = Path(__file__).resolve().parent
    raw_dir = root / "raw"
    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_path = raw_dir / f"{language}-eng.zip"

    url = DATASET_URLS[language]

    if not zip_path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(
            url,
            zip_path
        )

    print(f"Extracting {zip_path}")

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:
        archive.extractall(raw_dir)

    expected_file = (
        raw_dir /
        f"{language}-eng.txt"
    )

    if expected_file.exists():
        print(
            f"Dataset ready: {expected_file}"
        )
        return expected_file

    candidates = list(
        raw_dir.rglob("*.txt")
    )

    if not candidates:
        raise FileNotFoundError(
            "No translation text file found."
        )

    print(
        f"Dataset ready: {candidates[0]}"
    )

    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        default="fra",
        choices=["fra", "spa"]
    )

    args = parser.parse_args()

    download_dataset(
        args.language
    )

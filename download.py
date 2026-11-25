"""Step 1: download the raw data into data/raw/. Safe to re-run: finished downloads are skipped.

    MDC_API_KEY=... python download.py

- FLEURS Urdu (Google, CC-BY-4.0): about 10 hours of read Wikipedia sentences, from Hugging Face.
- Common Voice Urdu (Mozilla, CC0): about 6 GB of volunteers reading sentences, from the
  Mozilla Data Collective. Needs a free account, the dataset's terms accepted once on the website, and an
  API key (Account -> Credentials). Its terms forbid re-hosting the audio, so it is never uploaded anywhere.
"""

import argparse
import os
import urllib.request
from pathlib import Path

FLEURS = "https://huggingface.co/api/datasets/google/fleurs/parquet/ur_pk/{split}/0.parquet"
CV_URDU = "cmu5x1w3400cyo107st8cgsoq"  # Common Voice Scripted Speech 27.0 - Urdu (last part of its URL; pinned for reproducibility)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cv-dataset", default=CV_URDU, help="Mozilla Data Collective dataset ID")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--find-cv", action="store_true", help="list the Urdu speech datasets currently on the site")
    a = ap.parse_args()

    if a.find_cv:  # Mozilla replaces old releases, so pinned IDs eventually stop working
        from datacollective import list_datasets
        for d in list_datasets(locale="ur", task="ASR", sort="newest").items:
            print(d.id, "|", d.name, "|", d.licenseAbbreviation, "|", f"{(d.sizeBytes or 0) / 1e9:.1f} GB")
        return

    fleurs = Path(a.raw) / "fleurs"
    fleurs.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        dest = fleurs / f"{split}.parquet"
        if dest.exists():
            print(f"FLEURS {split}: already downloaded")
            continue
        print(f"FLEURS {split}: downloading ...", flush=True)
        urllib.request.urlretrieve(FLEURS.format(split=split), f"{dest}.part")
        os.replace(f"{dest}.part", dest)  # only a complete file gets the final name

    if a.skip_cv:
        return
    if not os.environ.get("MDC_API_KEY") and not Path(".env").exists():  # the client also reads .env
        raise SystemExit("Set MDC_API_KEY or put MDC_API_KEY=... in a .env file (Mozilla Data Collective -> "
                         "Account -> Credentials), or pass --skip-cv to continue with FLEURS only.")
    from datacollective import download_dataset
    path = download_dataset(a.cv_dataset, download_directory=str(Path(a.raw) / "common_voice"), extract=True)
    print("Common Voice extracted to", path)


if __name__ == "__main__":
    main()

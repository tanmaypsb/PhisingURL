"""Download the UCI archive and parse its ARFF data using the standard library."""
import csv
import re
import urllib.request
import zipfile
from pathlib import Path
from .utils import DATA_RAW

UCI_URL = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/327/phishing+websites"
RAW_CSV = DATA_RAW / "raw_dataset.csv"

def load_raw_dataset(force_download=False):
    if RAW_CSV.exists() and not force_download:
        return _read_csv(RAW_CSV)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    archive = DATA_RAW / "uci_phishing_websites.zip"
    try:
        urllib.request.urlretrieve(UCI_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            arff_name = next(n for n in zf.namelist() if n.endswith("Training Dataset.arff"))
            raw = zf.read(arff_name).decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the UCI dataset from {UCI_URL}: {exc}. "
            f"Place the original ARFF or a CSV named raw_dataset.csv in {DATA_RAW}."
        ) from exc
    columns, rows = [], []
    in_data = False
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            if line.lower().startswith("@attribute"):
                match = re.match(r"@attribute\s+([^\s]+)", line, flags=re.I)
                if match:
                    columns.append(match.group(1).strip("'\""))
            elif line.lower() == "@data":
                in_data = True
            continue
        rows.append(next(csv.reader([line], skipinitialspace=True)))
    if not columns or not rows:
        raise RuntimeError("UCI archive was downloaded but its ARFF schema/data could not be parsed.")
    if any(len(row) != len(columns) for row in rows):
        raise RuntimeError("ARFF row width does not match the declared schema.")
    with RAW_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(columns); writer.writerows(rows)
    return _read_csv(RAW_CSV)

def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path, na_values=["?", "NA", ""])

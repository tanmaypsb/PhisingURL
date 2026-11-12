from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "outputs"
FIGURES = OUT / "figures"
METRICS = OUT / "metrics"
STATS = OUT / "statistical_results"
SUMMARIES = OUT / "summaries"

def ensure_directories():
    for path in (DATA_RAW, DATA_PROCESSED, FIGURES, METRICS, STATS, SUMMARIES,
                 PROJECT_ROOT / "models"):
        path.mkdir(parents=True, exist_ok=True)

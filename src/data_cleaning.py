import pandas as pd
from .data_loader import RAW_CSV
from .utils import DATA_PROCESSED, SUMMARIES

TARGET_RAW = "Result"
TARGET = "is_phishing"

def clean_dataset(raw):
    SUMMARIES.mkdir(parents=True, exist_ok=True); DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    original_shape = raw.shape
    missing = pd.DataFrame({"missing_count": raw.isna().sum(),
                            "missing_percent": raw.isna().mean().mul(100)})
    missing.to_csv(SUMMARIES / "missing_values_summary.csv", index_label="column")
    duplicates = int(raw.duplicated().sum())
    data = raw.drop_duplicates().copy()
    if TARGET_RAW not in data:
        raise ValueError(f"Expected target column {TARGET_RAW!r}; found {list(raw.columns)}")
    allowed = {-1, 0, 1}
    converted = data.apply(pd.to_numeric, errors="coerce")
    invalid = converted.isna().any(axis=1) | ~converted.drop(columns=[TARGET_RAW]).isin(allowed).all(axis=1)
    target_valid = converted[TARGET_RAW].isin([-1, 1])
    invalid_count = int((invalid | ~target_valid).sum())
    data = converted.loc[~(invalid | ~target_valid)].copy()
    # Preserve only verified source labels; positive class explicitly means phishing.
    data[TARGET] = (data.pop(TARGET_RAW) == -1).astype("int8")
    data.to_csv(DATA_PROCESSED / "cleaned_dataset.csv", index=False)
    feature_cols = [c for c in data.columns if c != TARGET]
    outlier_rows = []
    for col in feature_cols:
        q1, q3 = data[col].quantile([.25, .75]); iqr = q3-q1
        lo, hi = q1-1.5*iqr, q3+1.5*iqr
        outlier_rows.append({"feature":col,"iqr_lower":lo,"iqr_upper":hi,
                             "flagged_count":int(((data[col]<lo)|(data[col]>hi)).sum()),
                             "action":"reported; retained (discrete security indicator)"})
    pd.DataFrame(outlier_rows).to_csv(SUMMARIES / "outlier_summary.csv", index=False)
    summary = pd.DataFrame([
        {"item":"rows_before","value":original_shape[0]}, {"item":"columns_before","value":original_shape[1]},
        {"item":"duplicate_rows_removed","value":duplicates}, {"item":"invalid_or_missing_rows_removed","value":invalid_count},
        {"item":"rows_after","value":len(data)}, {"item":"columns_after_including_target","value":data.shape[1]},
        {"item":"missing_cells_before","value":int(raw.isna().sum().sum())},
        {"item":"outlier_policy","value":"IQR flags reported; no discrete feature rows removed"},
        {"item":"target_mapping","value":"Result=-1 phishing => is_phishing=1; Result=1 legitimate => 0"},
        {"item":"class_counts","value":str(data[TARGET].value_counts().sort_index().to_dict())},
        {"item":"source","value":"UCI Phishing Websites dataset 327"},
    ])
    summary.to_csv(SUMMARIES / "cleaning_summary.csv", index=False)
    return data, missing

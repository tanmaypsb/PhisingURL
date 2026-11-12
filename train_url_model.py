"""Train/evaluate a separate Random Forest using only URL-derived UCI features."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.data_cleaning import TARGET
from src.url_feature_extractor import FEATURE_DEFINITIONS, URL_FEATURES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"
MODEL_PATH = ROOT / "models" / "url_only_random_forest.pkl"
SCHEMA_PATH = ROOT / "models" / "url_only_feature_schema.json"
METRICS_PATH = ROOT / "outputs" / "url_only_model_metrics.csv"
CONFUSION_PATH = ROOT / "outputs" / "url_only_confusion_matrices.csv"
REPORT_PATH = ROOT / "outputs" / "url_only_model_report.txt"


def _measure(name, estimator, X_test, y_test):
    prediction = estimator.predict(X_test)
    probability = estimator.predict_proba(X_test)[:, list(estimator.classes_).index(1)]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, prediction),
        "precision_phishing": precision_score(y_test, prediction, zero_division=0),
        "recall_phishing": recall_score(y_test, prediction, zero_division=0),
        "f1_phishing": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "test_rows": len(y_test),
    }, confusion_matrix(y_test, prediction, labels=[0, 1])


def train():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Cleaned UCI dataset not found: {DATA_PATH}. Run the existing project pipeline first.")
    data = pd.read_csv(DATA_PATH)
    missing = [name for name in (*URL_FEATURES, TARGET) if name not in data.columns]
    if missing:
        raise ValueError(f"Cleaned data is missing required columns: {missing}")
    X = data.loc[:, URL_FEATURES].copy()
    y = data[TARGET].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected cleaned target is_phishing to contain classes 0 and 1.")

    # Same deterministic split used by model_training.py, enabling a fair
    # held-out comparison with the existing saved full-feature Random Forest.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    results = []
    matrices = []
    for name, estimator in (("Dummy baseline", baseline), ("URL-only Random Forest", model)):
        metrics, matrix = _measure(name, estimator, X_test, y_test)
        results.append(metrics)
        for actual, row in zip((0, 1), matrix):
            for predicted, count in zip((0, 1), row):
                matrices.append({"model": name, "actual_class": actual, "predicted_class": predicted, "count": int(count)})

    # Evaluate the existing fitted model on this exact held-out split too.
    full_model_path = ROOT / "models" / "best_model.pkl"
    if not full_model_path.is_file():
        raise FileNotFoundError(f"Existing full-feature model not found: {full_model_path}")
    full_model = joblib.load(full_model_path)
    full_features = list(getattr(full_model, "feature_names_in_", []))
    if not full_features or not set(full_features).issubset(data.columns):
        raise ValueError("Existing model feature schema is not compatible with the cleaned dataset.")
    full_metrics, full_matrix = _measure(
        "Existing full-feature Random Forest", full_model,
        data.loc[X_test.index, full_features], y_test,
    )
    results.append(full_metrics)
    for actual, row in zip((0, 1), full_matrix):
        for predicted, count in zip((0, 1), row):
            matrices.append({"model": "Existing full-feature Random Forest", "actual_class": actual, "predicted_class": predicted, "count": int(count)})

    ROOT.joinpath("models").mkdir(exist_ok=True)
    ROOT.joinpath("outputs").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    schema = {
        "model_file": MODEL_PATH.name,
        "target": TARGET,
        "class_labels": {"0": "Legitimate", "1": "Phishing"},
        "feature_order": list(URL_FEATURES),
        "feature_definitions_and_encodings": FEATURE_DEFINITIONS,
        "training_source": "Existing cleaned UCI Phishing Websites dataset (dataset 327)",
        "split": {"test_size": 0.2, "stratified": True, "random_state": 42},
        "limitations": [
            "The UCI table contains pre-extracted feature values but no original URL strings.",
            "The shortened-URL feature uses an explicit offline provider list.",
            "No DNS, WHOIS, TLS certificate, HTTP, webpage, or reputation features are used.",
        ],
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(METRICS_PATH, index=False)
    pd.DataFrame(matrices).to_csv(CONFUSION_PATH, index=False)
    report = [
        "URL-only vs full-feature model evaluation",
        "Same stratified 80:20 split, random_state=42; phishing is the positive class.",
        "The full-feature model is the existing saved Random Forest evaluated on this held-out split.",
        "The UCI table has no raw URL strings, so it cannot provide URL-level held-out validation.",
        "",
        pd.DataFrame(results).to_string(index=False),
        "",
        "Interpretation: URL-only performance uses only eight URL-string features; the full model also uses webpage, certificate, domain, DNS, and reputation indicators. These scores describe the UCI engineered-feature benchmark, not deployment performance on newly collected URLs.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Trained URL-only Random Forest with {len(URL_FEATURES)} ordered features.")
    print(f"Train rows: {len(X_train):,}; held-out test rows: {len(X_test):,}.")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved feature schema: {SCHEMA_PATH}")
    return model, pd.DataFrame(results)


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        raise SystemExit(f"URL-only model training failed: {exc}") from exc

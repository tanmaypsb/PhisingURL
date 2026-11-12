"""Interactive command-line URL phishing prediction using the URL-only model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.url_feature_extractor import URL_FEATURES, InvalidURL, extract_url_features, normalize_url

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "url_only_random_forest.pkl"
SCHEMA_PATH = ROOT / "models" / "url_only_feature_schema.json"


def load_detector():
    if not MODEL_PATH.is_file() or not SCHEMA_PATH.is_file():
        raise FileNotFoundError("URL-only model/schema is missing. Train it first with: python train_url_model.py")
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    expected = list(schema.get("feature_order", []))
    if expected != list(URL_FEATURES):
        raise ValueError("Saved feature schema does not match the URL feature extractor order.")
    if list(getattr(model, "feature_names_in_", [])) != expected:
        raise ValueError("Saved model's feature names/order do not match the saved schema.")
    return model, expected


def predict_one(url, model, feature_order):
    normalized = normalize_url(url)
    features = extract_url_features(normalized)
    if list(features) != feature_order:
        raise ValueError("Extracted feature order does not match the trained model schema.")
    row = pd.DataFrame([[features[name] for name in feature_order]], columns=feature_order)
    prediction = int(model.predict(row)[0])
    if prediction not in (0, 1):
        raise ValueError(f"Unexpected model class {prediction!r}.")
    probabilities = None
    if hasattr(model, "predict_proba"):
        output = model.predict_proba(row)[0]
        probabilities = {int(label): float(prob) for label, prob in zip(model.classes_, output)}
    return normalized, features, prediction, probabilities


def main():
    model, feature_order = load_detector()
    print("URL Phishing Detection (URL-only UCI model)")
    print("Enter an absolute http:// or https:// URL. Type 'exit' to quit.\n")
    while True:
        try:
            url = input("Enter a URL or type 'exit': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting URL detector.")
            break
        if url.lower() == "exit":
            print("Exiting URL detector.")
            break
        try:
            normalized, features, prediction, probabilities = predict_one(url, model, feature_order)
        except InvalidURL as exc:
            print(f"Invalid URL: {exc}\n")
            continue
        except Exception as exc:
            print(f"Prediction error: {exc}\n")
            continue
        print(f"Prediction: {'PHISHING' if prediction == 1 else 'LEGITIMATE'}")
        if probabilities is not None and 1 in probabilities:
            print(f"Phishing Probability: {probabilities[1]:.2%}")
            print(f"Legitimate Probability: {probabilities.get(0, 0.0):.2%}")
        print("Extracted UCI URL features:")
        for name in feature_order:
            print(f"  {name:<34} {features[name]}")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Unable to start URL detector: {exc}") from exc

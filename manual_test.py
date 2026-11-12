"""Interactive manual testing for the saved phishing detection model.

Run from any working directory with the project's Python environment:
    python manual_test.py
"""
from pathlib import Path
import sys

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pkl"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "cleaned_dataset.csv"
TARGET_COLUMN = "is_phishing"
LABELS = {0: "Legitimate", 1: "Phishing"}


def load_assets():
    """Load the fitted model and cleaned observations, checking their schema."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Cleaned dataset is missing target column {TARGET_COLUMN!r}.")

    expected_features = list(getattr(model, "feature_names_in_", []))
    if not expected_features:
        raise ValueError("The saved model does not expose feature_names_in_; cannot safely verify feature order.")
    dataset_features = [column for column in data.columns if column != TARGET_COLUMN]
    if len(expected_features) != len(set(expected_features)):
        raise ValueError("The model contains duplicate feature names.")
    if set(expected_features) != set(dataset_features):
        missing = sorted(set(expected_features) - set(dataset_features))
        extra = sorted(set(dataset_features) - set(expected_features))
        raise ValueError(f"Model and dataset feature schemas differ. Missing in dataset: {missing}; extra: {extra}")
    if data.empty:
        raise ValueError("Cleaned dataset contains no samples.")
    return model, data, expected_features


def print_features(features):
    """Display the named website indicators in model input order."""
    print("\nWebsite feature values (encoded dataset indicators):")
    for name, value in features.items():
        print(f"  {name:<34} {value:g}")


def choose_menu_option():
    options = {
        "1": "Random sample from dataset",
        "2": "Sample by row index",
        "3": "Enter feature values manually",
        "q": "Quit",
    }
    print("\nChoose a test method:")
    for key, description in options.items():
        print(f"  {key}. {description}")
    while True:
        answer = input("Selection: ").strip().lower()
        if answer in options:
            return answer
        print("Please enter 1, 2, 3, or q.")


def choose_row_index(data):
    """Ask for a zero-based row position, reprompting on invalid input."""
    maximum = len(data) - 1
    while True:
        answer = input(f"Enter a row index from 0 to {maximum} (zero-based): ").strip()
        try:
            index = int(answer)
        except ValueError:
            print("Please enter a whole-number row index.")
            continue
        if 0 <= index < len(data):
            return index
        print(f"Index out of range. Choose a number from 0 to {maximum}.")


def enter_features(data, feature_names):
    """Prompt for every model feature and validate against observed codes."""
    entered = {}
    print("\nEnter each encoded feature value. Allowed values are shown in brackets.")
    for name in feature_names:
        allowed = sorted(pd.to_numeric(data[name], errors="coerce").dropna().unique().tolist())
        allowed_text = ", ".join(f"{value:g}" for value in allowed)
        while True:
            answer = input(f"{name} [{allowed_text}]: ").strip()
            try:
                value = float(answer)
            except ValueError:
                print("Please enter a numeric value from the allowed list.")
                continue
            if value not in allowed:
                print(f"Invalid value for {name}. Choose one of: {allowed_text}.")
                continue
            entered[name] = value
            break
    return pd.Series(entered, name="manual input", dtype=float)


def predict(model, features, feature_names):
    """Predict from a DataFrame whose columns match the fitted model exactly."""
    sample = pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)
    prediction = int(model.predict(sample)[0])
    if prediction not in LABELS:
        raise ValueError(f"Model returned unexpected class {prediction!r}; expected 0 or 1.")
    probabilities = None
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(sample)[0]
        probabilities = {int(label): float(probability) for label, probability in zip(model.classes_, values)}
    return prediction, probabilities


def display_result(prediction, probabilities, actual=None):
    print(f"\nPredicted website class: {LABELS[prediction].upper()}")
    if probabilities is not None:
        print("Prediction probabilities:")
        for label in (0, 1):
            if label in probabilities:
                print(f"  {LABELS[label]:<12} {probabilities[label]:.1%}")
    if actual is not None:
        actual = int(actual)
        if actual not in LABELS:
            raise ValueError(f"Dataset contains unexpected target label {actual!r}.")
        print(f"Actual website class:    {LABELS[actual].upper()}")
        print(f"Prediction correct:      {'Yes' if prediction == actual else 'No'}")


def run():
    model, data, feature_names = load_assets()
    print("Phishing Website Detection — Manual Test")
    print(f"Loaded model: {type(model).__name__}")
    print(f"Loaded cleaned dataset: {len(data):,} rows, {len(feature_names)} features")
    print("Target encoding: 1 = Phishing; 0 = Legitimate")

    while True:
        choice = choose_menu_option()
        if choice == "q":
            print("Exiting manual test.")
            return

        actual = None
        if choice == "1":
            index = int(pd.Series(range(len(data))).sample(n=1).iloc[0])
            row = data.iloc[index]
            features = row[feature_names]
            actual = row[TARGET_COLUMN]
            print(f"\nRandom dataset sample at row index {index}:")
        elif choice == "2":
            index = choose_row_index(data)
            row = data.iloc[index]
            features = row[feature_names]
            actual = row[TARGET_COLUMN]
            print(f"\nDataset sample at row index {index}:")
        else:
            features = enter_features(data, feature_names)
            print("\nManually entered website features:")

        print_features(features)
        prediction, probabilities = predict(model, features, feature_names)
        display_result(prediction, probabilities, actual)


if __name__ == "__main__":
    try:
        run()
    except (EOFError, KeyboardInterrupt):
        print("\nManual test cancelled.")
    except Exception as exc:
        print(f"\nUnable to run manual test: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

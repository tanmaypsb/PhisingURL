import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.url_feature_extractor import (
    InvalidURL, URL_FEATURES, extract_url_features, normalize_url,
)

ROOT = Path(__file__).resolve().parents[1]


class URLFeatureExtractorTests(unittest.TestCase):
    def test_plain_https_url(self):
        url = "https://example.com"
        self.assertEqual(normalize_url("  " + url + "  "), url)
        features = extract_url_features(url)
        self.assertEqual(tuple(features), URL_FEATURES)
        self.assertEqual(features["having_IP_Address"], 1)
        self.assertEqual(features["URL_Length"], 1)
        self.assertEqual(features["port"], 1)

    def test_ip_based_urls(self):
        for url in ("http://192.0.2.10/path", "https://[2001:db8::1]/",
                    "http://0x58.0xCC.0xCA.0x62/path"):
            with self.subTest(url=url):
                self.assertEqual(extract_url_features(url)["having_IP_Address"], -1)

    def test_url_length_bins(self):
        def sized_url(length):
            prefix = "https://example.com/"
            return prefix + "a" * (length - len(prefix))
        for length, expected in ((53, 1), (54, 0), (75, 0), (76, -1)):
            with self.subTest(length=length):
                self.assertEqual(extract_url_features(sized_url(length))["URL_Length"], expected)

    def test_shortener_and_suspicious_url_features(self):
        self.assertEqual(extract_url_features("https://bit.ly/abc")["Shortining_Service"], -1)
        features = extract_url_features("https://secure-example.com/a//b?next=x@y.test")
        self.assertEqual(features["having_At_Symbol"], -1)
        self.assertEqual(features["double_slash_redirecting"], -1)
        self.assertEqual(features["Prefix_Suffix"], -1)
        self.assertEqual(extract_url_features("http://https-example.com/")["HTTPS_token"], -1)

    def test_nonstandard_port(self):
        self.assertEqual(extract_url_features("http://example.com:8080/")["port"], -1)
        self.assertEqual(extract_url_features("https://example.com:443/")["port"], 1)

    def test_invalid_urls_are_rejected(self):
        for value in ("", "example.com", "ftp://example.com", "https:///no-host",
                      "https://example.com:99999/", "https://exa mple.com", "https://example.com\\@evil.test"):
            with self.subTest(value=value), self.assertRaises(InvalidURL):
                normalize_url(value)


class URLModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "models" / "url_only_random_forest.pkl"
        schema_path = ROOT / "models" / "url_only_feature_schema.json"
        if not model_path.exists() or not schema_path.exists():
            raise RuntimeError("Train the URL-only model before running tests: python train_url_model.py")
        cls.model = joblib.load(model_path)
        cls.schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def test_feature_schema_matches_trained_model(self):
        self.assertEqual(self.schema["feature_order"], list(URL_FEATURES))
        self.assertEqual(list(self.model.feature_names_in_), list(URL_FEATURES))
        clean = pd.read_csv(ROOT / "data" / "processed" / "cleaned_dataset.csv")
        ordinary = extract_url_features("https://example.com")
        for name in URL_FEATURES:
            self.assertIn(name, clean.columns)
            observed = set(pd.Series(clean[name].unique()).dropna())
            self.assertTrue(observed.issubset({-1, 0, 1}))
            self.assertIn(ordinary[name], observed)

    def test_model_prediction_and_probability(self):
        features = extract_url_features("https://example.com/login")
        row = pd.DataFrame([[features[name] for name in URL_FEATURES],], columns=URL_FEATURES)
        prediction = int(self.model.predict(row)[0])
        probability = self.model.predict_proba(row)[0]
        self.assertIn(prediction, (0, 1))
        self.assertAlmostEqual(float(probability.sum()), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Phishing Website Detection Using Machine Learning

Reproducible pipeline using the **Phishing Websites** dataset
from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)
(dataset 327; [direct archive](https://archive.ics.uci.edu/static/public/327/phishing+websites)).
The source's `Result` values are `-1` for phishing and `1` for legitimate. The
pipeline models `is_phishing` with 1 meaning phishing. The raw ARFF observations
are preserved as `data/raw/raw_dataset.csv`; a separate cleaned table is saved
under `data/processed/`.

## Run

From this directory, create/use an isolated environment and install dependencies:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

The first run downloads the UCI archive. If automatic download is unavailable,
download the source archive and extract `Training Dataset.arff`; alternatively,
place a compatible already-converted CSV named `raw_dataset.csv` in
`data/raw/`. The expected CSV uses the UCI column names and `Result` target.

## Workflow and methods

`main.py` inspects schema, missingness and duplicates, validates feature codes
and target labels, removes exact duplicate/invalid records, reports discrete
IQR flags without deleting valid extremes, and generates descriptive statistics
and EDA. Inference uses Welch's independent samples t-test on encoded
`URL_Length` and a Pearson chi-square test on `Prefix_Suffix`; the two p-values
receive Bonferroni correction at family-wise alpha 0.05. Effect sizes and the
t-test confidence interval are saved.

A stratified 80:20 split with `random_state=42` reserves the test data for final
evaluation. The models are a majority-class `DummyClassifier`, scaled logistic
regression, decision tree, and random forest. No resampling or feature selection
uses test data. The saved model is selected by phishing-class F1, then recall
and ROC-AUC. Feature importance is predictive, not causal.

## Outputs

- `outputs/summaries/`: cleaning, missingness, descriptive statistics, class distribution, outlier flags, run summary.
- `outputs/figures/`: class counts, missingness, correlation, feature distributions/boxplots, comparison, confusion matrices, ROC/PR, feature importance.
- `outputs/statistical_results/`: structured test results and interpretations.
- `outputs/metrics/`: model metrics, classification reports, feature importances.
- `models/best_model.pkl`: fitted best model.
- `notebooks/phishing_detection_analysis.ipynb`: workflow notebook; run from this project directory.

Python 3.10+ is recommended. Results are based on a benchmark dataset and do
not establish real-world performance on newly observed URLs.

## URL-only prediction (separate model)

The original `models/best_model.pkl` uses all 30 UCI features; many require
webpage content, certificates, DNS/WHOIS, HTTP behavior, or external reputation
data. It cannot safely predict from an unvisited URL alone. A separate model
uses only eight values that can be derived from an absolute URL string:

- `having_IP_Address`
- `URL_Length`
- `Shortining_Service`
- `having_At_Symbol`
- `double_slash_redirecting`
- `Prefix_Suffix`
- `port`
- `HTTPS_token`

Of the original 30 columns, `having_IP_Address`, `URL_Length`,
`Shortining_Service`, `having_At_Symbol`, `double_slash_redirecting`,
`Prefix_Suffix`, `port`, and `HTTPS_token` are used here. `having_Sub_Domain`
needs a reliable registrable-domain/public-suffix boundary and is omitted.
`SSLfinal_State` needs certificate inspection; `Domain_registeration_length`
and `age_of_domain` need registration/WHOIS data; `Favicon`, `Request_URL`,
`URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email`, `on_mouseover`,
`RightClick`, `popUpWidnow`, and `Iframe` need page content; `Abnormal_URL`
needs identity/registration checks; `Redirect` needs an HTTP request; and
`DNSRecord`, `web_traffic`, `Page_Rank`, `Google_Index`,
`Links_pointing_to_page`, and `Statistical_report` need DNS or external index,
reputation, or threat-report data. None of those are filled with guesses.

Encodings follow the UCI Phishing Websites feature guide distributed with the
dataset. URL length is `1` below 54 characters, `0` from 54 through 75, and `-1`
above 75. Binary encodings use the UCI feature's `-1`/`1` categories. The
shortening-service check uses a small, explicit offline list; it does not resolve
redirects. Subdomain depth is omitted because reliably identifying the
registrable domain requires a Public Suffix List. URL parsing uses Python's
standard library and performs no network requests.

Train the separate model and write its schema and held-out comparison outputs:

```powershell
.\.venv\Scripts\python.exe train_url_model.py
```

Start the interactive URL detector (enter one or more absolute `http://` or
`https://` URLs, then type `exit`):

```powershell
.\.venv\Scripts\python.exe predict_url.py
```

Run extractor and model-interface tests after training:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The URL-only Random Forest and its ordered feature metadata are saved separately
as `models/url_only_random_forest.pkl` and
`models/url_only_feature_schema.json`. Metrics and confusion-matrix counts are
saved to `outputs/url_only_model_metrics.csv` and
`outputs/url_only_confusion_matrices.csv`; `outputs/url_only_model_report.txt`
summarizes the comparison. Evaluation uses the same stratified split as the
existing full-feature evaluation: `random_state=42`, 80:20 train/test. The saved
full-feature Random Forest is evaluated on that same held-out test split.

The UCI dataset contains engineered feature values, not source URL strings.
Therefore it supports training the URL-only model on its existing labels but
does not support testing the extractor against original held-out URLs. Its test
metrics are feature-table benchmark metrics, not a measure of live-web safety.
The CLI's phishing probability is the model's `predict_proba` output, not a
guarantee or a calibrated real-world risk estimate. The model does not check
certificate validity, DNS, current redirects, page content, or threat feeds.

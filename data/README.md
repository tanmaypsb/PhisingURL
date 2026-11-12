# Dataset provenance

Dataset: **Phishing Websites** (UCI Machine Learning Repository, dataset 327).
Source page: https://archive.ics.uci.edu/dataset/327/phishing+websites
Direct archive: https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
Original file: `Training Dataset.arff`; the pipeline converts its original rows to
`raw/raw_dataset.csv` without changing values. The target is `Result` (`-1`
phishing, `1` legitimate); the modeling target is mapped to `is_phishing`
(`1` phishing, `0` legitimate). No observations are fabricated.

The raw dataset is preserved. `processed/cleaned_dataset.csv` is a separate
dataset with duplicates removed, invalid feature codes excluded, and the
documented binary target. IQR outliers are reported but not removed because the
features are mostly discrete security indicators and extreme codes can carry
signal.

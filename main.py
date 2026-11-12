"""Run the complete, reproducible phishing website analysis."""
from src.utils import ensure_directories, SUMMARIES
from src.data_loader import load_raw_dataset, SOURCE_PAGE
from src.data_cleaning import clean_dataset
from src.eda import run_eda
from src.statistical_analysis import run_statistical_tests
from src.model_training import train_models
from src.model_evaluation import evaluate_models, save_importance

def main():
    ensure_directories()
    print("[1/7] Acquiring/loading UCI Phishing Websites dataset...")
    raw=load_raw_dataset()
    print(f"      Raw dimensions: {raw.shape}; source: {SOURCE_PAGE}")
    print("[2/7] Inspecting schema, missingness, duplicates, labels, and cleaning...")
    print(f"      Columns/dtypes:\n{raw.dtypes.to_string()}")
    data,missing=clean_dataset(raw)
    print(f"      Clean dimensions: {data.shape}; class counts: {data.is_phishing.value_counts().to_dict()}")
    print("[3/7] Generating descriptive statistics and EDA figures...")
    run_eda(data,missing)
    print("[4/7] Running Welch t-test and chi-square test (Bonferroni adjusted)...")
    tests=run_statistical_tests(data)
    print(tests[["test","feature","p_value","adjusted_p_value_bonferroni","reject_null"]].to_string(index=False))
    print("[5/7] Creating stratified 80:20 split and fitting four prespecified models...")
    X_train,X_test,y_train,y_test,models=train_models(data)
    print(f"      Train rows={len(y_train)}, held-out test rows={len(y_test)}")
    print("[6/7] Evaluating models against majority-class baseline...")
    metrics,best_name=evaluate_models(X_test,y_test,models)
    importance=save_importance(models,X_train.columns)
    print(metrics.to_string(index=False))
    best=metrics.loc[metrics.model==best_name].iloc[0]
    (SUMMARIES/"run_summary.txt").write_text(
        f"Dataset: UCI Phishing Websites (327)\nRaw rows/columns: {raw.shape[0]}/{raw.shape[1]}\nClean rows/features: {data.shape[0]}/{data.shape[1]-1}\nBest by phishing F1, then recall and ROC-AUC: {best_name}\n"
        f"Accuracy={best.accuracy:.4f}; precision_phishing={best.precision_phishing:.4f}; recall_phishing={best.recall_phishing:.4f}; F1_phishing={best.f1_phishing:.4f}; ROC-AUC={best.roc_auc:.4f}\n"
        f"Top 10 Random Forest impurity importances:\n{importance.head(10).to_string()}\n\nImportance describes predictive splits, not causal effects.\n",encoding="utf-8")
    print("[7/7] Saving outputs and model...")
    print("      Completed. See outputs/ and models/best_model.pkl")

if __name__=="__main__":
    try: main()
    except Exception as exc:
        raise SystemExit(f"Pipeline failed: {exc}. See README.md for setup and dataset recovery steps.") from exc

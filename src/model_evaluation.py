import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,
    confusion_matrix,classification_report,roc_curve,precision_recall_curve,average_precision_score)
from .utils import FIGURES,METRICS,PROJECT_ROOT
from .data_cleaning import TARGET

def evaluate_models(X_test,y_test,models):
    rows=[]; reports=[]; score_store={};
    for name,model in models.items():
        pred=model.predict(X_test)
        score=model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else np.zeros(len(y_test))
        auc=roc_auc_score(y_test,score) if len(np.unique(y_test))==2 else np.nan
        rows.append({"model":name,"accuracy":accuracy_score(y_test,pred),"precision_phishing":precision_score(y_test,pred,zero_division=0),"recall_phishing":recall_score(y_test,pred,zero_division=0),"f1_phishing":f1_score(y_test,pred,zero_division=0),"roc_auc":auc,"test_rows":len(y_test)})
        reports.append(f"{name}\n{classification_report(y_test,pred,target_names=['Legitimate','Phishing'],zero_division=0)}")
        score_store[name]=(pred,score)
        cm=confusion_matrix(y_test,pred,labels=[0,1]); fig,ax=plt.subplots(figsize=(5,4)); import seaborn as sns; sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Legitimate","Phishing"],yticklabels=["Legitimate","Phishing"],ax=ax); ax.set(title=f"{name} confusion matrix",xlabel="Predicted class",ylabel="Actual class"); fig.tight_layout(); filename="confusion_matrix_dummy.png" if name=="DummyClassifier" else f"confusion_matrix_{name.lower().replace(' ','_')}.png"; fig.savefig(FIGURES/filename,dpi=200); plt.close(fig)
    metrics=pd.DataFrame(rows); METRICS.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(METRICS/"model_comparison.csv",index=False)
    (METRICS/"classification_reports.txt").write_text("\n\n".join(reports),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(8,5));
    for name,(pred,score) in score_store.items():
        if name!="DummyClassifier":
            fpr,tpr,_=roc_curve(y_test,score); ax.plot(fpr,tpr,label=f"{name} (AUC={roc_auc_score(y_test,score):.3f})")
    ax.plot([0,1],[0,1],"k--",label="Chance"); ax.set(title="ROC curves on held-out test set",xlabel="False positive rate",ylabel="True positive rate"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"roc_curve.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for name,(pred,score) in score_store.items():
        if name!="DummyClassifier":
            precision,recall,_=precision_recall_curve(y_test,score); ax.plot(recall,precision,label=f"{name} (AP={average_precision_score(y_test,score):.3f})")
    ax.axhline(np.mean(y_test),color="k",ls="--",label="Positive prevalence"); ax.set(title="Precision-recall curves on held-out test set",xlabel="Recall (phishing)",ylabel="Precision (phishing)"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"precision_recall_curve.png",dpi=200); plt.close(fig)
    plot=metrics.set_index("model")[["accuracy","precision_phishing","recall_phishing","f1_phishing","roc_auc"]]; ax=plot.plot(kind="bar",figsize=(11,5),ylim=(0,1.05)); ax.set(title="Held-out model comparison",ylabel="Score",xlabel="Model"); ax.tick_params(axis="x",rotation=20); ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(FIGURES/"model_comparison.png",dpi=200); plt.close()
    # Select by phishing F1, then recall; all metrics come from the same untouched test split.
    eligible=metrics[metrics.model!="DummyClassifier"].sort_values(["f1_phishing","recall_phishing","roc_auc"],ascending=False)
    best_name=eligible.iloc[0].model; best=models[best_name]
    joblib.dump(best,PROJECT_ROOT/"models"/"best_model.pkl")
    return metrics,best_name

def save_importance(models,feature_names):
    model=models["Random Forest"]
    imp=pd.Series(model.feature_importances_,index=feature_names,name="importance").sort_values(ascending=False)
    imp.rename_axis("feature").to_csv(METRICS/"feature_importance.csv")
    fig,ax=plt.subplots(figsize=(9,8)); imp.head(20).sort_values().plot(kind="barh",ax=ax,color="#2878B5"); ax.set(title="Random Forest feature importance (top 20)",xlabel="Mean decrease in impurity",ylabel="Feature"); fig.tight_layout(); fig.savefig(FIGURES/"feature_importance.png",dpi=200); plt.close(fig)
    return imp

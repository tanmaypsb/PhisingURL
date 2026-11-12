import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .utils import FIGURES, SUMMARIES
from .data_cleaning import TARGET

sns.set_theme(style="whitegrid", context="notebook")

def run_eda(data, missing):
    FIGURES.mkdir(parents=True, exist_ok=True)
    stats=data.drop(columns=TARGET).describe().T
    stats["median"]=data.drop(columns=TARGET).median()
    stats["mode"]=data.drop(columns=TARGET).mode().iloc[0]
    stats["variance"]=data.drop(columns=TARGET).var()
    stats["iqr"]=stats["75%"]-stats["25%"]
    stats.to_csv(SUMMARIES/"descriptive_statistics.csv")
    data[TARGET].value_counts().rename(index={0:"Legitimate",1:"Phishing"}).rename_axis("class").to_csv(SUMMARIES/"class_distribution.csv")
    fig,ax=plt.subplots(figsize=(6,4)); sns.countplot(data=data,x=TARGET,hue=TARGET,legend=False,ax=ax,palette={0:"#2878B5",1:"#D9534F"}); ax.set_xticks([0,1], ["Legitimate","Phishing"]); ax.set(title="Website class distribution",xlabel="Website class",ylabel="Number of records"); fig.tight_layout(); fig.savefig(FIGURES/"class_distribution.png",dpi=200); plt.close(fig)
    miss=missing[missing.missing_count>0].sort_values("missing_count",ascending=False)
    fig,ax=plt.subplots(figsize=(8,3.5));
    if len(miss): sns.barplot(x=miss.index,y=miss.missing_count,ax=ax,color="#D99A2B"); ax.tick_params(axis="x",rotation=70)
    else: ax.text(.5,.5,"No missing values in raw dataset",ha="center",va="center",transform=ax.transAxes); ax.set_xticks([])
    ax.set(title="Missing values by feature",xlabel="Feature",ylabel="Missing count"); fig.tight_layout(); fig.savefig(FIGURES/"missing_values.png",dpi=200); plt.close(fig)
    corr=data.drop(columns=TARGET).corr()
    fig,ax=plt.subplots(figsize=(14,11)); sns.heatmap(corr,cmap="vlag",center=0,square=True,linewidths=.15,cbar_kws={"shrink":.7},ax=ax); ax.set_title("Correlation between encoded website indicators"); fig.tight_layout(); fig.savefig(FIGURES/"correlation_heatmap.png",dpi=200); plt.close(fig)
    selected=["URL_Length","having_Sub_Domain","SSLfinal_State","web_traffic"]
    selected=[c for c in selected if c in data]
    fig,axes=plt.subplots(2,2,figsize=(10,7));
    for ax,col in zip(axes.flat,selected): sns.countplot(data=data,x=col,hue=TARGET,ax=ax,palette={0:"#2878B5",1:"#D9534F"}); ax.set(title=f"{col} by class",xlabel="Encoded value",ylabel="Count"); ax.legend(title="is_phishing",labels=["Legitimate","Phishing"])
    for ax in axes.flat[len(selected):]: ax.remove()
    fig.tight_layout(); fig.savefig(FIGURES/"feature_distribution.png",dpi=200); plt.close(fig)
    boxcols=[c for c in ["URL_Length","having_Sub_Domain","SSLfinal_State","web_traffic","Page_Rank","Links_pointing_to_page"] if c in data]
    fig,ax=plt.subplots(figsize=(11,5)); long=data.melt(id_vars=TARGET,value_vars=boxcols,var_name="Feature",value_name="Encoded indicator"); sns.boxplot(data=long,x="Feature",y="Encoded indicator",hue=TARGET,ax=ax,palette={0:"#2878B5",1:"#D9534F"},showfliers=False); ax.set(title="Encoded feature values by website class",xlabel="Feature",ylabel="Encoded value"); ax.legend(title="is_phishing",labels=["Legitimate","Phishing"]); fig.tight_layout(); fig.savefig(FIGURES/"feature_boxplots.png",dpi=200); plt.close(fig)
    return corr

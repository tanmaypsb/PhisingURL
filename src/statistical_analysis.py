"""Two prespecified group tests with family-wise Bonferroni adjustment."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import chi2_contingency
from .data_cleaning import TARGET
from .utils import STATS

def run_statistical_tests(data):
    STATS.mkdir(parents=True, exist_ok=True)
    rows=[]; alpha=.05; tests=2
    # Ordinal URL_Length code comparison; Welch's test avoids equal-variance assumption.
    ph=data.loc[data[TARGET]==1,"URL_Length"].astype(float); leg=data.loc[data[TARGET]==0,"URL_Length"].astype(float)
    res=stats.ttest_ind(ph,leg,equal_var=False)
    diff=float(ph.mean()-leg.mean()); se=float(np.sqrt(ph.var(ddof=1)/len(ph)+leg.var(ddof=1)/len(leg)))
    df=(ph.var(ddof=1)/len(ph)+leg.var(ddof=1)/len(leg))**2/((ph.var(ddof=1)/len(ph))**2/(len(ph)-1)+(leg.var(ddof=1)/len(leg))**2/(len(leg)-1))
    crit=stats.t.ppf(.975,df); pooled=np.sqrt(((len(ph)-1)*ph.var(ddof=1)+(len(leg)-1)*leg.var(ddof=1))/(len(ph)+len(leg)-2)); d=diff/pooled
    p_adj=min(float(res.pvalue)*tests,1.0)
    rows.append({"test":"Welch independent samples t-test","feature":"URL_Length","null_hypothesis":"Mean encoded URL_Length is equal between phishing and legitimate websites","alternative_hypothesis":"Mean encoded URL_Length differs between the classes","statistic":float(res.statistic),"degrees_of_freedom":float(df),"p_value":float(res.pvalue),"adjusted_p_value_bonferroni":p_adj,"alpha_familywise":alpha,"effect_size":"Cohen's d","effect_size_value":float(d),"mean_difference_phishing_minus_legitimate":diff,"ci95_difference_low":diff-crit*se,"ci95_difference_high":diff+crit*se,"reject_null":p_adj<alpha,"interpretation":("Evidence of a class difference" if p_adj<alpha else "Insufficient evidence of a class difference")+" in encoded URL_Length; association does not imply causation."})
    # Categorical association of the signed prefix/suffix indicator with class.
    tab=pd.crosstab(data["Prefix_Suffix"],data[TARGET]); chi2,p,dof,expected=chi2_contingency(tab,correction=False)
    n=tab.to_numpy().sum(); v=float(np.sqrt(chi2/(n*min(tab.shape[0]-1,tab.shape[1]-1))))
    p_adj=min(float(p)*tests,1.0)
    rows.append({"test":"Pearson chi-square test of independence","feature":"Prefix_Suffix","null_hypothesis":"Prefix_Suffix and phishing class are independent","alternative_hypothesis":"Prefix_Suffix and phishing class are associated","statistic":float(chi2),"degrees_of_freedom":int(dof),"p_value":float(p),"adjusted_p_value_bonferroni":p_adj,"alpha_familywise":alpha,"effect_size":"Cramer's V","effect_size_value":v,"mean_difference_phishing_minus_legitimate":np.nan,"ci95_difference_low":np.nan,"ci95_difference_high":np.nan,"reject_null":p_adj<alpha,"interpretation":("Evidence of association" if p_adj<alpha else "Insufficient evidence of association")+" between Prefix_Suffix and phishing class; association does not imply causation."})
    out=pd.DataFrame(rows); out.to_csv(STATS/"hypothesis_test_results.csv",index=False)
    with (STATS/"hypothesis_interpretations.txt").open("w",encoding="utf-8") as f:
        f.write("Statistical inference (family-wise alpha = 0.05; Bonferroni correction for 2 tests)\n\n")
        f.write("Welch t-test compares mean encoded URL_Length values without assuming equal variances; reported 95% CI is for phishing minus legitimate mean.\n")
        f.write("Chi-square tests independence between the categorical Prefix_Suffix feature and target; Cramer's V measures association.\n")
        for r in rows: f.write(f"\n{r['test']} — {r['feature']}: statistic={r['statistic']:.4g}, p={r['p_value']:.4g}, Bonferroni p={r['adjusted_p_value_bonferroni']:.4g}, effect={r['effect_size']} {r['effect_size_value']:.4g}. {r['interpretation']}\n")
        f.write("These tests are descriptive inferential analyses of the available benchmark sample; avoid causal claims.\n")
    return out

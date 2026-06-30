"""
Hypothesis tests for the churn drivers.

EDA charts tell you two groups *look* different. They don't tell you whether the
gap is real or whether it's big enough to care about. This module answers both:

  - categorical features  -> chi-square test of independence + Cramer's V
  - numeric features      -> Welch's t-test + Cohen's d

p-value = "is the effect real?", effect size = "is it large enough to act on?".
Reporting only one of the two is how people end up chasing significant-but-tiny
differences in large samples.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Conventional thresholds. Cramer's V follows Cohen's guidance (adjusted for the
# table's degrees of freedom); Cohen's d uses the usual 0.2/0.5/0.8 bands.
SIG_ALPHA = 0.05


def cramers_v(confusion: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V for a contingency table."""
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    # Bergsma's bias correction - matters on small/sparse tables
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def _v_label(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with a pooled standard deviation."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var == 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def _d_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def chi_square_test(df: pd.DataFrame, feature: str, target: str = "Churn") -> dict:
    """Chi-square test of independence between a categorical feature and churn."""
    table = pd.crosstab(df[feature], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = cramers_v(table)
    return {
        "feature": feature,
        "test": "chi-square",
        "statistic": round(chi2, 2),
        "dof": int(dof),
        "p_value": p,
        "effect_size": round(v, 3),
        "effect_metric": "Cramer's V",
        "effect_magnitude": _v_label(v),
        "significant": p < SIG_ALPHA,
    }


def t_test(df: pd.DataFrame, feature: str, target: str = "Churn") -> dict:
    """Welch's t-test comparing a numeric feature across churned vs retained."""
    churned = df.loc[df[target] == "Yes", feature].dropna().to_numpy()
    retained = df.loc[df[target] == "No", feature].dropna().to_numpy()
    # Welch's, not Student's - the two groups have unequal sizes and variances.
    t, p = stats.ttest_ind(churned, retained, equal_var=False)
    d = cohens_d(churned, retained)
    return {
        "feature": feature,
        "test": "Welch t-test",
        "statistic": round(t, 2),
        "mean_churned": round(churned.mean(), 2),
        "mean_retained": round(retained.mean(), 2),
        "p_value": p,
        "effect_size": round(d, 3),
        "effect_metric": "Cohen's d",
        "effect_magnitude": _d_label(d),
        "significant": p < SIG_ALPHA,
    }


def run_all_tests(df: pd.DataFrame,
                  categorical=("Contract", "PaymentMethod", "InternetService"),
                  numeric=("tenure", "MonthlyCharges"),
                  target: str = "Churn") -> pd.DataFrame:
    """Run every test and return a tidy, sorted results table."""
    rows = [chi_square_test(df, c, target) for c in categorical]
    rows += [t_test(df, n, target) for n in numeric]
    out = pd.DataFrame(rows)
    # Strongest effects first - that's the order you actually care about.
    return out.sort_values("effect_size", ascending=False, key=abs).reset_index(drop=True)


def format_p(p: float) -> str:
    """Human-friendly p-value (scipy underflows tiny p-values to 0.0)."""
    if p == 0.0 or p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_report(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Print a readable significance report and return the results frame."""
    results = run_all_tests(df, **kwargs)
    print("Which churn drivers are statistically significant?\n")
    for _, r in results.iterrows():
        verdict = "significant" if r["significant"] else "not significant"
        print(
            f"{r['feature']:<16} {r['test']:<13} "
            f"p = {format_p(r['p_value']):>10}  "
            f"{r['effect_metric']} = {r['effect_size']:>6} ({r['effect_magnitude']})  "
            f"-> {verdict}"
        )
    return results


if __name__ == "__main__":
    from .data_loader import load_from_sqlite
    print_report(load_from_sqlite())

"""
Turning churn probabilities into dollars.

A ranking model is only useful if someone can act on it with a budget. This
module answers the question a retention manager actually asks: "if I can only
call a fraction of customers, who do I call, what does it cost, and what do I get
back?"

Two pieces:
  1. A targeting analysis - rank by calibrated risk, sweep the cutoff decile by
     decile, and compute cost / revenue retained / ROI at each depth.
  2. A simulated A/B test - the honest way to estimate uplift. We can't just
     assume the intervention works; we model a control vs treated group, draw
     outcomes, and run a significance test on the difference.

All the money assumptions live in one place (CampaignAssumptions) so they're
easy to challenge - they're estimates, not gospel.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED
from .data_loader import load_from_sqlite
from .model import build_pipeline, prepare_xy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


@dataclass
class CampaignAssumptions:
    intervention_cost: float = 50.0    # $ to contact + offer per targeted customer
    uplift: float = 0.30               # fraction of would-be churners actually saved
    horizon_months: int = 12           # window over which we count retained revenue


def build_scored_test(df=None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Train + calibrate, then score a held-out test set.

    Returns one row per test customer with their monthly charge, calibrated churn
    probability, and (for the simulation) the true churn label.
    """
    if df is None:
        df = load_from_sqlite()
    X, y = prepare_xy(df)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    base = build_pipeline(X_tr, scale_pos_weight=pos_weight).fit(X_tr, y_tr)
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(X_cal, y_cal)

    scored = pd.DataFrame({
        "monthly_charges": X_te["MonthlyCharges"].to_numpy(),
        "churn_prob": model.predict_proba(X_te)[:, 1],
        "actual_churn": y_te.to_numpy(),
    })
    return scored.sort_values("churn_prob", ascending=False).reset_index(drop=True)


def decile_analysis(scored: pd.DataFrame, a: CampaignAssumptions = None) -> pd.DataFrame:
    """
    Walk down the risk ranking in 10% steps. At each depth, assume we contact
    everyone above the cutoff and report the economics.

    Revenue retained = (expected churners in group) x uplift x value_per_customer.
    We use calibrated probabilities for the expectation, which is exactly why
    calibration matters here - a miscalibrated score makes this number fiction.
    """
    a = a or CampaignAssumptions()
    scored = scored.copy()
    scored["value"] = scored["monthly_charges"] * a.horizon_months
    n = len(scored)

    rows = []
    for d in range(1, 11):
        k = int(round(n * d / 10))
        grp = scored.iloc[:k]
        expected_churners = grp["churn_prob"].sum()
        saved = expected_churners * a.uplift
        revenue_retained = float((grp["churn_prob"] * a.uplift * grp["value"]).sum())
        cost = k * a.intervention_cost
        net = revenue_retained - cost
        rows.append({
            "decile": d,
            "customers_targeted": k,
            "expected_churners": round(expected_churners, 1),
            "expected_saved": round(saved, 1),
            "intervention_cost": round(cost, 0),
            "revenue_retained": round(revenue_retained, 0),
            "net_benefit": round(net, 0),
            "roi": round(net / cost, 2) if cost else 0.0,
        })
    return pd.DataFrame(rows)


def simulate_ab_test(scored: pd.DataFrame, target_decile: float = 0.1,
                     a: CampaignAssumptions = None, seed: int = RANDOM_SEED) -> dict:
    """
    Simulate a randomized retention experiment on the highest-risk customers.

    Design: take the top `target_decile` by risk, randomly split 50/50 into
    control and treatment. Treatment receives an intervention that removes each
    customer's churn with probability `uplift`. We draw the actual outcomes from
    the calibrated probabilities (control churns at its true rate; treatment at
    the reduced rate), then run a two-proportion z-test on the churn difference.

    This is the part that separates "we think it helps" from "we measured that it
    helps and here's the p-value".
    """
    a = a or CampaignAssumptions()
    rng = np.random.default_rng(seed)

    k = int(round(len(scored) * target_decile))
    cohort = scored.iloc[:k].copy()

    assign = rng.random(k) < 0.5  # True -> treatment
    p = cohort["churn_prob"].to_numpy()

    # Control: churn ~ Bernoulli(p). Treatment: intervention saves a would-be
    # churner w.p. uplift, so effective churn prob = p * (1 - uplift).
    base_draw = rng.random(k) < p
    saved_draw = rng.random(k) < a.uplift
    churn = np.where(assign, base_draw & ~saved_draw, base_draw)

    ctrl = ~assign
    trt = assign
    ctrl_rate = churn[ctrl].mean()
    trt_rate = churn[trt].mean()

    # Two-proportion z-test (pooled).
    x1, n1 = churn[ctrl].sum(), ctrl.sum()
    x2, n2 = churn[trt].sum(), trt.sum()
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (ctrl_rate - trt_rate) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    value = cohort["monthly_charges"].to_numpy() * a.horizon_months
    customers_saved = int(churn[ctrl].sum() / n1 * n2 - churn[trt].sum())  # vs control rate
    revenue_retained = float((trt.sum() * (ctrl_rate - trt_rate)) * value.mean())
    cost = trt.sum() * a.intervention_cost

    return {
        "cohort_size": k,
        "control_n": int(n1),
        "treatment_n": int(n2),
        "control_churn_rate": round(float(ctrl_rate), 4),
        "treatment_churn_rate": round(float(trt_rate), 4),
        "absolute_reduction": round(float(ctrl_rate - trt_rate), 4),
        "relative_reduction": round(float((ctrl_rate - trt_rate) / ctrl_rate), 4) if ctrl_rate else 0.0,
        "z_stat": round(float(z), 2),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "est_customers_saved_vs_control": max(customers_saved, 0),
        "treatment_revenue_retained": round(revenue_retained, 0),
        "treatment_cost": round(float(cost), 0),
    }


def main():
    a = CampaignAssumptions()
    scored = build_scored_test()
    print(f"Assumptions: ${a.intervention_cost:.0f}/contact, "
          f"{a.uplift:.0%} uplift, {a.horizon_months}-month horizon\n")

    table = decile_analysis(scored, a)
    print("Targeting economics by risk depth")
    print(table.to_string(index=False))

    best = table.loc[table["net_benefit"].idxmax()]
    print(f"\nBest net benefit at decile {int(best['decile'])}: "
          f"${best['net_benefit']:,.0f} from {int(best['customers_targeted'])} "
          f"customers (ROI {best['roi']:.2f}x)")

    print("\nSimulated A/B test on the top-risk decile")
    print("-" * 40)
    ab = simulate_ab_test(scored, target_decile=0.1, a=a)
    for k, v in ab.items():
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()

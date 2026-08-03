"""
Probability calibration.

A model can rank customers well (good AUC) and still be a liar about
probabilities - if it says "80% chance of churn" for a group, you want roughly
80% of them to actually churn. That property is what makes a risk score usable
for budgeting ("expected revenue at risk = sum of probabilities x value"), so it
matters more here than another decimal point of AUC.

We measure it with a reliability curve and Expected Calibration Error (ECE), and
fix it with either isotonic regression or Platt scaling (sigmoid), fit on a
held-out slice so we're not grading our own homework.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, RANDOM_SEED, COLORS
from .model import build_pipeline, prepare_xy, train as train_base
from .data_loader import load_from_sqlite


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    ECE: average gap between predicted confidence and actual frequency,
    weighted by how many points land in each bin. 0 = perfectly calibrated.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def calibrate(method: str = "isotonic", df=None):
    """
    Fit the base model, then wrap it in a calibrator fit on a separate slice.

    Returns the calibrated classifier plus before/after metrics so you can see
    whether calibration actually bought you anything.
    """
    if df is None:
        df = load_from_sqlite()
    X, y = prepare_xy(df)

    # Three-way split: train base model, fit calibrator, evaluate. No leakage.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_SEED)

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    base = build_pipeline(X_tr, scale_pos_weight=pos_weight).fit(X_tr, y_tr)

    # Base model is already fit; FrozenEstimator stops sklearn re-fitting it so
    # the calibrator only learns the probability mapping (sklearn >=1.6 API).
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X_cal, y_cal)

    raw_prob = base.predict_proba(X_te)[:, 1]
    cal_prob = calibrated.predict_proba(X_te)[:, 1]

    report = {
        "method": method,
        "ece_before": round(expected_calibration_error(y_te.to_numpy(), raw_prob), 4),
        "ece_after": round(expected_calibration_error(y_te.to_numpy(), cal_prob), 4),
        "brier_before": round(brier_score_loss(y_te, raw_prob), 4),
        "brier_after": round(brier_score_loss(y_te, cal_prob), 4),
        # AUC/PR-AUC should barely move - calibration reshapes probabilities,
        # it doesn't reorder them.
        "roc_auc": round(roc_auc_score(y_te, cal_prob), 4),
        "pr_auc": round(average_precision_score(y_te, cal_prob), 4),
    }
    return calibrated, (y_te.to_numpy(), raw_prob, cal_prob), report


def plot_reliability(y_true, raw_prob, cal_prob, report, save_path=None):
    """Reliability diagram: raw vs calibrated against the diagonal (perfect)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")

    for prob, color, name, ece in [
        (raw_prob, COLORS["danger"], "raw", report["ece_before"]),
        (cal_prob, COLORS["success"], f"{report['method']}", report["ece_after"]),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=color, label=f"{name} (ECE={ece})")

    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn frequency")
    ax.set_title("Reliability curve")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "reliability_curve.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def fit_for_serving(df=None, method: str = "isotonic"):
    """
    Fit a calibrated model for the app to score live customers with.

    Unlike calibrate(), this keeps no test hold-out - it spends all the data on
    training + calibration because here we want the best possible scorer, not an
    unbiased performance estimate. Returns (calibrated_model, base_pipeline); the
    base pipeline is handed to explain.py for SHAP.
    """
    if df is None:
        df = load_from_sqlite()
    X, y = prepare_xy(df)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    base = build_pipeline(X_tr, scale_pos_weight=pos_weight).fit(X_tr, y_tr)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X_cal, y_cal)
    return calibrated, base


def main():
    calibrated, (y_te, raw, cal), report = calibrate("isotonic")
    print("Calibration report (isotonic)")
    print("-" * 40)
    for k, v in report.items():
        print(f"  {k:<13} {v}")
    plot_reliability(y_te, raw, cal, report)
    print(f"\nSaved reliability curve -> {FIGURES_DIR / 'reliability_curve.png'}")


if __name__ == "__main__":
    main()

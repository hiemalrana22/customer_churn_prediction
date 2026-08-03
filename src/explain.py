"""
SHAP explanations - global and per-customer.

"The model says 82%" is useless to a retention agent on a call. "82%, driven by
month-to-month contract and 2 months tenure" is something they can act on. SHAP
gives us both views from the same trained model:

  - global: which features move churn predictions the most, across everyone
  - local: for one customer, which of their attributes pushed risk up or down

The model is a Pipeline (preprocess -> XGBoost), so we explain on the transformed
matrix and carry the one-hot feature names through, then fold them back into
readable labels.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import FIGURES_DIR, COLORS
from .model import load as load_model, train, save, prepare_xy
from .data_loader import load_from_sqlite


def _unpack(pipeline):
    """Return (preprocessor, classifier, feature_names) from a fitted pipeline."""
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    return pre, clf, names


def _prettify(name: str) -> str:
    """num__tenure -> tenure ; cat__Contract_Two year -> Contract = Two year."""
    name = name.split("__", 1)[-1]
    for col in ("Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
                "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV",
                "StreamingMovies", "MultipleLines", "PaperlessBilling", "Partner",
                "Dependents", "SeniorCitizen", "gender", "PhoneService"):
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def shap_values(pipeline, X: pd.DataFrame) -> tuple:
    """
    Exact TreeSHAP values via XGBoost's native `pred_contribs`.

    This is the same TreeSHAP algorithm the `shap` package uses, computed
    straight from the booster - which sidesteps the shap/XGBoost version
    mismatch and is faster. Values are in log-odds space; the final column
    pred_contribs returns is the bias term, which we drop.
    """
    pre, clf, names = _unpack(pipeline)
    Xt = pre.transform(X)
    booster = clf.get_booster()
    dm = xgb.DMatrix(Xt, feature_names=names)
    contribs = booster.predict(dm, pred_contribs=True)
    return contribs[:, :-1], names


def global_importance(pipeline, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Mean |SHAP| per feature, aggregated back to original columns."""
    sv, names = shap_values(pipeline, X)

    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": [_prettify(n) for n in names], "mean_abs_shap": mean_abs})
    # Collapse one-hot columns (Contract = X, Contract = Y, ...) onto the base field.
    imp["group"] = imp["feature"].str.split(" = ").str[0]
    grouped = (imp.groupby("group")["mean_abs_shap"].sum()
               .sort_values(ascending=False).head(top_n).reset_index())
    grouped.columns = ["feature", "mean_abs_shap"]
    return grouped


def plot_global_importance(pipeline, X, top_n: int = 12, save_path=None):
    grouped = global_importance(pipeline, X, top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(grouped["feature"][::-1], grouped["mean_abs_shap"][::-1],
            color=COLORS["primary"], edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Mean |SHAP value| (impact on churn log-odds)")
    ax.set_title("Global churn drivers")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    if save_path is None:
        save_path = FIGURES_DIR / "shap_global_importance.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def explain_customer(pipeline, x_row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    Per-customer breakdown. Positive shap = pushes churn up, negative = pulls down.
    Returns the top_n features by absolute contribution.
    """
    sv_all, names = shap_values(pipeline, x_row)
    sv = sv_all[0]

    out = pd.DataFrame({
        "feature": [_prettify(n) for n in names],
        "shap_value": sv,
    })
    out["direction"] = np.where(out["shap_value"] >= 0, "increases risk", "lowers risk")
    out["abs"] = out["shap_value"].abs()
    out = out[out["abs"] > 1e-6].sort_values("abs", ascending=False).head(top_n)
    return out.drop(columns="abs").reset_index(drop=True)


def main():
    df = load_from_sqlite()
    try:
        bundle = load_model()
        pipeline = bundle["pipeline"]
    except FileNotFoundError:
        res = train(df)
        save(res)
        pipeline = res.pipeline

    X, _ = prepare_xy(df)

    print("Top global churn drivers (mean |SHAP|)")
    print("-" * 40)
    print(global_importance(pipeline, X.sample(min(1000, len(X)), random_state=0)).to_string(index=False))
    plot_global_importance(pipeline, X.sample(min(1000, len(X)), random_state=0))
    print(f"\nSaved -> {FIGURES_DIR / 'shap_global_importance.png'}")

    print("\nExample per-customer explanation")
    print("-" * 40)
    print(explain_customer(pipeline, X.iloc[[0]]).to_string(index=False))


if __name__ == "__main__":
    main()

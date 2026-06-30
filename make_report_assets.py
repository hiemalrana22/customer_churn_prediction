"""
Regenerate the figures used in the README.

Keeps the charts in the README reproducible instead of hand-pasted - run this
after retraining and the docs stay honest. Writes into docs/images/.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from src.config import COLORS
from src.data_loader import load_from_sqlite
from src.model import prepare_xy, train, save
from src.calibrate import calibrate, plot_reliability
from src.explain import plot_global_importance
from src.business_impact import build_scored_test, decile_analysis, CampaignAssumptions

DOCS = Path(__file__).parent / "docs" / "images"
DOCS.mkdir(parents=True, exist_ok=True)


def main():
    df = load_from_sqlite()

    # Train + persist the model so the rest of the project can load it.
    res = train(df)
    save(res)
    print("model metrics:", res.metrics)

    # 1. SHAP global importance
    X, _ = prepare_xy(df)
    plot_global_importance(res.pipeline, X.sample(1500, random_state=0),
                           save_path=DOCS / "shap_global_importance.png")

    # 2. Reliability curve (calibration)
    _, (y_te, raw, cal), report = calibrate("isotonic", df)
    plot_reliability(y_te, raw, cal, report, save_path=DOCS / "reliability_curve.png")
    print("calibration:", report)

    # 3. ROI by targeting depth
    a = CampaignAssumptions()
    table = decile_analysis(build_scored_test(df), a)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(table["decile"], table["net_benefit"] / 1000,
            color=COLORS["primary"], alpha=0.85, edgecolor="black", linewidth=0.6,
            label="Net benefit ($k)")
    ax1.set_xlabel("Customers targeted (risk decile, deepest = all)")
    ax1.set_ylabel("Net benefit ($ thousands)")
    ax2 = ax1.twinx()
    ax2.plot(table["decile"], table["roi"], "o-", color=COLORS["danger"], label="ROI")
    ax2.set_ylabel("ROI (x)")
    ax1.set_title("Retention campaign economics by targeting depth")
    ax1.set_xticks(table["decile"])
    fig.tight_layout()
    fig.savefig(DOCS / "roi_by_decile.png", dpi=200, bbox_inches="tight")
    best = table.loc[table["net_benefit"].idxmax()]
    print(f"best net benefit: decile {int(best['decile'])} "
          f"${best['net_benefit']:,.0f} (ROI {best['roi']}x)")

    print(f"\nWrote figures to {DOCS}")


if __name__ == "__main__":
    main()

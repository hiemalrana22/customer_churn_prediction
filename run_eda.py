#!/usr/bin/env python3
"""
Run the full EDA from the command line and dump all figures to outputs/figures.

Same analysis as the notebook, just scripted so I can regenerate everything in
one shot after a data refresh. For the modelling/stats/business pieces use the
modules in src/ (or `python -m src.model`, `python -m src.stats_tests`, etc.).

    python run_eda.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import set_plot_style, RANDOM_SEED, FIGURES_DIR
from src.data_loader import (
    load_from_sqlite,
    check_data_quality,
    get_numerical_summary,
    split_features_by_type,
    calculate_churn_rate,
)
from src.stats_tests import print_report
from src.visualizations import (
    plot_churn_rate,
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_churn_rate_by_feature,
    plot_boxplots_by_churn,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)
np.random.seed(RANDOM_SEED)
set_plot_style()


def main():
    print("Telco churn EDA")
    print(f"figures -> {FIGURES_DIR}\n")

    # Pulled through SQLite rather than the raw CSV (see data_loader).
    df = load_from_sqlite()
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    check_data_quality(df, verbose=True)

    numerical, categorical, target = split_features_by_type(df)
    print(f"\nnumeric:     {', '.join(numerical)}")
    print(f"categorical: {len(categorical)} columns")
    print(f"target:      {target}")

    m = calculate_churn_rate(df)
    print(f"\nChurn rate: {m['churn_rate_pct']:.2f}% "
          f"({m['churned_customers']:,} of {m['total_customers']:,})")

    plot_churn_rate(df, show=False)
    plot_numerical_distributions(df, numerical, show=False)
    plot_boxplots_by_churn(df, numerical, show=False)
    plot_correlation_heatmap(df, numerical, show=False)
    plot_categorical_distributions(df, categorical, show=False)
    for feat in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
        plot_churn_rate_by_feature(df, feat, show=False)

    print("\nNumerical summary:")
    print(get_numerical_summary(df, numerical))

    # The hypothesis tests live in src/stats_tests so the notebook and the app
    # use the exact same numbers.
    print()
    print_report(df)

    print(f"\nDone. Figures saved to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

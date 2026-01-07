#!/usr/bin/env python3
"""
Production-grade EDA runner for Telco Customer Churn dataset.

This script performs comprehensive exploratory data analysis:
- Data quality assessment
- Churn rate analysis
- Distribution plots for numerical features
- Count plots for categorical features
- Statistical significance testing
- Correlation analysis

Usage:
    python run_eda.py
"""

import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# Add project root to path for package imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from src package
from src.config import set_plot_style, RANDOM_SEED, FIGURES_DIR
from src.data_loader import (
    load_telco_data,
    get_data_info,
    check_data_quality,
    get_numerical_summary,
    get_categorical_summary,
    split_features_by_type,
    calculate_churn_rate
)
from src.visualizations import (
    plot_churn_rate,
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_churn_rate_by_feature,
    plot_boxplots_by_churn
)

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)
np.random.seed(RANDOM_SEED)
set_plot_style()


def main():
    """Main execution function for EDA."""
    print("=" * 80)
    print("TELCO CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Output directory: {FIGURES_DIR}\n")
    
    # 1. Load data
    print("📥 Loading dataset...")
    try:
        df = load_telco_data()
        print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease download the dataset using:")
        print("  ./download_dataset.sh")
        print("or manually from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return 1
    
    # 2. Data quality assessment
    print("=" * 80)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 80)
    quality_report = check_data_quality(df, verbose=True)
    
    # 3. Feature type identification
    print("\n" + "=" * 80)
    print("FEATURE CATEGORIZATION")
    print("=" * 80)
    numerical_features, categorical_features, target = split_features_by_type(df)
    print(f"\n📊 Numerical Features ({len(numerical_features)}): {', '.join(numerical_features)}")
    print(f"📋 Categorical Features ({len(categorical_features)}): {', '.join(categorical_features)}")
    print(f"🎯 Target Variable: {target}")
    
    # 4. Churn rate analysis
    print("\n" + "=" * 80)
    print("CHURN RATE ANALYSIS")
    print("=" * 80)
    churn_metrics = calculate_churn_rate(df)
    print(f"Total Customers: {churn_metrics['total_customers']:,}")
    print(f"Churned Customers: {churn_metrics['churned_customers']:,}")
    print(f"Retained Customers: {churn_metrics['retained_customers']:,}")
    print(f"📈 Churn Rate: {churn_metrics['churn_rate_pct']:.2f}%")
    print(f"📉 Retention Rate: {churn_metrics['retention_rate_pct']:.2f}%")
    
    print("\n📊 Generating churn rate visualization...")
    plot_churn_rate(df, show=False)
    print("✅ Saved: churn_rate_overview.png")
    
    # 5. Numerical features analysis
    print("\n" + "=" * 80)
    print("NUMERICAL FEATURES ANALYSIS")
    print("=" * 80)
    numerical_summary = get_numerical_summary(df, numerical_features)
    print("\nStatistical Summary:")
    print(numerical_summary)
    
    print("\n📊 Generating distribution plots...")
    plot_numerical_distributions(df, numerical_features, show=False)
    print("✅ Saved: numerical_distributions.png")
    
    print("\n📊 Generating box plots...")
    plot_boxplots_by_churn(df, numerical_features, show=False)
    print("✅ Saved: boxplots_by_churn.png")
    
    # Statistical tests for numerical features
    print("\nStatistical Significance Tests (T-test):")
    print("-" * 80)
    for feature in numerical_features:
        churn_yes = df[df['Churn'] == 'Yes'][feature].dropna()
        churn_no = df[df['Churn'] == 'No'][feature].dropna()
        t_stat, p_value = stats.ttest_ind(churn_yes, churn_no)
        significance = "✓ Significant" if p_value < 0.05 else "✗ Not Significant"
        print(f"{feature:20s} | p-value: {p_value:.4f} | {significance}")
    
    # 6. Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    print("\n📊 Generating correlation heatmap...")
    plot_correlation_heatmap(df, numerical_features, show=False)
    print("✅ Saved: correlation_heatmap.png")
    
    corr_matrix = df[numerical_features].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # 7. Categorical features analysis
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("=" * 80)
    categorical_summaries = get_categorical_summary(df, categorical_features)
    
    print("\n📊 Generating categorical distribution plots...")
    plot_categorical_distributions(df, categorical_features, show=False)
    print("✅ Saved: categorical_distributions.png")
    
    # 8. Churn rate by categorical features
    print("\n" + "=" * 80)
    print("CHURN RATE BY CATEGORICAL FEATURES")
    print("=" * 80)
    
    # Statistical tests for categorical features
    print("\nStatistical Significance Tests (Chi-square):")
    print("-" * 80)
    key_features = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']
    
    for feature in categorical_features:
        churn_rate = df.groupby(feature)['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        contingency_table = pd.crosstab(df[feature], df['Churn'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        significance = "✓ Significant" if p_value < 0.05 else "✗ Not Significant"
        
        print(f"\n{feature}:")
        for category, rate in churn_rate.items():
            count = len(df[df[feature] == category])
            print(f"  {category:30s} | Churn Rate: {rate:5.2f}% | n={count:,}")
        print(f"  Chi-square p-value: {p_value:.4f} | {significance}")
        
        # Generate visualization for key features
        if feature in key_features:
            plot_churn_rate_by_feature(df, feature, show=False)
            print(f"  ✅ Saved: churn_rate_by_{feature.lower()}.png")
    
    # 9. Summary
    print("\n" + "=" * 80)
    print("EDA COMPLETED SUCCESSFULLY! ✓")
    print("=" * 80)
    print(f"\n📁 All visualizations saved to: {FIGURES_DIR}")
    print("\n🎯 Key Findings:")
    print("  • Overall churn rate: {:.2f}%".format(churn_metrics['churn_rate_pct']))
    print("  • Dataset ready for feature engineering and modeling")
    print("  • Statistical tests identify significant predictors")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)


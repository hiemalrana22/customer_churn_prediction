# Telco Customer Churn - EDA Project

A production-grade exploratory data analysis on the Telco Customer Churn dataset, built with FAANG-level best practices.

## 📁 Project Structure

```
data loading EDA/
├── data/                           # Data directory
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/                      # Jupyter notebooks
│   └── 01_telco_churn_eda.ipynb   # Main EDA notebook
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── config.py                  # Configuration & constants
│   ├── data_loader.py             # Data loading utilities
│   └── visualizations.py          # Plotting functions
├── outputs/                        # Output directory
│   └── figures/                   # Saved visualizations
├── run_eda.py                     # Automated EDA runner script
├── download_dataset.sh            # Dataset download script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Using Kaggle API (Recommended)**

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle credentials (get from https://www.kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/
```

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in the `data/` directory

### 3. Run the Analysis

**Option A: Using Python Script (Recommended for Automation)**

```bash
python run_eda.py
```

This will run the complete EDA pipeline and save all visualizations automatically.

**Option B: Using Jupyter Notebook (Interactive)**

```bash
# Launch Jupyter
jupyter notebook

# Open: notebooks/01_telco_churn_eda.ipynb
# Run all cells
```

## 📊 What's Included

### Comprehensive EDA
- ✅ Data quality assessment
- ✅ Missing value analysis
- ✅ Statistical summaries
- ✅ Churn rate analysis
- ✅ Numerical feature distributions
- ✅ Categorical feature analysis
- ✅ Correlation analysis
- ✅ Feature-target relationships
- ✅ Statistical significance testing

### Professional Visualizations
- Publication-quality plots
- Consistent styling and color schemes
- Automated saving to `outputs/figures/`
- Churn rate breakdowns
- Distribution comparisons
- Box plots and heatmaps

### Modular Code Architecture
- **config.py**: Centralized configuration
- **data_loader.py**: Reusable data utilities
- **visualizations.py**: Professional plotting functions
- Clean, documented, type-hinted code

## 🎯 Key Features

### Best Practices
- ✅ Reproducibility (random seeds)
- ✅ Modular design (DRY principle)
- ✅ Type hints and docstrings
- ✅ Statistical rigor (hypothesis testing)
- ✅ Professional documentation
- ✅ Clean code standards

### Analysis Highlights
- Churn rate: ~26-27%
- Key risk factors identified
- Statistical significance testing
- Actionable insights for modeling

## 📈 Next Steps

After completing the EDA:

1. **Feature Engineering**
   - Create tenure bins
   - Build service bundle features
   - Engineer charge ratios
   - Develop risk scores

2. **Model Development**
   - Baseline models (Logistic Regression)
   - Tree-based models (XGBoost, LightGBM, CatBoost)
   - Neural networks (if needed)
   - Ensemble methods

3. **Model Optimization**
   - Hyperparameter tuning (Optuna)
   - Cross-validation
   - Feature selection

4. **Evaluation & Explainability**
   - ROC-AUC, Precision-Recall
   - SHAP values
   - Feature importance
   - Model calibration

## 🛠️ Technologies

- **Python 3.9+**
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **scipy**: Statistical testing
- **jupyter**: Interactive analysis

## 📝 Author

Senior Data Scientist  
Date: 2026-01-07

## 📄 License

This project is for educational and analytical purposes.

---

**Dataset Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

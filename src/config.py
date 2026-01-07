"""
Configuration module for Telco Customer Churn EDA.

Contains all project-level constants, paths, and visualization settings
to ensure consistency and reproducibility across the analysis.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

TELCO_DATA_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color palette
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep magenta
    'accent': '#F18F01',       # Vibrant orange
    'success': '#06A77D',      # Teal green
    'danger': '#D62828',       # Strong red
    'churn_yes': '#E63946',    # Red for churned
    'churn_no': '#06A77D',     # Green for retained
}

# Seaborn style
PLOT_STYLE = 'whitegrid'
PLOT_CONTEXT = 'notebook'
PALETTE = 'Set2'

# Figure sizes
FIGSIZE_SMALL = (8, 5)
FIGSIZE_MEDIUM = (10, 6)
FIGSIZE_LARGE = (12, 8)
FIGSIZE_WIDE = (14, 6)

# Font sizes
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# ============================================================================
# DOMAIN KNOWLEDGE
# ============================================================================

# Expected categorical columns
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Expected numerical columns
NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges'
]

# Target variable
TARGET = 'Churn'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_plot_style():
    """Apply consistent plotting style across all visualizations."""
    sns.set_style(PLOT_STYLE)
    sns.set_context(PLOT_CONTEXT)
    sns.set_palette(PALETTE)
    
    plt.rcParams.update({
        'figure.figsize': FIGSIZE_MEDIUM,
        'axes.titlesize': TITLE_FONTSIZE,
        'axes.labelsize': LABEL_FONTSIZE,
        'xtick.labelsize': TICK_FONTSIZE,
        'ytick.labelsize': TICK_FONTSIZE,
        'legend.fontsize': TICK_FONTSIZE,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def get_churn_colors():
    """Return consistent color mapping for churn status."""
    return {
        'Yes': COLORS['churn_yes'],
        'No': COLORS['churn_no']
    }

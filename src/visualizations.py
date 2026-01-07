"""
Visualization utilities for Telco Customer Churn EDA.

This module provides reusable, publication-quality plotting functions
for exploratory data analysis with consistent styling and best practices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Union
import warnings

from .config import (
    FIGSIZE_SMALL, FIGSIZE_MEDIUM, FIGSIZE_LARGE, FIGSIZE_WIDE,
    COLORS, get_churn_colors, FIGURES_DIR
)


def plot_churn_rate(df: pd.DataFrame, 
                    churn_col: str = 'Churn',
                    save_path: Optional[Path] = None,
                    show: bool = True) -> plt.Figure:
    """
    Create a professional visualization of the overall churn rate.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing churn data.
    churn_col : str, default='Churn'
        Name of the churn column.
    save_path : Path, optional
        Path to save the figure. If None, uses default location.
    show : bool, default=True
        Whether to display the plot.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    # Calculate metrics
    churn_counts = df[churn_col].value_counts()
    churn_pcts = df[churn_col].value_counts(normalize=True) * 100
    colors = [get_churn_colors()[val] for val in churn_counts.index]
    
    # Bar plot
    bars = ax1.bar(churn_counts.index, churn_counts.values, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Churn Status', fontweight='bold')
    ax1.set_ylabel('Number of Customers', fontweight='bold')
    ax1.set_title('Customer Churn Distribution', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        churn_counts.values, 
        labels=churn_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=(0.05, 0),
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title('Churn Rate Percentage', fontweight='bold', fontsize=14)
    
    # Add summary text
    churn_rate = churn_pcts.get('Yes', 0)
    fig.suptitle(f'Overall Churn Rate: {churn_rate:.2f}%', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = FIGURES_DIR / 'churn_rate_overview.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_numerical_distributions(df: pd.DataFrame,
                                 numerical_cols: List[str],
                                 churn_col: str = 'Churn',
                                 save_path: Optional[Path] = None,
                                 show: bool = True) -> plt.Figure:
    """
    Create distribution plots for numerical features with churn overlay.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numerical_cols : List[str]
        List of numerical column names to plot.
    churn_col : str, default='Churn'
        Name of the churn column for overlay.
    save_path : Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    churn_colors = get_churn_colors()
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        # Plot distributions for each churn status
        for churn_status in ['No', 'Yes']:
            data = df[df[churn_col] == churn_status][col].dropna()
            ax.hist(data, bins=30, alpha=0.6, 
                   label=f'Churn: {churn_status}',
                   color=churn_colors[churn_status],
                   edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(col, fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'Distribution of {col}', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Numerical Features Distribution by Churn Status', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = FIGURES_DIR / 'numerical_distributions.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_categorical_distributions(df: pd.DataFrame,
                                   categorical_cols: List[str],
                                   churn_col: str = 'Churn',
                                   save_path: Optional[Path] = None,
                                   show: bool = True) -> plt.Figure:
    """
    Create count plots for categorical features with churn breakdown.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_cols : List[str]
        List of categorical column names to plot.
    churn_col : str, default='Churn'
        Name of the churn column.
    save_path : Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    churn_colors = list(get_churn_colors().values())
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        
        # Create count plot
        sns.countplot(data=df, x=col, hue=churn_col, ax=ax,
                     palette=churn_colors, edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel(col, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'{col} Distribution', fontweight='bold')
        ax.legend(title='Churn', title_fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Rotate x-labels if needed
        if df[col].nunique() > 3:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Categorical Features Distribution by Churn Status', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = FIGURES_DIR / 'categorical_distributions.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                             numerical_cols: List[str],
                             save_path: Optional[Path] = None,
                             show: bool = True) -> plt.Figure:
    """
    Create a correlation heatmap for numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numerical_cols : List[str]
        List of numerical columns to include.
    save_path : Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Correlation Heatmap - Numerical Features', 
                fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = FIGURES_DIR / 'correlation_heatmap.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_churn_rate_by_feature(df: pd.DataFrame,
                               feature: str,
                               churn_col: str = 'Churn',
                               save_path: Optional[Path] = None,
                               show: bool = True) -> plt.Figure:
    """
    Plot churn rate breakdown by a specific feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature : str
        Feature to analyze churn rate by.
    churn_col : str, default='Churn'
        Name of the churn column.
    save_path : Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    # Calculate churn rate by feature
    churn_by_feature = df.groupby(feature)[churn_col].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    
    bars = ax.barh(churn_by_feature.index.astype(str), churn_by_feature.values,
                   color=COLORS['primary'], edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels
    for i, (idx, val) in enumerate(churn_by_feature.items()):
        ax.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Churn Rate (%)', fontweight='bold')
    ax.set_ylabel(feature, fontweight='bold')
    ax.set_title(f'Churn Rate by {feature}', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = FIGURES_DIR / f'churn_rate_by_{feature.lower()}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_boxplots_by_churn(df: pd.DataFrame,
                           numerical_cols: List[str],
                           churn_col: str = 'Churn',
                           save_path: Optional[Path] = None,
                           show: bool = True) -> plt.Figure:
    """
    Create box plots for numerical features grouped by churn status.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numerical_cols : List[str]
        List of numerical columns to plot.
    churn_col : str, default='Churn'
        Name of the churn column.
    save_path : Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    n_cols = len(numerical_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    churn_colors = list(get_churn_colors().values())
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        sns.boxplot(data=df, x=churn_col, y=col, ax=ax,
                   palette=churn_colors, linewidth=1.5)
        
        ax.set_xlabel('Churn Status', fontweight='bold')
        ax.set_ylabel(col, fontweight='bold')
        ax.set_title(f'{col} by Churn', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('Numerical Features Distribution by Churn (Box Plots)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = FIGURES_DIR / 'boxplots_by_churn.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

"""
Churn model: gradient-boosted trees with a clean preprocessing pipeline.

Everything the rest of the project needs to talk to a trained model lives here -
the feature split, the sklearn Pipeline, training, evaluation and (de)serialise.
calibrate.py, explain.py, business_impact.py and app.py all build on top of this
so there's exactly one definition of "the model".

Why PR-AUC matters: churn is imbalanced (~27% positives). ROC-AUC can look
flattering on imbalanced data because the huge negative class inflates the true
negative rate. PR-AUC only cares about the positives, which is what a retention
team actually chases, so I report both and lead with PR-AUC.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .config import MODEL_FILE, RANDOM_SEED
from .data_loader import load_from_sqlite

TARGET = "Churn"
DROP_COLS = ["customerID", TARGET]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


@dataclass
class TrainResult:
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metrics: dict
    threshold: float  # decision threshold tuned for F1, not the naive 0.5


def prepare_xy(df: pd.DataFrame):
    """Split a raw dataframe into features X and binary target y (1 = churned)."""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y


def _categorical_cols(X: pd.DataFrame):
    return [c for c in X.columns if c not in NUMERIC_COLS]


def build_pipeline(X: pd.DataFrame, scale_pos_weight: float = 1.0) -> Pipeline:
    """ColumnTransformer + XGBoost. Numeric is imputed; categoricals one-hot."""
    cat_cols = _categorical_cols(X)
    num_cols = [c for c in NUMERIC_COLS if c in X.columns]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _best_f1_threshold(y_true, y_prob) -> float:
    """Pick the probability cutoff that maximises F1 on the given data."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    # thr has one fewer element than prec/rec
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def train(df: pd.DataFrame = None, test_size: float = 0.2) -> TrainResult:
    """Train on a stratified split and report held-out metrics."""
    if df is None:
        df = load_from_sqlite()

    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    # Counter the class imbalance instead of resampling - keeps the data honest.
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pipe = build_pipeline(X_train, scale_pos_weight=pos_weight)
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    thr = _best_f1_threshold(y_test, prob)
    pred = (prob >= thr).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "pr_auc": round(average_precision_score(y_test, prob), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "base_rate": round(float(y_test.mean()), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return TrainResult(pipe, X_train, X_test, y_train, y_test, metrics, thr)


def save(result: TrainResult, path: Path = MODEL_FILE) -> Path:
    """Persist the fitted pipeline + threshold so the app can load it cold."""
    joblib.dump({"pipeline": result.pipeline, "threshold": result.threshold,
                 "metrics": result.metrics}, path)
    return path


def load(path: Path = MODEL_FILE) -> dict:
    return joblib.load(path)


def main():
    res = train()
    print("Held-out performance")
    print("-" * 40)
    for k, v in res.metrics.items():
        print(f"  {k:<10} {v}")
    print(f"  {'threshold':<10} {round(res.threshold, 3)}")
    print()
    pred = (res.pipeline.predict_proba(res.X_test)[:, 1] >= res.threshold).astype(int)
    print(classification_report(res.y_test, pred, target_names=["retained", "churned"]))
    save(res)
    print(f"Saved model -> {MODEL_FILE}")


if __name__ == "__main__":
    main()

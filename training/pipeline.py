import os
import json
import joblib

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import DROP_COLS, OUTPUT_PATH, PARAM_GRIDS, TARGET_COL
from preprocessing.process import run_pipeline


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load_and_split(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    print("\n[1/6] Loading master dataset...")
    master = run_pipeline()

    cols_to_drop = [c for c in master.columns if c in DROP_COLS]
    X = master.drop(columns=cols_to_drop)
    y = master[TARGET_COL]
    X = X.select_dtypes(exclude=["datetime64[ns]", "datetime64"])

    print(f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(
        f"  Class balance — 0: {(y == 0).sum():,}  1: {(y == 1).sum():,}  "
        f"({y.mean():.1%} positive)"
    )

    print("\n[2/6] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save feature defaults so Streamlit can fill unseen fields with sensible values
    num_defaults = X.select_dtypes(include="number").median().to_dict()
    cat_defaults = {c: X[c].mode()[0] for c in X.select_dtypes(include=["object", "category"]).columns}
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_PATH, "readmission_defaults.json"), "w") as f:
        json.dump({**num_defaults, **cat_defaults}, f)

    return X_train, X_test, y_train, y_test, X


# ---------------------------------------------------------------------------
# 2. Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    print("\n[3/6] Building preprocessor...")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ], remainder="drop")


# ---------------------------------------------------------------------------
# 3. Baseline training
# ---------------------------------------------------------------------------

def train_baselines(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, dict]:
    print("\n[4/6] Training baseline models...")

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, scale_pos_weight=pos_weight,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }

    results = {}
    for model_name, model in candidate_models.items():
        print(f"  Training {model_name}...")
        with mlflow.start_run(run_name=f"baseline_{model_name}"):
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_pred_prob = pipe.predict_proba(X_test)[:, 1]
            metrics = _compute_metrics(y_test, y_pred, y_pred_prob)

            mlflow.log_param("model", model_name)
            mlflow.log_param("stage", "baseline")
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, "model")

            results[model_name] = {"pipe": pipe, **metrics}
            print(
                f"    ROC-AUC={metrics['roc_auc']}  F1={metrics['f1']}  "
                f"Recall={metrics['recall']}  Precision={metrics['precision']}"
            )

    return results, candidate_models


# ---------------------------------------------------------------------------
# 4. Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_best_model(
    best_name: str,
    candidate_models: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple:
    print(f"\n[5/6] Tuning {best_name} with RandomizedSearchCV...")

    search = RandomizedSearchCV(
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", clone(candidate_models[best_name])),
        ]),
        param_distributions=PARAM_GRIDS[best_name],
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_cv_auc = round(search.best_score_, 4)
    print(f"  Best CV ROC-AUC : {best_cv_auc}")
    print(f"  Best params     : {search.best_params_}")
    return search.best_estimator_, search.best_params_, best_cv_auc


# ---------------------------------------------------------------------------
# 5. Evaluation & reporting
# ---------------------------------------------------------------------------

def evaluate_and_log(
    best_name: str,
    final_pipe,
    best_params: dict,
    best_cv_auc: float,
    baseline_results: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    print(f"\n[6/6] Evaluating and logging final model ({best_name})...")

    y_pred = final_pipe.predict(X_test)
    y_pred_prob = final_pipe.predict_proba(X_test)[:, 1]

    final_metrics = {
        **_compute_metrics(y_test, y_pred, y_pred_prob),
        "best_cv_roc_auc": best_cv_auc,
    }

    with mlflow.start_run(run_name=f"best_model_{best_name}_tuned"):
        mlflow.log_param("model", best_name)
        mlflow.log_param("stage", "tuned_final")
        mlflow.log_params({k: str(v) for k, v in best_params.items()})
        mlflow.log_metrics(final_metrics)
        mlflow.sklearn.log_model(final_pipe, "best_model")
        mlflow.log_artifact(_save_confusion_matrix(y_test, y_pred, best_name))

    # Save model to disk for Streamlit inference
    model_path = os.path.join(OUTPUT_PATH, "readmission_model.joblib")
    joblib.dump(final_pipe, model_path)
    print(f"  Model saved → {model_path}")

    _print_report(best_name, final_metrics, y_test, y_pred, baseline_results)
    return final_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred, y_pred_prob) -> dict:
    return {
        "roc_auc":   round(roc_auc_score(y_true, y_pred_prob), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }


def _save_confusion_matrix(y_test, y_pred, model_name: str) -> str:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Not Readmitted", "Readmitted"],
        yticklabels=["Not Readmitted", "Readmitted"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (tuned)")
    plt.tight_layout()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    path = os.path.join(OUTPUT_PATH, "confusion_matrix.png")
    plt.savefig(path)
    plt.show()
    return path


def _print_report(model_name, metrics, y_test, y_pred, baseline_results) -> None:
    print("\n")
    print("  FINAL MODEL RESULTS")
    print(f"  Model      : {model_name} (tuned)")
    print(f"  ROC-AUC    : {metrics['roc_auc']}")
    print(f"  F1 Score   : {metrics['f1']}")
    print(f"  Precision  : {metrics['precision']}")
    print(f"  Recall     : {metrics['recall']}")
    print("=" * 55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Readmitted", "Readmitted"]))

    print("\nBaseline Model Comparison:")
    comparison = pd.DataFrame(
        {k: {m: v for m, v in baseline_results[k].items() if m != "pipe"}
         for k in baseline_results}
    ).T.sort_values("roc_auc", ascending=False)
    print(comparison.to_string())

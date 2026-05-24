"""
High-Risk Patient Identification Pipeline
==========================================
Target (engineered):
    high_risk = 1 if any of the following is true:
        - readmitted within 30 days
        - length of stay > 7 days
        - ICU stay > 0 days
        - 3+ comorbidities
        - 3+ prior admissions

Steps:
    1. Load master dataset
    2. Engineer target
    3. Prepare features  (drop leakage columns)
    4. Split data
    5. Build preprocessor
    6. Train & evaluate  (Logistic Regression, Random Forest, XGBoost)
    7. Hyperparameter tune the best baseline model
    8. Final evaluation & MLflow log
"""

import os
import sys

# Ensure the project root is on the path when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config import MLFLOW_HIGH_RISK_EXPERIMENT, MLFLOW_TRACKING_URI, OUTPUT_PATH, PARAM_GRIDS
from tracking.mlflow_utils import check_mlflow_connection


# ---------------------------------------------------------------------------
# Columns that would cause data leakage
# (they are either direct inputs to the target formula or derived from them)
# ---------------------------------------------------------------------------
LEAKAGE_COLS = [
    # --- Direct target inputs ---
    "readmitted_within_30d",
    "length_of_stay_days",
    "icu_days",
    "total_comorbidities",
    "num_prior_admissions",

    # --- Derived leakage ---
    "icu_admitted",
    "had_icu_stay",
    "icu_ratio",
    "cost_per_day",
    "is_high_utiliser",
    "readmission_reason",

    # --- Post-outcome / severity leakage ---
    "discharge_disposition",
    "total_charges_usd",
    "insurance_paid_usd",
    "out_of_pocket_cost",
    "insurance_coverage_ratio",

    # High-cardinality / noise
    "primary_diagnosis_desc",
    "attending_physician_id",
    "zip_code",
    "hospital",
    "ward",
    "primary_hospital",

    # Redundant features
    "icu_admitted",
    "had_icu_stay",
    "is_high_comorbidity",
    "age_group",
    "low_social_support",
]


# Standard ID and date columns — no predictive value
ID_DATE_COLS = [
    "admission_id",
    "patient_id",
    "admission_date",
    "discharge_date",
    "original_discharge_date",
    "date_of_birth",   # replace with AGE instead
    "registered_date",
]


# ---------------------------------------------------------------------------
# Step 1: Load master dataset
# ---------------------------------------------------------------------------
def _load_dataset() -> pd.DataFrame:
    path = os.path.join(OUTPUT_PATH, "master_dataset.csv")
    print(f"  Loading: {path}")
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Step 2: Engineer the high_risk target
# ---------------------------------------------------------------------------
def _engineer_target(df: pd.DataFrame) -> pd.Series:
    high_risk = (
        (df["readmitted_within_30d"] == 1) |
        (df["length_of_stay_days"] > 7)    |
        (df["icu_days"] > 0)               |
        (df["total_comorbidities"] >= 3)   |
        (df["num_prior_admissions"] >= 3)
    ).astype(int)
    return high_risk


# ---------------------------------------------------------------------------
# Step 3: Prepare features — remove leakage and ID/date columns
# ---------------------------------------------------------------------------
def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in df.columns if c in LEAKAGE_COLS + ID_DATE_COLS]
    X = df.drop(columns=cols_to_drop)
    X = X.select_dtypes(exclude=["datetime64[ns]", "datetime64"])
    return X


# ---------------------------------------------------------------------------
# Step 5: Build preprocessor
# ---------------------------------------------------------------------------
def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
# Step 6: Train and evaluate one model — logs to MLflow
# ---------------------------------------------------------------------------
def _train_and_evaluate(
    model_name: str,
    model,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred      = pipe.predict(X_test)
    y_pred_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":   round(roc_auc_score(y_test, y_pred_prob), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }

    try:
        with mlflow.start_run(run_name=f"high_risk_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("target", "high_risk")
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path=model_name)
    except Exception as e:
        print(f"  MLflow logging skipped — {e}")

    print(
        f"  {model_name:<25} "
        f"ROC-AUC={metrics['roc_auc']}  "
        f"F1={metrics['f1']}  "
        f"Precision={metrics['precision']}  "
        f"Recall={metrics['recall']}"
    )
    return {**metrics, "pipe": pipe}


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------
def _save_confusion_matrix(y_test, y_pred) -> str:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — High-Risk Model (tuned XGBoost)")
    plt.tight_layout()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    path = os.path.join(OUTPUT_PATH, "high_risk_confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.show()
    print(f"  Confusion matrix saved → {path}")
    return path


# ---------------------------------------------------------------------------
# SHAP explainability — top 30 positive-contributing features
# ---------------------------------------------------------------------------
def _shap_analysis(final_pipe, X_test: pd.DataFrame) -> None:
    if not SHAP_AVAILABLE:
        print("\n  SHAP skipped — run: pip install shap")
        return

    print("\n── SHAP Analysis: Top 30 Features Driving High-Risk ──")

    # Step 1: get the preprocessor and model out of the pipeline
    preprocessor = final_pipe.named_steps["preprocessor"]
    xgb_model    = final_pipe.named_steps["model"]

    # Step 2: get feature names (numeric + one-hot encoded categoricals)
    num_names = list(preprocessor.transformers_[0][2])
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(preprocessor.transformers_[1][2])
    )
    feature_names = num_names + cat_names

    # Step 3: transform the test data and compute SHAP values
    X_transformed = preprocessor.transform(X_test)
    shap_values   = shap.TreeExplainer(xgb_model).shap_values(X_transformed)

    # Step 4: average SHAP value per feature → positive = pushes toward high-risk
    mean_shap = pd.Series(shap_values.mean(axis=0), index=feature_names)
    top30     = mean_shap[mean_shap > 0].nlargest(30)

    # Step 5: print ranked table
    print(f"\n  {'Rank':<5} {'Feature':<45} {'Mean SHAP':>10}")
    print(f"  {'-'*62}")
    for rank, (feat, val) in enumerate(top30.items(), 1):
        print(f"  {rank:<5} {feat:<45} {val:>10.4f}")

    # Step 6: save bar chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9))
    top30.sort_values().plot(kind="barh", ax=ax, color="#d62728")
    ax.set_title("Top 30 Features — Positive SHAP Contribution to High-Risk")
    ax.set_xlabel("Mean SHAP Value")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PATH, "shap_top30_positive.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\n  Chart saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------
def run_high_risk_pipeline():
    # --- Step 1: Load ---
    print("\n[1/6] Loading master dataset...")
    df = _load_dataset()
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    # --- Step 2: Engineer target ---
    print("\n[2/6] Engineering high_risk target...")
    y = _engineer_target(df)
    print(f"  high_risk — 0: {(y == 0).sum():,}  |  1: {(y == 1).sum():,}  ({y.mean():.1%} positive)")

    # --- Step 3: Prepare features ---
    print("\n[3/6] Preparing features (removing leakage columns)...")
    X = _prepare_features(df)
    print(f"  Features remaining: {X.shape[1]}")

    # Save feature defaults for Streamlit inference
    num_defaults = X.select_dtypes(include="number").median().to_dict()
    cat_defaults = {c: X[c].mode()[0] for c in X.select_dtypes(include=["object", "category"]).columns}
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_PATH, "high_risk_defaults.json"), "w") as f:
        json.dump({**num_defaults, **cat_defaults}, f)

    # --- Step 4: Split ---
    print("\n[4/6] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # --- Step 5: Preprocessor ---
    print("\n[5/6] Building preprocessor...")
    preprocessor = _build_preprocessor(X_train)

    # --- Step 6: Train all models ---
    print("\n[6/8] Training baseline models...\n")

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # DummyClassifier sets the performance floor — any real model must beat this
    candidate_models = {
        "dummy_baseline": DummyClassifier(strategy="most_frequent", random_state=42),
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
        results[model_name] = _train_and_evaluate(
            model_name, model, preprocessor, X_train, y_train, X_test, y_test
        )

    # --- Baseline summary ---
    print("\n" + "=" * 65)
    print("  BASELINE MODEL COMPARISON")
    print("=" * 65)
    comparison = pd.DataFrame(
        {k: {m: v for m, v in results[k].items() if m != "pipe"} for k in results}
    ).T.sort_values("recall", ascending=False)  # recall is primary in healthcare
    print(comparison.to_string())
    print("=" * 65)

    # XGBoost selected as best overall model based on results analysis
    best_name = "xgboost"
    print(f"\n  Selected model for tuning : {best_name}  "
          f"(Recall={comparison.loc[best_name, 'recall']}  ROC-AUC={comparison.loc[best_name, 'roc_auc']})")
    print(f"  Dummy baseline : ROC-AUC={comparison.loc['dummy_baseline', 'roc_auc']}  "
          f"(gap: +{round(comparison.loc[best_name, 'roc_auc'] - comparison.loc['dummy_baseline', 'roc_auc'], 4)})")

    # --- Step 7: Hyperparameter tuning ---
    print(f"\n[7/8] Tuning {best_name} with RandomizedSearchCV...")

    tuning_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(candidate_models[best_name])),
    ])
    search = RandomizedSearchCV(
        tuning_pipe,
        param_distributions=PARAM_GRIDS[best_name],
        n_iter=50,           # more combinations searched
        scoring="roc_auc",   # optimise discrimination directly; threshold=0.4 handles recall
        cv=5,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_cv_auc = round(search.best_score_, 4)
    print(f"  Best CV ROC-AUC : {best_cv_auc}")
    print(f"  Best params     : {search.best_params_}")

    # --- Step 8: Final evaluation & log ---
    print(f"\n[8/8] Evaluating tuned {best_name} on test set...")

    final_pipe  = search.best_estimator_
    y_pred_prob = final_pipe.predict_proba(X_test)[:, 1]

    # Lower threshold flags more patients as high-risk — maximises recall.
    # Default sklearn threshold is 0.5. Use 0.4 to catch more high-risk patients
    # at the cost of more false alarms (clinically preferred over missing cases).
    THRESHOLD = 0.4
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    final_metrics = {
        "roc_auc":        round(roc_auc_score(y_test, y_pred_prob), 4),
        "f1":             round(f1_score(y_test, y_pred), 4),
        "precision":      round(precision_score(y_test, y_pred), 4),
        "recall":         round(recall_score(y_test, y_pred), 4),
        "best_cv_roc_auc": best_cv_auc,
    }

    try:
        with mlflow.start_run(run_name=f"high_risk_{best_name}_tuned"):
            mlflow.log_param("model", best_name)
            mlflow.log_param("target", "high_risk")
            mlflow.log_param("stage", "tuned_final")
            mlflow.log_param("decision_threshold", THRESHOLD)
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics(final_metrics)
            mlflow.sklearn.log_model(final_pipe, artifact_path=f"{best_name}_tuned")
    except Exception as e:
        print(f"  MLflow logging skipped — {e}")

    # Save model to disk for Streamlit inference
    model_path = os.path.join(OUTPUT_PATH, "high_risk_model.joblib")
    joblib.dump(final_pipe, model_path)
    print(f"  Model saved → {model_path}")

    # SHAP explainability
    _shap_analysis(final_pipe, X_test)

    print("\n" + "=" * 65)
    print("  FINAL TUNED MODEL RESULTS")
    print("=" * 65)
    print(f"  Model      : {best_name} (tuned)")
    print(f"  ROC-AUC    : {final_metrics['roc_auc']}")
    print(f"  F1 Score   : {final_metrics['f1']}")
    print(f"  Precision  : {final_metrics['precision']}")
    print(f"  Recall     : {final_metrics['recall']}")
    print("=" * 65)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    print("\nConfusion Matrix:")
    _save_confusion_matrix(y_test, y_pred)

    return results, final_pipe, final_metrics


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_HIGH_RISK_EXPERIMENT)
    except Exception:
        print("Warning: MLflow server not reachable — MLflow logging will be skipped.")

    run_high_risk_pipeline()

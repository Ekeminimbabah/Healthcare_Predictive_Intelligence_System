import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "healthcare_intelligence")
MLFLOW_HIGH_RISK_EXPERIMENT: str = os.getenv("MLFLOW_HIGH_RISK_EXPERIMENT", "healthcare_high_risk")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
OUTPUT_PATH: str = os.getenv("OUTPUT_PATH", "model")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TARGET_COL = "readmitted_within_30d"

DROP_COLS = [
    "readmitted_within_30d",
    "admission_id",
    "patient_id",
    "admission_date",
    "discharge_date",
    "original_discharge_date",
    "date_of_birth",
    "registered_date",
]

# ---------------------------------------------------------------------------
# Hyperparameter search grids
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict = {
    "logistic_regression": {
        "model__C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l1", "l2"],
        "model__solver":  ["liblinear", "saga"],
    },
    "random_forest": {
        "model__n_estimators":      [100, 200, 300],
        "model__max_depth":         [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":      ["sqrt", "log2"],
    },
    "xgboost": {
        "model__n_estimators":     [100, 200, 300, 500],
        "model__max_depth":        [3, 4, 5, 6, 7, 9],
        "model__learning_rate":    [0.005, 0.01, 0.05, 0.1, 0.2],
        "model__subsample":        [0.5, 0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.5, 0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__gamma":            [0, 0.1, 0.3, 0.5],
        "model__reg_alpha":        [0, 0.01, 0.1, 1.0],
        "model__reg_lambda":       [0.5, 1.0, 2.0, 5.0],
    },
}

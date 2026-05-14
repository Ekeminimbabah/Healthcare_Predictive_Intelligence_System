from tracking.mlflow_utils import check_mlflow_connection, setup_mlflow
from training.pipeline import (
    build_preprocessor,
    evaluate_and_log,
    load_and_split,
    train_baselines,
    tune_best_model,
)


def run_training_pipeline():
    # 1 & 2: Load data and split
    X_train, X_test, y_train, y_test, X_full = load_and_split()

    # 3: Build preprocessor
    preprocessor = build_preprocessor(X_full)

    # 4: Train all baseline models
    results, candidate_models = train_baselines(preprocessor, X_train, y_train, X_test, y_test)

    # 5: Select best and tune
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n  Best baseline: {best_name}  (ROC-AUC={results[best_name]['roc_auc']})")
    final_pipe, best_params, best_cv_auc = tune_best_model(
        best_name, candidate_models, preprocessor, X_train, y_train
    )

    # 6: Evaluate, log, and report
    final_metrics = evaluate_and_log(
        best_name, final_pipe, best_params, best_cv_auc, results, X_test, y_test
    )
    return final_pipe, final_metrics


if __name__ == "__main__":
    setup_mlflow()
    if not check_mlflow_connection():
        raise SystemExit("Aborting: MLflow server is not reachable.")
    run_training_pipeline()

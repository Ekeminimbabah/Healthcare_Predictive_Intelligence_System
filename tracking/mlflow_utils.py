import mlflow

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def check_mlflow_connection() -> bool:
    print("\n" + "=" * 50)
    print("  MLflow Connectivity Check")
    print("=" * 50)

    tracking_uri = mlflow.get_tracking_uri()
    print(f"  Tracking URI : {tracking_uri}")

    try:
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        experiment_names = [e.name for e in experiments]
        print(f"  Status       : CONNECTED")
        print(f"  Experiments  : {experiment_names}")
        print("=" * 50 + "\n")
        return True
    except Exception as e:
        print(f"  Status       : FAILED")
        print(f"  Error        : {e}")
        print("  Hint         : Run 'mlflow server' in a separate terminal first.")
        print("=" * 50 + "\n")
        return False

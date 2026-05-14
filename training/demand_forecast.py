"""
ED Demand Forecasting Pipeline
================================
Target: Daily Emergency Department (ED) visit count

Steps:
    1.  Build daily ED visits dataset  (group admissions by date)
    2.  Basic EDA  (plot trend, check missing dates)
    3.  Set date as index
    4.  Check stationarity  (ADF test — is the series stable over time?)
    5.  Check seasonality   (seasonal decomposition — weekly/monthly pattern?)
    6.  Decide model        (ARIMA if no seasonality, SARIMAX if seasonal)
    7.  Time-based split    (train = up to 2023, test = 2024 onward)
    8.  Fit AutoARIMA       (past observations only — no external features)
    9.  Evaluate            (MAE, RMSE, R²)
    10. Fit Prophet         (compare — handles spikes and seasonality)
    11. Log to MLflow
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")   # works without a screen (servers, notebooks)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import joblib
import mlflow

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MLFLOW_TRACKING_URI

# ── Constants ──────────────────────────────────────────────────────────────
DATA_PATH         = os.path.join("model", "master_dataset.csv")
PLOT_DIR          = os.path.join("model", "demand_forecast_plots")
TRAIN_END         = "2023-12-31"
TEST_START        = "2024-01-01"
MLFLOW_EXPERIMENT = "healthcare_demand_forecast"

os.makedirs(PLOT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Build daily ED visit dataset
# ══════════════════════════════════════════════════════════════════════════
def build_dataset() -> pd.Series:
    """
    Load master_dataset.csv and count total ED visits per day.
    Result: a pandas Series  date → ed_visit_count
    """
    print("\n── STEP 1: Build daily ED visit dataset ──")

    df = pd.read_csv(DATA_PATH, usecols=["admission_date", "num_ed_visits"])
    df["admission_date"] = pd.to_datetime(df["admission_date"])

    # Group by date → sum ED visits for that day
    daily = df.groupby("admission_date")["num_ed_visits"].sum().sort_index()

    # Fill any missing calendar dates with linear interpolation
    full_calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing_dates = full_calendar.difference(daily.index)
    daily = daily.reindex(full_calendar).interpolate(method="time")

    print(f"  Date range   : {daily.index.min().date()} → {daily.index.max().date()}")
    print(f"  Total days   : {len(daily)}")
    print(f"  Missing days : {len(missing_dates)} (filled by interpolation)")
    print(f"  Mean visits  : {daily.mean():.1f} per day")
    print(f"  Min / Max    : {daily.min():.0f} / {daily.max():.0f}")

    return daily


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Basic EDA: plot the series
# ══════════════════════════════════════════════════════════════════════════
def eda(daily: pd.Series) -> None:
    """
    Plot the raw daily series + weekly average so we can SEE the trend.
    Saves chart to model/demand_forecast_plots/eda_raw.png
    """
    print("\n── STEP 2: Basic EDA — plotting the series ──")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    daily.plot(ax=axes[0], color="#1f77b4", title="Daily ED Visits (raw)")
    axes[0].set_ylabel("ED Visits / Day")

    daily.resample("W").mean().plot(
        ax=axes[1], color="#ff7f0e", title="Weekly Average ED Visits (smoother trend)"
    )
    axes[1].set_ylabel("Avg ED Visits / Week")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "eda_raw.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Set date as index
# ══════════════════════════════════════════════════════════════════════════
def set_index(daily: pd.Series) -> pd.Series:
    """
    Make sure the series has a proper daily DatetimeIndex.
    This is required by all time-series models.
    """
    print("\n── STEP 3: Set date as index ──")
    daily.index.name = "date"
    daily.index.freq = "D"          # tell pandas this is daily data
    daily.name = "ed_visits"
    print("  DatetimeIndex set with freq='D'  ✓")
    return daily


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Check stationarity (ADF test)
# ══════════════════════════════════════════════════════════════════════════
def check_stationarity(daily: pd.Series) -> bool:
    """
    ADF (Augmented Dickey-Fuller) test:
      H0: the series has a unit root (NOT stationary)
      If p-value < 0.05 → reject H0 → series IS stationary

    Stationary = the mean and variance don't drift over time.
    If NOT stationary, ARIMA will apply differencing (d > 0) to fix it.
    """
    print("\n── STEP 4: Check stationarity (ADF test) ──")

    adf_stat, p_value, *_ = adfuller(daily.dropna())
    is_stationary = p_value < 0.05

    print(f"  ADF statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")

    if is_stationary:
        print("  Result        : STATIONARY ✓  (no differencing needed, d=0)")
    else:
        print("  Result        : NOT stationary ✗  (ARIMA will difference the data, d≥1)")

    return is_stationary


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Check seasonality (seasonal decomposition)
# ══════════════════════════════════════════════════════════════════════════
def check_seasonality(daily: pd.Series) -> bool:
    """
    Seasonal decomposition splits the series into:
       Trend + Seasonal + Residual

    We measure 'seasonal strength':
       strength = 1 - Var(Residual) / Var(Seasonal + Residual)
       Range: 0 (no seasonality) → 1 (strong seasonality)

    If strength > 0.30 → we treat it as seasonal → use SARIMAX
    Otherwise           → plain ARIMA is enough
    """
    print("\n── STEP 5: Check seasonality ──")

    # Decompose with a weekly period (7 days) — most common in healthcare
    decomp = seasonal_decompose(daily, model="additive", period=7, extrapolate_trend="freq")

    seasonal  = decomp.seasonal
    residual  = decomp.resid.dropna()

    # Seasonal strength formula (from Hyndman & Athanasopoulos)
    var_residual          = np.var(residual)
    var_seasonal_residual = np.var(seasonal.dropna() + residual)
    seasonal_strength     = max(0, 1 - var_residual / var_seasonal_residual)
    is_seasonal = seasonal_strength > 0.30

    print(f"  Seasonal strength (weekly) : {seasonal_strength:.3f}")
    print(f"  Threshold                  : 0.30")

    if is_seasonal:
        print(f"  Result : SEASONAL pattern detected → will use SARIMAX (m=7)")
    else:
        print(f"  Result : No strong seasonality → will use plain ARIMA")

    # Save decomposition plot so we can visually inspect it
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    decomp.observed.plot(ax=axes[0],  title="Observed",  color="#1f77b4")
    decomp.trend.plot(ax=axes[1],     title="Trend",     color="#ff7f0e")
    decomp.seasonal.plot(ax=axes[2],  title="Seasonal",  color="#2ca02c")
    decomp.resid.plot(ax=axes[3],     title="Residual",  color="#d62728")
    plt.suptitle(f"Seasonal Decomposition (period=7 days)  |  Seasonal strength={seasonal_strength:.3f}")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "seasonal_decomposition.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Decomposition plot saved → {path}")

    return is_seasonal


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Decide which model to use
# ══════════════════════════════════════════════════════════════════════════
def decide_model(is_seasonal: bool) -> str:
    """
    ARIMA  → works on non-seasonal data  (p, d, q)
    SARIMAX → extends ARIMA for seasonal data  (p,d,q)(P,D,Q,m)

    We let AutoARIMA optimise the exact numbers automatically.
    """
    print("\n── STEP 6: Model decision ──")

    if is_seasonal:
        model_choice = "SARIMAX"
        print("  → Seasonal data detected")
        print("  → Chosen model: SARIMAX  (handles both trend and seasonality)")
        print("  → AutoARIMA will find the best (p,d,q)(P,D,Q,7) automatically")
    else:
        model_choice = "ARIMA"
        print("  → No strong seasonality")
        print("  → Chosen model: ARIMA  (handles trend only)")
        print("  → AutoARIMA will find the best (p,d,q) automatically")

    return model_choice


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Time-based train / test split
# ══════════════════════════════════════════════════════════════════════════
def split(daily: pd.Series):
    """
    For time series we NEVER shuffle randomly.
    We cut at a fixed date:
      Train → 2020-2023  (the model learns from this history)
      Test  → 2024       (we evaluate on unseen future dates)
    """
    print("\n── STEP 7: Time-based train/test split ──")

    train_y = daily[daily.index <= TRAIN_END]
    test_y  = daily[daily.index >= TEST_START]

    print(f"  Train: {train_y.index.min().date()} → {train_y.index.max().date()}  ({len(train_y)} days)")
    print(f"  Test : {test_y.index.min().date()} → {test_y.index.max().date()}  ({len(test_y)} days)")

    return train_y, test_y


# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — Fit AutoARIMA
# ══════════════════════════════════════════════════════════════════════════
def fit_autoarima(train_y: pd.Series, model_choice: str):
    """
    AutoARIMA searches for the best (p, d, q) — or (p,d,q)(P,D,Q,7) for
    seasonal data — using only the past values of the series.
    No external/calendar features: the model learns purely from history.
    """
    print(f"\n── STEP 8: Fit AutoARIMA ({model_choice}) ──")
    print("  Searching for best parameters... (may take 1-2 minutes)")

    use_seasonal = (model_choice == "SARIMAX")

    model = pm.auto_arima(
        train_y,
        seasonal=use_seasonal,
        m=7 if use_seasonal else 1,   # weekly period
        start_p=1, max_p=3,
        start_q=1, max_q=3,
        d=None,                       # auto-select differencing
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=None,
        stepwise=True,
        information_criterion="aic",
        test="adf",
        error_action="ignore",
        suppress_warnings=True,
        n_jobs=1,
    )

    print(f"  Best order found : {model.order}")
    if use_seasonal:
        print(f"  Seasonal order   : {model.seasonal_order}")
    print(f"  AIC              : {model.aic():.2f}")

    return model


# ══════════════════════════════════════════════════════════════════════════
# STEP 11 — Evaluate the model
# ══════════════════════════════════════════════════════════════════════════
def evaluate(model, train_y: pd.Series, test_y: pd.Series):
    """
    Forecast the test period using only past observations and measure accuracy.
      MAE  → average error in visits/day (easy to interpret)
      RMSE → penalises large spikes more than MAE
      R²   → how much variance the model explains (1.0 = perfect)
    """
    print("\n── STEP 9: Evaluate model ──")

    y_pred, conf_arr = model.predict(n_periods=len(test_y), return_conf_int=True)
    y_true = test_y.values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  MAE  = {mae:6.2f} visits/day      │")
    print(f"  │  RMSE = {rmse:6.2f} visits/day      │")
    print(f"  │  R²   = {r2:6.3f}                 │")
    print(f"  └─────────────────────────────────┘")
    print(f"    On average, the model is off by {mae:.1f} ED visits per day")

    # ── Forecast plot ───────────────────────────────────────────────────
    conf_df = pd.DataFrame(conf_arr, index=test_y.index, columns=["lower", "upper"])

    fig, ax = plt.subplots(figsize=(16, 6))
    train_y.iloc[-90:].plot(ax=ax, label="Train (last 90 days)", color="#1f77b4")
    test_y.plot(ax=ax, label="Actual", color="#2ca02c")
    pd.Series(y_pred, index=test_y.index).plot(ax=ax, label="Forecast", color="#d62728", linestyle="--")
    ax.fill_between(test_y.index, conf_df["lower"], conf_df["upper"],
                    alpha=0.2, color="#d62728", label="95% confidence interval")
    ax.set_title(f"Daily ED Visits — Forecast vs Actual  (MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f})")
    ax.set_ylabel("ED Visits")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, "forecast_vs_actual.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\n  Forecast plot saved → {plot_path}")

    return mae, rmse, r2, plot_path


# ══════════════════════════════════════════════════════════════════════════
# STEP 11b — Prophet model (spike / holiday capture)
# ══════════════════════════════════════════════════════════════════════════
def fit_prophet(train_y: pd.Series, test_y: pd.Series) -> tuple:
    """
    Fits a simple Prophet model and evaluates it on the test set.
    Prophet is good at capturing spikes and holiday effects automatically.
    Returns (mae, rmse, plot_path) — or None if Prophet is not installed.
    """
    if not PROPHET_AVAILABLE:
        print("\n── STEP 11b: Prophet — SKIPPED (run: pip install prophet) ──")
        return None

    print("\n── STEP 11b: Fit Prophet model ──")

    # Prophet requires a DataFrame with columns 'ds' (date) and 'y' (value)
    train_df = train_y.reset_index().rename(columns={"date": "ds", "ed_visits": "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    model.fit(train_df)

    # Build a future DataFrame covering the test period
    future = pd.DataFrame({"ds": test_y.index})
    forecast = model.predict(future)

    y_pred = forecast["yhat"].values
    y_true = test_y.values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Prophet MAE  = {mae:6.2f}           │")
    print(f"  │  Prophet RMSE = {rmse:6.2f}           │")
    print(f"  │  Prophet R²   = {r2:6.3f}           │")
    print(f"  └─────────────────────────────────┘")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    train_y.iloc[-90:].plot(ax=ax, label="Train (last 90 days)", color="#1f77b4")
    test_y.plot(ax=ax, label="Actual", color="#2ca02c")
    pd.Series(y_pred, index=test_y.index).plot(ax=ax, label="Prophet Forecast",
                                                color="#9467bd", linestyle="--")
    ax.fill_between(test_y.index,
                    forecast["yhat_lower"].values,
                    forecast["yhat_upper"].values,
                    alpha=0.2, color="#9467bd", label="Uncertainty interval")
    ax.set_title(f"Prophet — Forecast vs Actual  (MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f})")
    ax.set_ylabel("ED Visits")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, "prophet_forecast_vs_actual.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Prophet forecast plot saved → {plot_path}")

    return mae, rmse, r2, plot_path


# ══════════════════════════════════════════════════════════════════════════
# STEP 10c — XGBoost with lag features (spike capture)
# ══════════════════════════════════════════════════════════════════════════
def fit_xgboost(daily: pd.Series, train_y: pd.Series, test_y: pd.Series) -> tuple:
    """
    XGBoost trained on lag features built from the historical series.

    Lag features used:
      lag_1          → yesterday's ED visits  (strongest short-term signal)
      lag_7          → same day last week     (captures weekly spikes)
      rolling_mean_7 → 7-day average          (smooths noise)

    Lags are computed from the full series so the first test-day lags
    correctly reference the last training-day values.
    """
    if not XGBOOST_AVAILABLE:
        print("\n── STEP 10c: XGBoost — SKIPPED (run: pip install xgboost) ──")
        return None

    print("\n── STEP 10c: Fit XGBoost with lag features ──")

    # Build lag features from the full series (train + test)
    # This ensures test lags reference real training history, not future values
    df = daily.to_frame(name="ed_visits")
    df["lag_1"]          = df["ed_visits"].shift(1)
    df["lag_7"]          = df["ed_visits"].shift(7)
    df["rolling_mean_7"] = df["ed_visits"].rolling(7).mean()
    df = df.dropna()   # drop the first 7 rows that have no lag history

    feature_cols = ["lag_1", "lag_7", "rolling_mean_7"]

    train_df = df[df.index <= TRAIN_END]
    test_df  = df[df.index >= TEST_START]

    X_train, y_train = train_df[feature_cols], train_df["ed_visits"]
    X_test,  y_true  = test_df[feature_cols],  test_df["ed_visits"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Save model and the last 7 training values (needed to seed lags at inference time)
    joblib.dump(model, os.path.join("model", "xgb_demand_model.joblib"))
    np.save(os.path.join("model", "xgb_seed_values.npy"),
            train_df["ed_visits"].values[-7:])
    print("  XGBoost model saved → model/xgb_demand_model.joblib")

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true.values - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  XGBoost MAE  = {mae:6.2f}           │")
    print(f"  │  XGBoost RMSE = {rmse:6.2f}           │")
    print(f"  │  XGBoost R²   = {r2:6.3f}           │")
    print(f"  └─────────────────────────────────┘")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    train_y.iloc[-90:].plot(ax=ax, label="Train (last 90 days)", color="#1f77b4")
    test_y.plot(ax=ax, label="Actual", color="#2ca02c")
    pd.Series(y_pred, index=y_true.index).plot(ax=ax, label="XGBoost Forecast",
                                               color="#e377c2", linestyle="--")
    ax.set_title(f"XGBoost (lag features) — Forecast vs Actual  (MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f})")
    ax.set_ylabel("ED Visits")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, "xgboost_forecast_vs_actual.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  XGBoost forecast plot saved → {plot_path}")

    return mae, rmse, r2, plot_path


# ══════════════════════════════════════════════════════════════════════════
# STEP 12 — Log everything to MLflow
# ══════════════════════════════════════════════════════════════════════════
def log_mlflow(model, model_choice: str, mae: float, rmse: float, r2: float,
                      train_size: int, test_size: int, plot_path: str):
    """Log params, metrics and the forecast plot to MLflow for tracking."""
    print("\n── STEP 12: Log to MLflow ──")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"autoarima_{model_choice.lower()}"):
        # Model configuration
        mlflow.log_param("model_type",    model_choice)
        mlflow.log_param("arima_order",   str(model.order))
        mlflow.log_param("seasonal_order",str(model.seasonal_order))
        mlflow.log_param("train_end",     TRAIN_END)
        mlflow.log_param("test_start",    TEST_START)
        mlflow.log_param("train_days",    train_size)
        mlflow.log_param("test_days",     test_size)
        mlflow.log_param("aic",           round(model.aic(), 2))

        # Evaluation metrics
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)

        # Save the forecast chart as an artifact
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(os.path.join(PLOT_DIR, "eda_raw.png"))
        mlflow.log_artifact(os.path.join(PLOT_DIR, "seasonal_decomposition.png"))

        run_id = mlflow.active_run().info.run_id
        print(f"  Run logged  →  run_id = {run_id}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN — run all steps in order
# ══════════════════════════════════════════════════════════════════════════
def run_demand_forecast():
    print("=" * 60)
    print("   ED DEMAND FORECASTING PIPELINE")
    print("=" * 60)

    # Step 1: Get the data
    daily = build_dataset()

    # Step 2: Plot it so we can see the shape
    eda(daily)

    # Step 3: Set date as index (required for time-series models)
    daily = set_index(daily)

    # Step 4: Is the series stationary? (affects differencing parameter d)
    is_stationary = check_stationarity(daily)

    # Step 5: Does it have a seasonal pattern?
    is_seasonal = check_seasonality(daily)

    # Step 6: Based on step 5, choose ARIMA or SARIMAX
    model_choice = decide_model(is_seasonal)

    # Step 7: Time-based split (NO random shuffle)
    train_y, test_y = split(daily)

    # Step 8: Fit model on past observations only
    model = fit_autoarima(train_y, model_choice)

    # Save model + last training date for Streamlit inference
    joblib.dump(model, os.path.join("model", "demand_model.joblib"))
    with open(os.path.join("model", "demand_last_date.txt"), "w") as f:
        f.write(str(train_y.index.max().date()))

    # Step 9: Evaluate with MAE, RMSE, R²
    mae, rmse, r2, plot_path = evaluate(model, train_y, test_y)

    # Step 10: Prophet — compare against ARIMA/SARIMAX
    prophet_result = fit_prophet(train_y, test_y)

    # Step 10c: XGBoost with lag features — best for spike capture
    xgb_result = fit_xgboost(daily, train_y, test_y)

    # Step 11: Log everything to MLflow (skip gracefully if server is offline)
    try:
        log_mlflow(model, model_choice, mae, rmse, r2,
                          len(train_y), len(test_y), plot_path)
    except Exception as e:
        print(f"\n  MLflow logging skipped — server unreachable: {e}")

    # ── Model comparison summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'-'*46}")
    print(f"  {model_choice:<20} {mae:>8.2f} {rmse:>8.2f} {r2:>8.3f}")
    if prophet_result:
        p_mae, p_rmse, p_r2, _ = prophet_result
        print(f"  {'Prophet':<20} {p_mae:>8.2f} {p_rmse:>8.2f} {p_r2:>8.3f}")
    if xgb_result:
        x_mae, x_rmse, x_r2, _ = xgb_result
        print(f"  {'XGBoost (lag feats)':<20} {x_mae:>8.2f} {x_rmse:>8.2f} {x_r2:>8.3f}")

    # Pick the winner by lowest MAE
    results = {model_choice: mae}
    if prophet_result: results["Prophet"] = prophet_result[0]
    if xgb_result:     results["XGBoost"] = xgb_result[0]
    winner = min(results, key=results.get)
    print(f"\n  → Best model (MAE): {winner}")

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE")
    print(f"   Model: {winner}  |  MAE={x_mae:.2f}  RMSE={x_rmse:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_demand_forecast()

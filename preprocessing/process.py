"""
Healthcare Intelligence — Preprocessing Pipeline

Pipelines:
    1. Data Ingestion      — load raw CSV files
    2. Data Cleaning       — handle nulls, types, duplicates, outliers
    3. Feature Engineering — derive analytical features per table
    4. Data Integration    — merge all tables into one master dataset
    5. Data Validation     — verify output quality before modelling

Entry point: run `python process.py` to execute the full pipeline.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)



# Configuration

BASE_PATH = Path(os.getenv("BASE_DATA_PATH", "mhn predictive dataset/MHN_Dataset"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "model"))

RAW_FILES = {
    "admissions":      "admissions.csv",
    "patients":        "patients.csv",
    "diagnoses":       "diagnoses.csv",
    "ed_visits":       "ed_visits.csv",
    "lab_results":     "lab_results.csv",
    "medications":     "medications.csv",
    "vitals":          "vitals.csv",
    "readmissions":    "readmissions.csv",
}

# Columns parsed as datetime
DATETIME_COLS = {
    "admissions":  ["admission_date", "discharge_date", "original_discharge_date"],
    "patients":    ["date_of_birth", "registered_date"],
    "ed_visits":   ["arrival_datetime", "departure_datetime"],
    "lab_results": ["drawn_at"],
    "medications": ["start_datetime", "end_datetime"],
    "vitals":      ["measured_at"],
    "readmissions": ["original_discharge_date","readmission_date"],
}

# Comorbidity flag columns in patients table
COMORBIDITY_COLS = [
    "has_diabetes", "has_hypertension", "has_heart_disease",
    "has_copd", "has_ckd", "has_depression", "has_obesity", "has_cancer",
]



#  Data Ingestion Pipeline

def load_raw_data() -> dict[str, pd.DataFrame]:
    """
    Load all raw CSV files from BASE_PATH.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by dataset name.
    """
    log.info("=== Pipeline 1: Data Ingestion ===")
    data = {}
    for name, filename in RAW_FILES.items():
        path = BASE_PATH / filename
        df = pd.read_csv(path, low_memory=False)
        log.info(f"  Loaded '{name}': {df.shape[0]:,} rows × {df.shape[1]} cols")
        data[name] = df
    return data



# PIPELINE 2 — Data Cleaning
# ===========================================================================

def _parse_datetimes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Parse specified columns to datetime, coercing errors to NaT."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _drop_duplicates(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Remove duplicate rows based on the primary key column."""
    before = len(df)
    df = df.drop_duplicates(subset=[id_col])
    dropped = before - len(df)
    if dropped:
        log.warning(f"    Dropped {dropped:,} duplicate rows on '{id_col}'")
    return df


def clean_admissions(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Cleaning: admissions")
    df = _drop_duplicates(df, "admission_id")
    df = _parse_datetimes(df, DATETIME_COLS["admissions"])
    df["length_of_stay_days"] = df["length_of_stay_days"].clip(lower=0)
    df["total_charges_usd"]   = df["total_charges_usd"].clip(lower=0)
    df["insurance_paid_usd"]  = df["insurance_paid_usd"].clip(lower=0)
    df["icu_days"]            = df["icu_days"].clip(lower=0)
    df["readmitted_within_30d"] = df["readmitted_within_30d"].fillna(0).astype(int)
    # Standardise free-text categoricals
    for col in ["admission_type", "admission_source", "discharge_disposition"]:
        df[col] = df[col].str.strip().str.title()
    return df


def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Cleaning: patients")
    df = _drop_duplicates(df, "patient_id")
    df = _parse_datetimes(df, DATETIME_COLS["patients"])
    # Drop PII columns not needed for modelling
    df = df.drop(columns=["first_name", "last_name", "mrn"], errors="ignore")
    # Standardise categoricals
    for col in ["gender", "ethnicity", "insurance_type",
                "smoking_status", "alcohol_use", "language_preference"]:
        df[col] = df[col].str.strip().str.title()
    df["alcohol_use"]   = df["alcohol_use"].fillna("Unknown")
    df["smoking_status"] = df["smoking_status"].fillna("Unknown")
    df["social_support_score"] = df["social_support_score"].clip(0, 10)
    return df


def clean_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning: diagnoses")
    df = _drop_duplicates(df, "diagnosis_id")
    df["poa_flag"] = df["poa_flag"].str.strip().str.upper()
    df["diagnosis_rank"] = df["diagnosis_rank"].clip(lower=1)
    return df


def clean_ed_visits(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning: ed_visits")
    df = _drop_duplicates(df, "ed_visit_id")
    df = _parse_datetimes(df, DATETIME_COLS["ed_visits"])
    df["wait_time_minutes"]    = df["wait_time_minutes"].clip(lower=0)
    df["door_to_doctor_min"]   = df["door_to_doctor_min"].clip(lower=0)
    df["ed_los_minutes"]       = df["ed_los_minutes"].clip(lower=0)
    df["triage_category"] = df["triage_category"].str.strip().str.title()
    return df


def clean_lab_results(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning: lab_results")
    df = _drop_duplicates(df, "lab_id")
    df = _parse_datetimes(df, DATETIME_COLS["lab_results"])
    df["flag"] = df["flag"].str.strip().str.upper().fillna("N")
    return df


def clean_medications(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Cleaning: medications")
    df = _drop_duplicates(df, "medication_id")
    df = _parse_datetimes(df, DATETIME_COLS["medications"])
    df["is_high_alert"] = df["is_high_alert"].fillna(0).astype(int)
    df["drug_class"] = df["drug_class"].str.strip().str.title()
    return df


def clean_vitals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning: vitals")
    df = _drop_duplicates(df, "vital_id")
    df = _parse_datetimes(df, DATETIME_COLS["vitals"])
    # Physiologically plausible ranges
    df.loc[df["systolic_bp"]      < 50,  "systolic_bp"]      = np.nan
    df.loc[df["systolic_bp"]      > 300, "systolic_bp"]      = np.nan
    df.loc[df["diastolic_bp"]     < 20,  "diastolic_bp"]     = np.nan
    df.loc[df["diastolic_bp"]     > 200, "diastolic_bp"]     = np.nan
    df.loc[df["heart_rate"]       < 20,  "heart_rate"]       = np.nan
    df.loc[df["heart_rate"]       > 300, "heart_rate"]       = np.nan
    df.loc[df["spo2_percent"]     < 50,  "spo2_percent"]     = np.nan
    df.loc[df["temperature_c"]    < 30,  "temperature_c"]    = np.nan
    df.loc[df["temperature_c"]    > 45,  "temperature_c"]    = np.nan
    df.loc[df["weight_kg"]        < 20,  "weight_kg"]        = np.nan
    df.loc[df["height_cm"]        < 100, "height_cm"]        = np.nan
    return df


def clean_readmissions(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Cleaning: readmissions")
    df = _drop_duplicates(df, "readmission_id")
    df["days_to_readmission"] = df["days_to_readmission"].clip(lower=0)
    df["planned_readmission"] = df["planned_readmission"].fillna(0).astype(int)
    df["same_diagnosis"]      = df["same_diagnosis"].fillna(0).astype(int)
    return df


def run_cleaning_pipeline(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply table-specific cleaning to every raw dataframe.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cleaned dataframes keyed by dataset name.
    """
    log.info("=== Pipeline 2: Data Cleaning ===")
    cleaners = {
        "admissions":   clean_admissions,
        "patients":     clean_patients,
        "diagnoses":    clean_diagnoses,
        "ed_visits":    clean_ed_visits,
        "lab_results":  clean_lab_results,
        "medications":  clean_medications,
        "vitals":       clean_vitals,
        "readmissions": clean_readmissions,
    }
    return {name: cleaners[name](df.copy()) for name, df in data.items()}


# ===========================================================================
# PIPELINE 3 — Feature Engineering
# ===========================================================================

def engineer_admissions_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive admission-level features."""
    log.info("  Feature Engineering: admissions")
    df["admission_year"]     = df["admission_date"].dt.year
    df["admission_month"]    = df["admission_date"].dt.month
    df["admission_dayofweek"] = df["admission_date"].dt.dayofweek   # 0=Mon
    df["is_weekend_admission"] = df["admission_dayofweek"].isin([5, 6]).astype(int)
    df["cost_per_day"]       = (df["total_charges_usd"] / df["length_of_stay_days"].replace(0, np.nan)).round(2)
    df["out_of_pocket_cost"] = (df["total_charges_usd"] - df["insurance_paid_usd"]).clip(lower=0)
    df["insurance_coverage_ratio"] = (df["insurance_paid_usd"] / df["total_charges_usd"].replace(0, np.nan)).round(4)
    df["icu_ratio"]          = (df["icu_days"] / df["length_of_stay_days"].replace(0, np.nan)).round(4)
    df["had_icu_stay"]       = (df["icu_days"] > 0).astype(int)
    return df


def engineer_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive patient-level features."""
    log.info("  Feature Engineering: patients")
    bins   = [0, 17, 34, 49, 64, 79, 120]
    labels = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    df["total_comorbidities"] = df[COMORBIDITY_COLS].sum(axis=1)
    df["is_high_comorbidity"]  = (df["charlson_comorbidity_index"] >= 3).astype(int)
    df["is_high_utiliser"] = (
        (df["num_prior_admissions"] >= 3) | (df["num_prior_ed_visits"] >= 5)
    ).astype(int)
    df["low_social_support"] = (df["social_support_score"] <= 3).astype(int)
    return df


def engineer_vitals_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vitals to one row per admission.
    Returns a summary dataframe indexed by admission_id.
    """
    log.info("  Feature Engineering: vitals → admission-level aggregation")
    numeric_vitals = [
        "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
        "temperature_c", "spo2_percent", "gcs_score", "news2_score", "pain_score",
    ]
    # BMI from first recorded weight/height per admission
    bmi_df = (
        df.dropna(subset=["weight_kg", "height_cm"])
        .groupby("admission_id")
        .first()
        .assign(bmi=lambda x: (x["weight_kg"] / (x["height_cm"] / 100) ** 2).round(1))
        [["bmi"]]
    )

    agg_funcs = {col: ["mean", "min", "max", "std"] for col in numeric_vitals}
    agg_funcs["vital_id"] = "count"

    vitals_agg = df.groupby("admission_id").agg(agg_funcs)
    vitals_agg.columns = ["_".join(c).strip("_") for c in vitals_agg.columns]
    vitals_agg = vitals_agg.rename(columns={"vital_id_count": "num_vital_readings"})

    vitals_agg = vitals_agg.join(bmi_df, how="left")
    return vitals_agg.reset_index()


def engineer_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate lab results to one row per admission.
    Returns a summary dataframe indexed by admission_id.
    """
    log.info("  Feature Engineering: lab_results → admission-level aggregation")
    df["is_abnormal"] = (df["flag"] != "N").astype(int)
    lab_agg = df.groupby("admission_id").agg(
        num_lab_tests     = ("lab_id", "count"),
        num_abnormal_labs = ("is_abnormal", "sum"),
    ).reset_index()
    lab_agg["pct_abnormal_labs"] = (
        lab_agg["num_abnormal_labs"] / lab_agg["num_lab_tests"].replace(0, np.nan)
    ).round(4)
    return lab_agg


def engineer_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate medications to one row per admission.
    Returns a summary dataframe indexed by admission_id.
    """
    log.info("  Feature Engineering: medications → admission-level aggregation")
    med_agg = df.groupby("admission_id").agg(
        num_medications         = ("medication_id", "count"),
        num_high_alert_meds     = ("is_high_alert", "sum"),
        num_unique_drug_classes = ("drug_class", "nunique"),
    ).reset_index()
    med_agg["has_high_alert_med"] = (med_agg["num_high_alert_meds"] > 0).astype(int)
    return med_agg


def engineer_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate diagnoses to one row per admission.
    Returns a summary dataframe indexed by admission_id.
    """
    log.info("  Feature Engineering: diagnoses → admission-level aggregation")
    diag_agg = df.groupby("admission_id").agg(
        num_diagnoses      = ("diagnosis_id", "count"),
        num_poa_diagnoses  = ("poa_flag", lambda x: (x == "Y").sum()),
    ).reset_index()
    return diag_agg


def engineer_ed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ED visits to one row per patient.
    Returns a summary dataframe indexed by patient_id.
    """
    log.info("  Feature Engineering: ed_visits → patient-level aggregation")
    ed_agg = df.groupby("patient_id").agg(
        num_ed_visits         = ("ed_visit_id", "count"),
        num_ed_admissions     = ("admitted_from_ed", "sum"),
        avg_ed_wait_minutes   = ("wait_time_minutes", "mean"),
        avg_ed_los_minutes    = ("ed_los_minutes", "mean"),
    ).reset_index()
    ed_agg["avg_ed_wait_minutes"] = ed_agg["avg_ed_wait_minutes"].round(1)
    ed_agg["avg_ed_los_minutes"]  = ed_agg["avg_ed_los_minutes"].round(1)
    return ed_agg


def run_feature_engineering_pipeline(clean: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply all feature engineering steps.

    Returns
    -------
    dict[str, pd.DataFrame]
        Engineered feature tables keyed by name.
    """
    log.info("=== Pipeline 3: Feature Engineering ===")
    return {
        "admissions":  engineer_admissions_features(clean["admissions"].copy()),
        "patients":    engineer_patient_features(clean["patients"].copy()),
        "vitals_agg":  engineer_vitals_features(clean["vitals"].copy()),
        "lab_agg":     engineer_lab_features(clean["lab_results"].copy()),
        "med_agg":     engineer_medication_features(clean["medications"].copy()),
        "diag_agg":    engineer_diagnosis_features(clean["diagnoses"].copy()),
        "ed_agg":      engineer_ed_features(clean["ed_visits"].copy()),
    }


# ===========================================================================
# PIPELINE 4 — Data Integration
# ===========================================================================

def run_integration_pipeline(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all feature tables into one master analytical dataset.
    The grain is one row per admission.

    Target column: `readmitted_within_30d`

    Returns
    -------
    pd.DataFrame
        Master dataset ready for modelling.
    """
    log.info("=== Pipeline 4: Data Integration ===")

    master = features["admissions"].copy()

    # Admission-level feature tables
    for key, join_col in [
        ("vitals_agg", "admission_id"),
        ("lab_agg",    "admission_id"),
        ("med_agg",    "admission_id"),
        ("diag_agg",   "admission_id"),
    ]:
        master = master.merge(features[key], on=join_col, how="left")
        log.info(f"  Merged '{key}' → shape {master.shape}")

    # Patient-level features (join on patient_id)
    master = master.merge(features["patients"], on="patient_id", how="left")
    log.info(f"  Merged 'patients' → shape {master.shape}")

    # ED visit summary (patient-level)
    master = master.merge(features["ed_agg"], on="patient_id", how="left")
    log.info(f"  Merged 'ed_agg' → shape {master.shape}")

    # Fill aggregated counts that are NaN (no records found) with 0
    count_cols = [c for c in master.columns if c.startswith("num_")]
    master[count_cols] = master[count_cols].fillna(0)

    log.info(f"  Master dataset: {master.shape[0]:,} rows × {master.shape[1]} cols")
    return master


# ===========================================================================
# PIPELINE 5 — Data Validation
# ===========================================================================

def run_validation_pipeline(master: pd.DataFrame) -> None:
    """
    Validate the master dataset quality after integration.
    Logs warnings for any issues found; does not raise exceptions.
    """
    log.info("=== Pipeline 5: Data Validation ===")
    total = len(master)

    # 1. Row count sanity
    log.info(f"  Total records: {total:,}")

    # 2. Missing values report
    missing = master.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        log.info("  No missing values detected.")
    else:
        log.info("  Missing values detected:")
        for col, cnt in missing.items():
            log.warning(f"    {col}: {cnt:,} ({cnt/total:.1%})")

    # 3. Target column check
    if "readmitted_within_30d" in master.columns:
        rate = master["readmitted_within_30d"].mean()
        log.info(f"  Target — 30-day readmission rate: {rate:.2%}")
    else:
        log.warning("  Target column 'readmitted_within_30d' not found.")

    # 4. Duplicate admission check
    dupes = master["admission_id"].duplicated().sum()
    if dupes:
        log.warning(f"  Duplicate admission_ids in master: {dupes:,}")
    else:
        log.info("  No duplicate admission_ids found.")

    # 5. Negative numeric values
    numeric_cols = master.select_dtypes(include="number").columns
    for col in numeric_cols:
        neg_count = (master[col] < 0).sum()
        if neg_count:
            log.warning(f"  Negative values in '{col}': {neg_count:,}")

    log.info("  Validation complete.")


# ===========================================================================
# Main Runner
# ===========================================================================

def run_pipeline() -> pd.DataFrame:
    """Execute all pipelines end-to-end and save the master dataset."""
    raw     = load_raw_data()
    cleaned = run_cleaning_pipeline(raw)
    features = run_feature_engineering_pipeline(cleaned)
    master  = run_integration_pipeline(features)
    run_validation_pipeline(master)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_PATH / "master_dataset.csv"
    master.to_csv(out_file, index=False)
    log.info(f"Master dataset saved to: {out_file}")

    return master


if __name__ == "__main__":
    master = run_pipeline()


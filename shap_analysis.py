"""
SHAP Feature Importance
========================
Loads the saved high-risk model and shows the top 30 features
that contribute positively to predicting a patient as high-risk.

Run:
    python shap_analysis.py
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH  = os.path.join("model", "high_risk_model.joblib")
DATA_PATH   = os.path.join("model", "master_dataset.csv")
OUTPUT_PATH = "model"

# ── 1. Load the saved model ────────────────────────────────────────────────
print("Loading model...")
pipe = joblib.load(MODEL_PATH)

preprocessor = pipe.named_steps["preprocessor"]
xgb_model    = pipe.named_steps["model"]

# ── 2. Load a sample of the data to explain ───────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False).sample(500, random_state=42)

# Keep only the columns the preprocessor expects
num_cols = list(preprocessor.transformers_[0][2])
cat_cols = list(preprocessor.transformers_[1][2])
X_sample = df[num_cols + cat_cols]

# ── 3. Transform the sample through the preprocessor ──────────────────────
X_transformed = preprocessor.transform(X_sample)

# ── 4. Get feature names (numeric + one-hot encoded) ──────────────────────
cat_names     = list(preprocessor.named_transformers_["cat"]
                     .named_steps["encoder"]
                     .get_feature_names_out(cat_cols))
feature_names = num_cols + cat_names

# ── 5. Compute SHAP values ─────────────────────────────────────────────────
print("Computing SHAP values...")
shap_values = shap.TreeExplainer(xgb_model).shap_values(X_transformed)

# ── 6. Average SHAP per feature → positive = pushes toward high-risk ───────
mean_shap = pd.Series(shap_values.mean(axis=0), index=feature_names)
top30     = mean_shap[mean_shap > 0].nlargest(30)

# ── 7. Print ranked table ──────────────────────────────────────────────────
print("\n  Top 30 Features Contributing Positively to High-Risk Prediction")
print(f"  {'Rank':<5} {'Feature':<45} {'Mean SHAP':>10}")
print(f"  {'-'*62}")
for rank, (feat, val) in enumerate(top30.items(), 1):
    print(f"  {rank:<5} {feat:<45} {val:>10.4f}")

# ── 8. Save bar chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
top30.sort_values().plot(kind="barh", ax=ax, color="#d62728")
ax.set_title("Top 30 Features — Positive SHAP Contribution to High-Risk")
ax.set_xlabel("Mean SHAP Value")
plt.tight_layout()

chart_path = os.path.join(OUTPUT_PATH, "shap_top30_positive.png")
plt.savefig(chart_path, dpi=120)
plt.close()
print(f"\n  Chart saved → {chart_path}")
print("  Open model/shap_top30_positive.png to view the bar chart.")

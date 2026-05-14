"""
Healthcare Intelligence — Streamlit Dashboard
==============================================
Three interactive prediction pages:
  1. Readmission Prediction   — 30-day readmission risk gauge + factor chart
  2. High-Risk Identification — high-risk score gauge + risk profile chart
  3. ED Demand Forecast       — full time-series chart + weekday breakdown
"""

import json
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Healthcare Intelligence", page_icon="🏥", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
for _key in ["readmission_result", "high_risk_result", "ed_result"]:
    if _key not in st.session_state:
        st.session_state[_key] = None

MODEL_DIR = "model"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f4f7fb; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2137 0%, #1a3c5e 100%);
    padding-top: 20px;
}
[data-testid="stSidebar"] * { color: #e8f4fd !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    display: block;
    font-size: 15px !important;
    font-weight: 500;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.15);
}

/* Hero */
.hero {
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
}
.hero h1 { color: white; font-size: 26px; font-weight: 700; margin: 0 0 6px; }
.hero p  { margin: 0; opacity: 0.85; font-size: 14px; }

/* Section label */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #8a9bb0;
    margin: 18px 0 8px;
}

/* Form card */
[data-testid="stForm"] {
    background: white;
    border-radius: 14px;
    padding: 24px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 14px rgba(0,0,0,0.05);
    border: 1px solid #e8eef4;
}

/* Submit button */
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #1a6fb5, #2e86c1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
    margin-top: 8px;
    box-shadow: 0 4px 12px rgba(46,134,193,0.3);
}
div.stFormSubmitButton > button:hover { opacity: 0.88 !important; }

/* Regular button */
div.stButton > button {
    background: linear-gradient(135deg, #1a6fb5, #2e86c1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    box-shadow: 0 4px 12px rgba(46,134,193,0.25);
}

/* Metric */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 18px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 14px rgba(0,0,0,0.05);
    border: 1px solid #e8eef4;
}
[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 700 !important; }

/* Panel card */
.panel {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 14px rgba(0,0,0,0.05);
    border: 1px solid #e8eef4;
}

/* Placeholder */
.empty-state {
    background: #f8fafc;
    border: 2px dashed #d1dce8;
    border-radius: 14px;
    padding: 56px 24px;
    text-align: center;
    color: #8a9bb0;
}
.empty-state .icon  { font-size: 52px; margin-bottom: 14px; }
.empty-state .title { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
.empty-state .sub   { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 10px 0 24px;">
    <div style="font-size:46px;">🏥</div>
    <div style="font-size:17px; font-weight:700; margin-top:8px; line-height:1.3;">
        Healthcare<br>Intelligence
    </div>
    <div style="font-size:11px; opacity:0.4; margin-top:6px; letter-spacing:0.5px;">
        AI-POWERED CLINICAL TOOLS
    </div>
</div>
<hr style="margin-bottom:16px;">
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🔁 Readmission Prediction", "⚠️ High-Risk Identification", "📈 ED Demand Forecast"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="font-size:11px; opacity:0.4; text-align:center; padding:0 10px;">
    Fill in patient details and run prediction to see live results.
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_defaults(path):
    with open(path) as f:
        return json.load(f)

def predict_classification(model_path, defaults_path, user_inputs: dict) -> float:
    defaults = load_defaults(defaults_path)
    df = pd.DataFrame([{**defaults, **user_inputs}])
    return float(load_model(model_path).predict_proba(df)[0][1])


# ── Chart builders ────────────────────────────────────────────────────────────
def gauge_chart(prob: float) -> go.Figure:
    if prob < 0.3:
        color, label = "#38a169", "Low Risk"
    elif prob < 0.6:
        color, label = "#d69e2e", "Moderate Risk"
    else:
        color, label = "#e53e3e", "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Inter"}},
        title={"text": label, "font": {"size": 15, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#ccc",
                     "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "#c6f6d5"},
                {"range": [30, 60],  "color": "#fefcbf"},
                {"range": [60, 100], "color": "#fed7d7"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="white",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def factor_bar_chart(factors: dict, title: str = "Patient Risk Profile") -> go.Figure:
    items  = sorted(factors.items(), key=lambda x: x[1])
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    colors = ["#e53e3e" if v > 0.6 else "#d69e2e" if v > 0.3 else "#48bb78" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
        textfont=dict(size=12, family="Inter"),
        hovertemplate="%{y}: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Inter"), x=0),
        height=max(200, len(labels) * 38 + 60),
        margin=dict(t=50, b=10, l=10, r=70),
        xaxis=dict(range=[0, 1.18], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, family="Inter")),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def forecast_line_chart(dates, preds, lower, upper) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates) + list(reversed(dates)),
        y=list(upper) + list(reversed(lower)),
        fill="toself",
        fillcolor="rgba(39,174,96,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=True, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=preds,
        mode="lines+markers",
        line=dict(color="#27ae60", width=2.5),
        marker=dict(size=4, color="#27ae60"),
        name="Forecast",
        hovertemplate="<b>%{x}</b><br>Visits: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="ED Visit Forecast", font=dict(size=15, family="Inter")),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#f0f2f5",
                   tickfont=dict(size=11)),
        yaxis=dict(title="Expected Visits", showgrid=True, gridcolor="#f0f2f5",
                   tickfont=dict(size=11)),
        height=360, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=12)),
        margin=dict(t=60, b=40, l=50, r=20),
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def weekday_bar_chart(dates, preds) -> go.Figure:
    df_w = pd.DataFrame({"date": pd.to_datetime(dates), "visits": preds})
    df_w["weekday"] = df_w["date"].dt.day_name()
    order  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = df_w.groupby("weekday")["visits"].mean().reindex(order).fillna(0)
    colors = ["#e53e3e" if d in ["Saturday", "Sunday"] else "#2980b9" for d in order]

    fig = go.Figure(go.Bar(
        x=order, y=weekly.values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}" for v in weekly.values],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{x}: %{y:.0f} visits<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Avg Visits by Day of Week", font=dict(size=14, family="Inter")),
        height=270,
        margin=dict(t=50, b=30, l=40, r=20),
        yaxis=dict(showgrid=True, gridcolor="#f0f2f5", tickfont=dict(size=11)),
        xaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def empty_state(icon, title, sub):
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icon}</div>
        <div class="title">{title}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Readmission Prediction
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔁 Readmission Prediction":
    st.markdown("""
    <div class="hero" style="background:linear-gradient(135deg,#1a3c5e 0%,#2980b9 100%);">
        <h1>🔁 Readmission Prediction</h1>
        <p>Predict whether a patient will be readmitted within 30 days of discharge.</p>
    </div>
    """, unsafe_allow_html=True)

    model_path    = os.path.join(MODEL_DIR, "readmission_model.joblib")
    defaults_path = os.path.join(MODEL_DIR, "readmission_defaults.json")
    if not os.path.exists(model_path):
        st.warning("Model not trained yet. Run `python main.py` first.")
        st.stop()

    form_col, result_col = st.columns([1, 1.3], gap="large")

    with form_col:
        with st.form("readmission_form"):
            st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            age            = c1.number_input("Age", 18, 95, 60)
            gender         = c2.selectbox("Gender", ["Male", "Female", "Non-Binary"])
            insurance_type = st.selectbox("Insurance Type",
                ["Medicare", "Medicaid", "Private Insurance", "Self-Pay", "Uninsured"])

            st.markdown('<div class="section-label">Admission</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            admission_type   = c1.selectbox("Admission Type",
                ["Emergency", "Urgent", "Elective", "Observation"])
            admission_source = c2.selectbox("Admission Source",
                ["Emergency Department", "Physician Referral", "Transfer",
                 "Direct Admission", "Clinic"])
            los = st.number_input("Length of Stay (days)", 1, 90, 4)

            st.markdown('<div class="section-label">Clinical</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            charlson             = c1.number_input("Charlson Index", 0, 15, 2)
            num_prior_admissions = c2.number_input("Prior Admissions", 0, 20, 0)
            num_medications      = c3.number_input("Medications", 0, 40, 5)

            st.markdown('<div class="section-label">Comorbidities</div>', unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns(4)
            diabetes      = cc1.checkbox("Diabetes")
            hypertension  = cc2.checkbox("Hypertension")
            heart_disease = cc3.checkbox("Heart Disease")
            copd          = cc4.checkbox("COPD")

            submitted = st.form_submit_button("Run Prediction →", use_container_width=True)

        if submitted:
            user_inputs = {
                "age": age, "gender": gender, "insurance_type": insurance_type,
                "admission_type": admission_type, "admission_source": admission_source,
                "length_of_stay_days": los, "charlson_comorbidity_index": charlson,
                "num_prior_admissions": num_prior_admissions,
                "num_medications": num_medications,
                "has_diabetes": int(diabetes), "has_hypertension": int(hypertension),
                "has_heart_disease": int(heart_disease), "has_copd": int(copd),
            }
            prob = predict_classification(model_path, defaults_path, user_inputs)
            st.session_state.readmission_result = (prob, user_inputs)

    with result_col:
        result = st.session_state.readmission_result
        if result is None:
            st.markdown("<br><br>", unsafe_allow_html=True)
            empty_state("🔁", "Awaiting Prediction",
                        "Fill in patient details on the left and click Run Prediction.")
        else:
            prob, inputs = result

            # Gauge
            st.markdown("**Readmission Risk Score**")
            st.plotly_chart(gauge_chart(prob), use_container_width=True,
                            config={"displayModeBar": False})

            # Factor profile
            comorbidities = (inputs["has_diabetes"] + inputs["has_hypertension"] +
                             inputs["has_heart_disease"] + inputs["has_copd"])
            factors = {
                "Age":              min(inputs["age"] / 90, 1.0),
                "Length of Stay":   min(inputs["length_of_stay_days"] / 30, 1.0),
                "Charlson Index":   min(inputs["charlson_comorbidity_index"] / 10, 1.0),
                "Prior Admissions": min(inputs["num_prior_admissions"] / 10, 1.0),
                "Medications":      min(inputs["num_medications"] / 20, 1.0),
                "Comorbidities":    comorbidities / 4,
            }
            st.plotly_chart(
                factor_bar_chart(factors, "Patient Risk Profile"),
                use_container_width=True, config={"displayModeBar": False}
            )

            # Active risk flags
            flags = []
            if inputs["age"] >= 75:                      flags.append("Age ≥ 75")
            if inputs["length_of_stay_days"] >= 7:       flags.append("LOS ≥ 7 days")
            if inputs["charlson_comorbidity_index"] >= 3: flags.append("Charlson ≥ 3")
            if inputs["num_prior_admissions"] >= 3:      flags.append("≥ 3 Prior Admissions")
            if comorbidities >= 2:                       flags.append(f"{comorbidities} Comorbidities")
            if inputs["admission_type"] == "Emergency":  flags.append("Emergency Admission")

            if flags:
                st.markdown("**Active Risk Flags**")
                cols = st.columns(min(len(flags), 3))
                for i, flag in enumerate(flags):
                    cols[i % 3].error(f"⚑ {flag}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — High-Risk Identification
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ High-Risk Identification":
    st.markdown("""
    <div class="hero" style="background:linear-gradient(135deg,#7b241c 0%,#e53935 100%);">
        <h1>⚠️ High-Risk Identification</h1>
        <p>Identify patients at high risk based on clinical, social, and lifestyle factors.</p>
    </div>
    """, unsafe_allow_html=True)

    model_path    = os.path.join(MODEL_DIR, "high_risk_model.joblib")
    defaults_path = os.path.join(MODEL_DIR, "high_risk_defaults.json")
    if not os.path.exists(model_path):
        st.warning("Model not trained yet. Run `python training/high_risk.py` first.")
        st.stop()

    form_col, result_col = st.columns([1, 1.3], gap="large")

    with form_col:
        with st.form("high_risk_form"):
            st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            age            = c1.number_input("Age", 18, 95, 60)
            gender         = c2.selectbox("Gender", ["Male", "Female", "Non-Binary"])
            admission_type = st.selectbox("Admission Type",
                ["Emergency", "Urgent", "Elective", "Observation"])

            st.markdown('<div class="section-label">Clinical</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            charlson             = c1.number_input("Charlson Index", 0, 15, 2)
            num_prior_admissions = c2.number_input("Prior Admissions", 0, 20, 0)
            num_ed_visits        = c3.number_input("ED Visits", 0, 50, 0)

            st.markdown('<div class="section-label">Lifestyle & Social</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            social_support = c1.number_input("Social Support (0–10)", 0.0, 10.0, 5.0, step=0.5)
            smoking_status = c2.selectbox("Smoking", ["Never", "Former", "Current"])
            alcohol_use    = c3.selectbox("Alcohol Use", ["Unknown", "Moderate", "Heavy"])

            st.markdown('<div class="section-label">Comorbidities</div>', unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns(4)
            diabetes      = cc1.checkbox("Diabetes")
            hypertension  = cc2.checkbox("Hypertension")
            heart_disease = cc3.checkbox("Heart Disease")
            copd          = cc4.checkbox("COPD")

            submitted = st.form_submit_button("Run Prediction →", use_container_width=True)

        if submitted:
            user_inputs = {
                "age": age, "gender": gender, "admission_type": admission_type,
                "charlson_comorbidity_index": charlson,
                "num_prior_admissions": num_prior_admissions,
                "num_ed_visits": num_ed_visits,
                "social_support_score": social_support,
                "smoking_status": smoking_status, "alcohol_use": alcohol_use,
                "has_diabetes": int(diabetes), "has_hypertension": int(hypertension),
                "has_heart_disease": int(heart_disease), "has_copd": int(copd),
            }
            prob = predict_classification(model_path, defaults_path, user_inputs)
            st.session_state.high_risk_result = (prob, user_inputs)

    with result_col:
        result = st.session_state.high_risk_result
        if result is None:
            st.markdown("<br><br>", unsafe_allow_html=True)
            empty_state("⚠️", "Awaiting Prediction",
                        "Fill in patient details on the left and click Run Prediction.")
        else:
            prob, inputs = result

            st.markdown("**High-Risk Score**")
            st.plotly_chart(gauge_chart(prob), use_container_width=True,
                            config={"displayModeBar": False})

            comorbidities = (inputs["has_diabetes"] + inputs["has_hypertension"] +
                             inputs["has_heart_disease"] + inputs["has_copd"])
            smoke_map   = {"Never": 0.0, "Former": 0.4, "Current": 1.0}
            alcohol_map = {"Unknown": 0.0, "Moderate": 0.4, "Heavy": 1.0}
            factors = {
                "Age":                min(inputs["age"] / 90, 1.0),
                "Charlson Index":     min(inputs["charlson_comorbidity_index"] / 10, 1.0),
                "Prior Admissions":   min(inputs["num_prior_admissions"] / 10, 1.0),
                "ED Visits":          min(inputs["num_ed_visits"] / 20, 1.0),
                "Low Social Support": 1 - (inputs["social_support_score"] / 10),
                "Comorbidities":      comorbidities / 4,
                "Smoking":            smoke_map[inputs["smoking_status"]],
                "Alcohol Use":        alcohol_map[inputs["alcohol_use"]],
            }
            st.plotly_chart(
                factor_bar_chart(factors, "Patient Risk Profile"),
                use_container_width=True, config={"displayModeBar": False}
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Charlson Index",   inputs["charlson_comorbidity_index"])
            m2.metric("Prior ED Visits",  inputs["num_ed_visits"])
            m3.metric("Social Support",   f"{inputs['social_support_score']:.1f} / 10")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ED Demand Forecast
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 ED Demand Forecast":
    st.markdown("""
    <div class="hero" style="background:linear-gradient(135deg,#145a32 0%,#27ae60 100%);">
        <h1>📈 ED Demand Forecast</h1>
        <p>Interactive forecast of Emergency Department visit volumes over any future period.</p>
    </div>
    """, unsafe_allow_html=True)

    model_path     = os.path.join(MODEL_DIR, "xgb_demand_model.joblib")
    seed_path      = os.path.join(MODEL_DIR, "xgb_seed_values.npy")
    last_date_path = os.path.join(MODEL_DIR, "demand_last_date.txt")
    if not os.path.exists(model_path):
        st.warning("XGBoost model not trained yet. Run `python training/demand_forecast.py` first.")
        st.stop()

    with open(last_date_path) as f:
        last_train_date = pd.Timestamp(f.read().strip())

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 0.8])
    forecast_days      = ctrl1.slider("Forecast horizon (days)", 7, 180, 30, step=7)
    highlight_weekends = ctrl2.toggle("Highlight weekends", value=True)
    run = ctrl3.button("Generate Forecast →", use_container_width=True)

    if run:
        future_dates = pd.date_range(
            last_train_date + timedelta(days=1), periods=forecast_days, freq="D"
        )
        # Iterative forecasting: predict one day at a time using lag features
        # Each prediction becomes the next day's lag_1
        xgb_model   = load_model(model_path)
        seed_values = np.load(seed_path)   # last 7 real training values
        history     = list(seed_values)
        preds       = []
        for _ in range(forecast_days):
            lag_1          = history[-1]
            lag_7          = history[-7]
            rolling_mean_7 = float(np.mean(history[-7:]))
            pred = float(xgb_model.predict([[lag_1, lag_7, rolling_mean_7]])[0])
            pred = max(0.0, pred)   # visits can't be negative
            preds.append(pred)
            history.append(pred)
        st.session_state.ed_result = {
            "dates": [str(d.date()) for d in future_dates],
            "preds": preds,
            "lower": preds,   # XGBoost has no built-in confidence interval
            "upper": preds,
            "days":  forecast_days,
        }

    ed = st.session_state.ed_result
    if ed is None:
        st.markdown("<br>", unsafe_allow_html=True)
        empty_state("📈", "No Forecast Yet",
                    "Choose a forecast horizon above and click Generate Forecast.")
    else:
        dates = ed["dates"]
        preds = ed["preds"]
        lower = ed["lower"]
        upper = ed["upper"]

        # KPI strip
        avg_visits  = round(float(np.mean(preds)))
        peak_idx    = int(np.argmax(preds))
        peak_visits = round(preds[peak_idx])
        peak_date   = dates[peak_idx]
        total       = round(sum(preds))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Forecast Period",   f"{ed['days']} days")
        m2.metric("Avg Daily Visits",  avg_visits)
        m3.metric("Peak Visits",       peak_visits, delta=f"on {peak_date}")
        m4.metric("Total Estimated",   f"{total:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Forecast chart
        fig = forecast_line_chart(dates, preds, lower, upper)
        if highlight_weekends:
            for d in pd.to_datetime(dates):
                if d.weekday() >= 5:
                    fig.add_vrect(
                        x0=str(d.date()), x1=str(d.date()),
                        fillcolor="rgba(0,0,0,0.04)", line_width=0, layer="below"
                    )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        # Weekday chart + table
        chart_col, table_col = st.columns([1.2, 1], gap="large")
        with chart_col:
            st.plotly_chart(
                weekday_bar_chart(dates, preds),
                use_container_width=True, config={"displayModeBar": False}
            )
        with table_col:
            st.markdown("**Forecast Table**")
            df_table = pd.DataFrame({
                "Date":     dates,
                "Day":      [pd.Timestamp(d).day_name()[:3] for d in dates],
                "Forecast": [round(v) for v in preds],
                "Low":      [round(v) for v in lower],
                "High":     [round(v) for v in upper],
            })
            st.dataframe(
                df_table.style.background_gradient(subset=["Forecast"], cmap="Greens"),
                use_container_width=True, height=300, hide_index=True,
            )




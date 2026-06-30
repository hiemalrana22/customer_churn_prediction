"""
Streamlit churn cockpit.

Pick a customer, get their calibrated churn probability, the factors driving it,
a recommended action, and the dollars on the table if we save them. This is the
thing a retention team would actually open - the notebooks and modules are the
engine, this is the dashboard bolted on top.

Run locally:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.business_impact import CampaignAssumptions
from src.calibrate import fit_for_serving
from src.data_loader import load_from_sqlite
from src.explain import explain_customer
from src.model import prepare_xy

st.set_page_config(page_title="Churn Cockpit", page_icon="📉", layout="wide")

ASSUMPTIONS = CampaignAssumptions()


@st.cache_resource(show_spinner="Training model...")
def _load():
    df = load_from_sqlite()
    calibrated, base = fit_for_serving(df)
    X, _ = prepare_xy(df)
    probs = calibrated.predict_proba(X)[:, 1]
    df = df.assign(churn_prob=probs)
    return df, X, calibrated, base


def risk_band(p: float) -> tuple:
    if p >= 0.7:
        return "High", "🔴"
    if p >= 0.4:
        return "Medium", "🟠"
    return "Low", "🟢"


def recommend(p: float, drivers: pd.DataFrame) -> str:
    """A plain-English next step keyed off risk and the top driver."""
    if p < 0.4:
        return "Low risk. No active intervention needed - keep on standard service."

    top = drivers.iloc[0]["feature"] if len(drivers) else ""
    if "Month-to-month" in top:
        return "Offer a 1-year contract with a loyalty discount - the month-to-month plan is the biggest risk lever."
    if "Fiber optic" in top:
        return "Proactive service-quality check + bundle tech support; fiber customers churn on reliability."
    if top.startswith("tenure"):
        return "Early-life customer. Assign onboarding outreach in the first 90 days."
    if "OnlineSecurity" in top or "TechSupport" in top:
        return "Bundle a free trial of security/tech-support add-ons - missing services drive this risk."
    return "Route to retention team for a targeted save offer."


df, X, calibrated, base = _load()

st.title("Customer Churn Cockpit")
st.caption("Calibrated risk scores, SHAP-driven explanations, and the dollar case for intervention.")

# ---- sidebar: customer picker ----
with st.sidebar:
    st.header("Pick a customer")
    sort_high = st.checkbox("Sort by highest risk", value=True)
    ids = df.sort_values("churn_prob", ascending=not sort_high)["customerID"].tolist()
    cid = st.selectbox("Customer ID", ids)

    st.divider()
    st.subheader("Campaign assumptions")
    cost = st.number_input("Intervention cost ($)", 0, 500, int(ASSUMPTIONS.intervention_cost), 10)
    uplift = st.slider("Save rate (uplift)", 0.0, 1.0, ASSUMPTIONS.uplift, 0.05)
    horizon = st.slider("Revenue horizon (months)", 1, 36, ASSUMPTIONS.horizon_months)

row = df[df["customerID"] == cid].iloc[0]
x_row = X[df["customerID"] == cid]
prob = float(row["churn_prob"])
band, dot = risk_band(prob)

# ---- top line: probability + economics ----
monthly = float(row["MonthlyCharges"])
value_at_risk = monthly * horizon
projected_saved = prob * uplift * value_at_risk

c1, c2, c3, c4 = st.columns(4)
c1.metric("Churn probability", f"{prob:.0%}", f"{dot} {band} risk")
c2.metric("Monthly charge", f"${monthly:,.0f}")
c3.metric(f"Value at risk ({horizon}mo)", f"${value_at_risk:,.0f}")
c4.metric("Projected $ saved", f"${projected_saved:,.0f}",
          help="prob × save-rate × value at risk, minus the intervention cost below")

st.progress(min(prob, 1.0))

# ---- explanation + action ----
left, right = st.columns([3, 2])

with left:
    st.subheader("Why this customer?")
    drivers = explain_customer(base, x_row, top_n=6)
    chart = drivers.assign(impact=drivers["shap_value"]).set_index("feature")["impact"]
    st.bar_chart(chart, horizontal=True, color="#2E86AB")
    st.caption("SHAP values in log-odds. Positive bars push churn risk up, negative pull it down.")

with right:
    st.subheader("Recommended action")
    st.info(recommend(prob, drivers))
    net = projected_saved - cost
    st.metric("Net benefit if we act", f"${net:,.0f}",
              f"ROI {net / cost:.1f}x" if cost else "")
    st.write("Customer snapshot")
    st.dataframe(
        row[["Contract", "tenure", "InternetService", "PaymentMethod",
             "OnlineSecurity", "TechSupport"]].rename("value").to_frame(),
        use_container_width=True,
    )

# ---- portfolio view ----
st.divider()
st.subheader("Portfolio: where the risk (and money) sits")
book = df.assign(
    value_at_risk=df["MonthlyCharges"] * horizon,
    expected_loss=lambda d: d["churn_prob"] * d["MonthlyCharges"] * horizon,
)
pc1, pc2, pc3 = st.columns(3)
pc1.metric("Customers", f"{len(book):,}")
pc2.metric("Avg churn probability", f"{book['churn_prob'].mean():.0%}")
pc3.metric("Total expected revenue at risk", f"${book['expected_loss'].sum():,.0f}")

st.bar_chart(
    pd.cut(book["churn_prob"], bins=np.linspace(0, 1, 11)).value_counts().sort_index(),
    color="#A23B72",
)
st.caption("Distribution of calibrated churn probabilities across the customer base.")

"""Python-only Flask dashboard deployed as a Vercel serverless function."""

from functools import lru_cache
from html import escape

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from src.calibrate import fit_for_serving
from src.data_loader import load_for_serverless
from src.explain import explain_customer
from src.model import prepare_xy

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_dashboard_data():
    """Train once per warm serverless instance and cache the scored customer book."""
    df = load_for_serverless()
    calibrated, base = fit_for_serving(df)
    X, _ = prepare_xy(df)
    return df.assign(churn_prob=calibrated.predict_proba(X)[:, 1]), X, base


def risk_band(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "High", "high"
    if probability >= 0.4:
        return "Medium", "medium"
    return "Low", "low"


def recommend(probability: float, drivers: pd.DataFrame) -> str:
    if probability < 0.4:
        return "Low risk. No active intervention needed; keep this customer on standard service."
    top = drivers.iloc[0]["feature"] if len(drivers) else ""
    if "Month-to-month" in top:
        return "Offer a one-year contract with a loyalty discount; the month-to-month plan is the biggest risk lever."
    if "Fiber optic" in top:
        return "Schedule a proactive service-quality check and bundle tech support for this fiber customer."
    if top.startswith("tenure"):
        return "Assign onboarding outreach during the customer's first 90 days."
    if "OnlineSecurity" in top or "TechSupport" in top:
        return "Offer a free trial of security or tech-support add-ons."
    return "Route to the retention team for a targeted save offer."


def money(value: float) -> str:
    return f"${value:,.0f}"


def number_arg(name: str, default: float, minimum: float, maximum: float, cast=float):
    try:
        value = cast(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/")
@app.get("/api")
def dashboard():
    df, X, base = load_dashboard_data()
    cost = number_arg("cost", 50, 0, 500, int)
    uplift = number_arg("uplift", 0.30, 0, 1)
    horizon = number_arg("horizon", 12, 1, 36, int)
    requested_id = request.args.get("customer_id")
    row = df.loc[df["customerID"] == requested_id].iloc[0] if requested_id in set(df["customerID"]) else df.loc[df["churn_prob"].idxmax()]
    customer_id = row["customerID"]
    x_row = X.loc[df["customerID"] == customer_id]
    probability = float(row["churn_prob"])
    band, band_class = risk_band(probability)
    drivers = explain_customer(base, x_row, top_n=6)
    monthly = float(row["MonthlyCharges"])
    value_at_risk = monthly * horizon
    projected_saved = probability * uplift * value_at_risk
    net = projected_saved - cost
    book = df.assign(expected_loss=lambda d: d["churn_prob"] * d["MonthlyCharges"] * horizon)
    top_customers = df.nlargest(20, "churn_prob")

    options = "".join(
        f'<option value="{escape(cid)}"{" selected" if cid == customer_id else ""}>{escape(cid)} — {prob:.0%}</option>'
        for cid, prob in zip(top_customers["customerID"], top_customers["churn_prob"])
    )
    driver_rows = "".join(
        f"<tr><td>{escape(str(item.feature))}</td><td class=\"{('up' if item.shap_value >= 0 else 'down')}\">{item.shap_value:+.2f}</td><td>{escape(item.direction)}</td></tr>"
        for item in drivers.itertuples(index=False)
    )
    snapshot_fields = ("Contract", "tenure", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport")
    snapshot_rows = "".join(f"<tr><th>{field}</th><td>{escape(str(row[field]))}</td></tr>" for field in snapshot_fields)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Customer Churn Cockpit</title><style>
:root{{color-scheme:light;--ink:#172033;--muted:#61708a;--panel:#fff;--bg:#f4f7fb;--blue:#1677c8;--red:#d43c4e;--orange:#cc7510;--green:#168568}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:36px 20px 60px}}h1{{margin:0;font-size:2.2rem}}h2{{font-size:1.1rem;margin-top:0}}.subtitle,.muted{{color:var(--muted)}}form,.card{{background:var(--panel);border-radius:12px;padding:20px;box-shadow:0 2px 12px #15203b12}}form{{display:grid;grid-template-columns:2fr repeat(3,1fr) auto;gap:12px;align-items:end;margin:26px 0}}label{{font-size:.82rem;font-weight:bold;display:grid;gap:6px}}input,select,button{{padding:10px;border-radius:7px;border:1px solid #ccd6e3;background:#fff}}button{{background:var(--blue);color:#fff;border:0;font-weight:bold;cursor:pointer}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}}.metric{{background:var(--panel);padding:18px;border-radius:12px;box-shadow:0 2px 12px #15203b12}}.metric strong{{display:block;font-size:1.65rem;margin-top:6px}}.risk{{font-weight:bold}}.high{{color:var(--red)}}.medium{{color:var(--orange)}}.low{{color:var(--green)}}.grid{{display:grid;grid-template-columns:3fr 2fr;gap:18px;margin-bottom:18px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px 4px;border-bottom:1px solid #e7edf5;text-align:left;font-size:.9rem}}th{{color:var(--muted)}}.up{{color:var(--red);font-weight:bold}}.down{{color:var(--green);font-weight:bold}}.action{{line-height:1.55;border-left:4px solid var(--blue);padding-left:14px}}@media(max-width:760px){{form,.metrics,.grid{{grid-template-columns:1fr}}main{{padding:24px 14px}}}}
</style></head><body><main>
<h1>Customer Churn Cockpit</h1><p class="subtitle">Calibrated risk scores, model-driven explanations, and the dollar case for intervention.</p>
<form method="get"><label>Customer (top 20 by risk)<select name="customer_id">{options}</select></label><label>Intervention cost ($)<input name="cost" type="number" min="0" max="500" value="{cost}"></label><label>Save rate<input name="uplift" type="number" min="0" max="1" step="0.05" value="{uplift:.2f}"></label><label>Revenue horizon (months)<input name="horizon" type="number" min="1" max="36" value="{horizon}"></label><button type="submit">Update analysis</button></form>
<section class="metrics"><div class="metric"><span>Churn probability</span><strong>{probability:.0%}</strong><span class="risk {band_class}">{band} risk</span></div><div class="metric"><span>Monthly charge</span><strong>{money(monthly)}</strong></div><div class="metric"><span>Value at risk ({horizon} mo)</span><strong>{money(value_at_risk)}</strong></div><div class="metric"><span>Projected saved</span><strong>{money(projected_saved)}</strong><span class="muted">gross expected value</span></div></section>
<section class="grid"><div class="card"><h2>Why this customer?</h2><table><thead><tr><th>Driver</th><th>Impact</th><th>Effect</th></tr></thead><tbody>{driver_rows}</tbody></table><p class="muted">Positive impact raises churn risk; negative impact lowers it.</p></div><div class="card"><h2>Recommended action</h2><p class="action">{escape(recommend(probability, drivers))}</p><h2>Net benefit if we act</h2><strong style="font-size:1.7rem">{money(net)}</strong><p class="muted">ROI {net / cost:.1f}x</p><h2>Customer snapshot</h2><table>{snapshot_rows}</table></div></section>
<section class="metrics"><div class="metric"><span>Customers</span><strong>{len(book):,}</strong></div><div class="metric"><span>Average churn probability</span><strong>{book['churn_prob'].mean():.0%}</strong></div><div class="metric"><span>Total expected revenue at risk</span><strong>{money(book['expected_loss'].sum())}</strong></div></section>
</main></body></html>"""

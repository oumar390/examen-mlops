"""Streamlit UI for the credit scoring API.

Provides a single-page form to score a credit application and visualises
the result with a gauge, risk badge, and approve/reject decision.

API URL is read from the API_URL env var (defaults to http://localhost:8000).
"""
from __future__ import annotations

import os
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring — UCI Credit Card Default",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Scoring")
st.caption("Modèle XGBoost entraîné sur UCI Credit Card Default (Taiwan).")

# ---------------------------------------------------------------------------
# Sidebar — model info & connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Service")
    st.code(API_URL, language="text")

    if st.button("🔁 Refresh model info", use_container_width=True):
        st.rerun()

    try:
        info = requests.get(f"{API_URL}/model/info", timeout=5).json()
        st.success("API connectée")
        st.metric("Modèle", info["model_name"])
        st.metric("Seuil de décision", f"{info['threshold']:.2f}")
        st.metric(
            "Business gain (test)",
            f"{info['test_metrics']['test_business_gain']:.3f}",
        )
        with st.expander("Détails complets"):
            st.json(info)
    except Exception as exc:  # noqa: BLE001
        st.error(f"API injoignable : {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.header("👤 Profil du client")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Démographie")
    age = st.slider("Âge", 18, 80, 35)
    sex = st.selectbox("Sexe", options=[1, 2], format_func=lambda x: {1: "Homme", 2: "Femme"}[x])
    education = st.selectbox(
        "Éducation",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "École supérieure",
            2: "Université",
            3: "Lycée",
            4: "Autres",
        }[x],
    )
    marriage = st.selectbox(
        "Statut marital",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Marié", 2: "Célibataire", 3: "Autres"}[x],
    )
    limit_bal = st.number_input(
        "Plafond de crédit (NT$)",
        min_value=10_000,
        max_value=1_000_000,
        value=200_000,
        step=10_000,
    )

with col2:
    st.subheader("Historique de paiement")
    st.caption("PAY : -2=no use, -1=paid duly, 0=revolving, 1+=retard en mois")
    pay_0 = st.slider("Septembre (PAY_0)", -2, 8, 0)
    pay_2 = st.slider("Août (PAY_2)", -2, 8, 0)
    pay_3 = st.slider("Juillet (PAY_3)", -2, 8, 0)
    pay_4 = st.slider("Juin (PAY_4)", -2, 8, 0)
    pay_5 = st.slider("Mai (PAY_5)", -2, 8, 0)
    pay_6 = st.slider("Avril (PAY_6)", -2, 8, 0)

with col3:
    st.subheader("Factures (BILL_AMT)")
    bill_1 = st.number_input("Sept (BILL_AMT1)", value=50_000, step=1_000)
    bill_2 = st.number_input("Août (BILL_AMT2)", value=48_000, step=1_000)
    bill_3 = st.number_input("Juillet (BILL_AMT3)", value=45_000, step=1_000)
    bill_4 = st.number_input("Juin (BILL_AMT4)", value=42_000, step=1_000)
    bill_5 = st.number_input("Mai (BILL_AMT5)", value=40_000, step=1_000)
    bill_6 = st.number_input("Avril (BILL_AMT6)", value=38_000, step=1_000)

with st.expander("💵 Montants payés mensuellement (PAY_AMT)"):
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    pay_amt_1 = p1.number_input("Sept", value=5_000, step=500, min_value=0)
    pay_amt_2 = p2.number_input("Août", value=5_000, step=500, min_value=0)
    pay_amt_3 = p3.number_input("Juillet", value=5_000, step=500, min_value=0)
    pay_amt_4 = p4.number_input("Juin", value=5_000, step=500, min_value=0)
    pay_amt_5 = p5.number_input("Mai", value=5_000, step=500, min_value=0)
    pay_amt_6 = p6.number_input("Avril", value=5_000, step=500, min_value=0)

# Pre-built sample profiles
st.subheader("⚡ Profils pré-remplis")
preset_col1, preset_col2, _ = st.columns([1, 1, 4])

preset_low = preset_col1.button("Bon payeur", use_container_width=True)
preset_high = preset_col2.button("Mauvais payeur", use_container_width=True)

if preset_low:
    st.session_state["preset"] = "low"
    st.rerun()
if preset_high:
    st.session_state["preset"] = "high"
    st.rerun()


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
st.markdown("---")
predict_clicked = st.button("🔮 Prédire le risque de défaut", type="primary", use_container_width=True)

if predict_clicked:
    payload = {
        "LIMIT_BAL": float(limit_bal),
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
        "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
        "BILL_AMT1": float(bill_1), "BILL_AMT2": float(bill_2),
        "BILL_AMT3": float(bill_3), "BILL_AMT4": float(bill_4),
        "BILL_AMT5": float(bill_5), "BILL_AMT6": float(bill_6),
        "PAY_AMT1": float(pay_amt_1), "PAY_AMT2": float(pay_amt_2),
        "PAY_AMT3": float(pay_amt_3), "PAY_AMT4": float(pay_amt_4),
        "PAY_AMT5": float(pay_amt_5), "PAY_AMT6": float(pay_amt_6),
    }

    with st.spinner("Calcul en cours…"):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Échec de la prédiction : {exc}")
            st.stop()

    # --- Display ---
    proba = result["probability_default"]
    threshold = result["threshold"]
    risk = result["risk_level"]
    decision = result["label"]

    risk_color = {
        "low": "#1ABC9C",
        "medium": "#F39C12",
        "high": "#E67E22",
        "very_high": "#C0392B",
    }[risk]

    risk_label = {
        "low": "Risque faible",
        "medium": "Risque modéré",
        "high": "Risque élevé",
        "very_high": "Risque très élevé",
    }[risk]

    decision_emoji = "✅" if decision == "approve" else "🛑"
    decision_label = "ACCORDER LE CRÉDIT" if decision == "approve" else "REFUSER LE CRÉDIT"
    decision_color = "#2ECC71" if decision == "approve" else "#E74C3C"

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Probabilité de défaut", f"{proba*100:.1f} %")
    k2.metric("Seuil de décision", f"{threshold*100:.0f} %")
    k3.metric("Niveau de risque", risk_label)
    k4.metric("Décision", decision_label)

    # Gauge
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": " %"},
            title={"text": "Probabilité de défaut"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
                "steps": [
                    {"range": [0, 20], "color": "#D5F5E3"},
                    {"range": [20, 40], "color": "#FCF3CF"},
                    {"range": [40, 65], "color": "#F5B041"},
                    {"range": [65, 100], "color": "#F1948A"},
                ],
            },
        )
    )
    gauge.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)

    # Decision banner
    st.markdown(
        f"""
        <div style="background-color:{decision_color}; padding: 20px; border-radius: 10px;
                    text-align: center; color: white; font-size: 24px; font-weight: bold;">
            {decision_emoji} {decision_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Détails techniques de la réponse"):
        st.json(result)

# Footer
st.markdown("---")
st.caption(
    f"⏱️  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "
    f"Modèle XGBoost, seuil métier optimisé (FN×5 = FP)."
)

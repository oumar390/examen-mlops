"""Streamlit UI for the credit scoring API.

Designed for ease of testing: the user does NOT have to fill 23 fields
by hand. Three workflows are available:

1. **One-click presets** — Excellent / Good / At-risk / Bad payer
2. **Random** — picks a random preset
3. **Manual edit** — once a preset is loaded, every field is still editable

The 23 raw inputs are organised into 3 collapsible sections to keep
the page short.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
PROFILES_PATH = Path(__file__).parent / "profiles.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring — UCI Credit Card Default",
    page_icon="💳",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Load profile catalog
# ---------------------------------------------------------------------------
@st.cache_data
def load_profiles() -> dict:
    return json.loads(PROFILES_PATH.read_text())


PROFILES = load_profiles()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Initialise st.session_state with the default profile (once)."""
    if "profile_data" not in st.session_state:
        st.session_state.profile_data = PROFILES["default"]["data"].copy()
        st.session_state.profile_name = "default"


def load_profile(name: str) -> None:
    """Replace current form data with the chosen preset."""
    st.session_state.profile_data = PROFILES[name]["data"].copy()
    st.session_state.profile_name = name


def load_random_profile() -> None:
    """Pick a random preset (excluding the neutral default)."""
    pool = [k for k in PROFILES if k != "default"]
    name = random.choice(pool)
    load_profile(name)


init_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💳 Credit Scoring")
st.caption("Modèle XGBoost entraîné sur UCI Credit Card Default (Taiwan).")


# ---------------------------------------------------------------------------
# Sidebar — service info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Service")
    st.code(API_URL, language="text")
    if st.button("🔁 Refresh", use_container_width=True):
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
# Quick presets — the star UX feature
# ---------------------------------------------------------------------------
st.markdown("### 🎯 Charger un profil-type")
st.caption(
    "Évite de remplir les 23 champs : un clic suffit. Tous les champs "
    "restent éditables ensuite si tu veux ajuster."
)

preset_cols = st.columns(5)
preset_buttons = [
    ("excellent", "🟢 Excellent"),
    ("good", "🔵 Bon"),
    ("at_risk", "🟡 À risque"),
    ("bad", "🔴 Mauvais"),
    ("random", "🎲 Aléatoire"),
]
for col, (key, label) in zip(preset_cols, preset_buttons):
    if col.button(label, use_container_width=True, key=f"btn_{key}"):
        if key == "random":
            load_random_profile()
        else:
            load_profile(key)
        st.rerun()

# Show currently loaded profile
current = PROFILES.get(st.session_state.profile_name, PROFILES["default"])
st.info(f"**Profil actif :** {current['label']} — {current['description']}")

# Convenience: shortcut to access current values
d = st.session_state.profile_data


# ---------------------------------------------------------------------------
# Form (3 expanders to keep page compact)
# ---------------------------------------------------------------------------
st.markdown("### 📝 Données du client")
st.caption("Tous les champs sont pré-remplis depuis le profil. Modifie ce que tu veux.")

with st.expander("👤 Démographie", expanded=False):
    c1, c2, c3 = st.columns(3)
    d["LIMIT_BAL"] = float(c1.number_input(
        "Plafond de crédit (NT$)",
        min_value=10_000, max_value=1_000_000,
        value=int(d["LIMIT_BAL"]), step=10_000,
    ))
    d["AGE"] = c1.slider("Âge", 18, 80, int(d["AGE"]))
    d["SEX"] = c2.selectbox(
        "Sexe",
        options=[1, 2],
        format_func=lambda x: {1: "Homme", 2: "Femme"}[x],
        index=[1, 2].index(d["SEX"]),
    )
    d["EDUCATION"] = c2.selectbox(
        "Éducation",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "École supérieure", 2: "Université",
            3: "Lycée", 4: "Autres"
        }[x],
        index=[1, 2, 3, 4].index(d["EDUCATION"]),
    )
    d["MARRIAGE"] = c3.selectbox(
        "Statut marital",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Marié", 2: "Célibataire", 3: "Autres"}[x],
        index=[1, 2, 3].index(d["MARRIAGE"]),
    )

with st.expander("💳 Historique de paiement (6 mois)", expanded=True):
    st.caption("PAY: −2=non utilisé · −1=payé à temps · 0=revolving · 1+=retard en mois")
    pay_cols = st.columns(6)
    pay_labels = [
        ("PAY_0", "Sept"), ("PAY_2", "Août"), ("PAY_3", "Juillet"),
        ("PAY_4", "Juin"), ("PAY_5", "Mai"), ("PAY_6", "Avril"),
    ]
    for col, (key, label) in zip(pay_cols, pay_labels):
        d[key] = col.slider(label, -2, 8, int(d[key]), key=f"slider_{key}")

with st.expander("💰 Montants mensuels (factures et paiements)", expanded=False):
    st.caption("BILL_AMT = facture émise · PAY_AMT = montant effectivement payé")
    months = ["Sept", "Août", "Juillet", "Juin", "Mai", "Avril"]

    st.markdown("**Factures (NT$)**")
    bill_cols = st.columns(6)
    for i, (col, month) in enumerate(zip(bill_cols, months), start=1):
        key = f"BILL_AMT{i}"
        d[key] = float(col.number_input(
            month, value=int(d[key]), step=1_000, key=f"bill_{i}"
        ))

    st.markdown("**Paiements (NT$)**")
    pay_amt_cols = st.columns(6)
    for i, (col, month) in enumerate(zip(pay_amt_cols, months), start=1):
        key = f"PAY_AMT{i}"
        d[key] = float(col.number_input(
            month, value=int(d[key]), step=500, min_value=0, key=f"payamt_{i}"
        ))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.markdown("---")

predict_clicked = st.button(
    "🔮 Prédire le risque de défaut",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    payload = st.session_state.profile_data
    with st.spinner("Calcul en cours…"):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Échec de la prédiction : {exc}")
            st.stop()

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
    "Modèle XGBoost · seuil métier optimisé (FN×5 = FP)."
)

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


st.set_page_config(page_title="Scoring cancer risk", page_icon=":bar_chart:", layout="wide")
st.title("Interface de scoring")

try:
    example = requests.get(f"{API_URL}/example", timeout=5).json()["features"]
except requests.RequestException as exc:
    st.error(f"API indisponible: {exc}")
    st.stop()

with st.form("scoring-form"):
    columns = st.columns(3)
    features: dict[str, float] = {}
    for index, (name, value) in enumerate(example.items()):
        with columns[index % 3]:
            min_value = max(0.0, float(value) * 0.25)
            max_value = max(float(value) * 2.5, min_value + 1.0)
            features[name] = st.number_input(
                name.replace("_", " "),
                min_value=min_value,
                max_value=max_value,
                value=float(value),
            )
    submitted = st.form_submit_button("Calculer le score")

if submitted:
    response = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=10)
    if response.ok:
        result = response.json()
        st.metric("Probabilité haut risque", f"{result['probability_high_risk']:.1%}")
        st.write(f"Classe prédite: **{result['label']}**")
    else:
        st.error(response.text)

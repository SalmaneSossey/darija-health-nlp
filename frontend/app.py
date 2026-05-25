from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
EXAMPLES = [
    "3ndi sda3 w dwakha",
    "عندي ألم فالصدر وضيق فالتنفس",
    "j’ai mal au ventre depuis 3 jours",
    "waldi 3ndo skhana w k7a",
]


st.set_page_config(page_title="Darija Health NLP", page_icon="DH", layout="centered")
st.title("Darija Health NLP")
st.subheader("Moroccan Medical Triage Assistant")
st.warning(
    "This tool is for academic and educational orientation only. It does not provide diagnosis and does not replace a doctor."
)

selected = st.selectbox("Example messages", [""] + EXAMPLES)
default_message = selected or "3ndi wje3 f sedri w di9 f nefs"
message = st.text_area("Patient message", value=default_message, height=130)

if st.button("Analyze", type="primary"):
    try:
        response = requests.post(API_URL, json={"message": message}, timeout=15)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
    else:
        st.markdown("### Results")
        col1, col2 = st.columns(2)
        col1.metric("Specialty", result["predicted_specialty"])
        col2.metric("Urgency", result["urgency"])
        st.write("Symptoms:", ", ".join(result["symptoms"]) if result["symptoms"] else "No canonical symptoms detected")
        st.info(result["recommendation"])
        st.caption(result["disclaimer"])

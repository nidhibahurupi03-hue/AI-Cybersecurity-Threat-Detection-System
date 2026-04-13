import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")

st.title("🚀 AI Cybersecurity Threat Detection System")

st.write("Upload CSV or use default dataset")

uploaded_file = st.file_uploader("Upload Network CSV", type=["csv"])

# =========================
# LOAD DATA SAFELY
# =========================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 🔥 FIX: limit size
    if len(df) > 50000:
        st.error("❌ File too large! Please upload smaller dataset (<= 50k rows)")
        st.stop()

    st.success("File uploaded successfully!")

else:
    st.info("Using default dataset")
    df = pd.read_csv("data/network_data.csv")
    df = df.sample(min(5000, len(df)))

st.write("Dataset Preview:")
st.dataframe(df.head())

# =========================
# SHOW OUTPUT GRAPHS
# =========================

st.header("📊 Model Results")

cm = "output/graph/confusion_matrix.png"
cmn = "output/graph/confusion_matrix_normalized.png"
roc = "output/graph/roc_curve.png"

if os.path.exists(cm):
    st.image(cm, caption="Confusion Matrix")

if os.path.exists(cmn):
    st.image(cmn, caption="Normalized Confusion Matrix")

if os.path.exists(roc):
    st.image(roc, caption="ROC Curve")

st.success("Dashboard Loaded Successfully 🚀")
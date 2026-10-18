"""Streamlit dashboard for Customer Spending Prediction.
Run from the project root: streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
SUMMARY_PATH = ROOT / "reports" / "project_summary.json"
METRICS_PATH = ROOT / "reports" / "model_comparison.csv"

st.set_page_config(page_title="Customer Spending Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Spending Prediction")
st.write("A machine-learning dashboard that estimates customer spending from demographic and purchase-behaviour features.")
st.caption("Academic demonstration only. Predictions are estimates, not financial advice or guaranteed outcomes.")

if not MODEL_PATH.exists():
    st.warning("Trained model not found. Run `python train_model.py` in the project folder first.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
m1, m2, m3 = st.columns(3)
m1.metric("Customers in dataset", f"{len(df):,}")
m2.metric("Average spending", f"{df['spending'].mean():,.2f}")
m3.metric("Best model R²", f"{summary.get('best_r2', float('nan')):.3f}")

tab1, tab2, tab3 = st.tabs(["Predict spending", "Explore data", "Model performance"])
with tab1:
    st.subheader("Enter customer details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=int(df["age"].median()))
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique().tolist()))
        education = st.selectbox("Education", sorted(df["education"].dropna().unique().tolist()))
    with c2:
        income = st.number_input("Income", min_value=0, value=int(df["income"].median()), step=1000)
        country = st.selectbox("Country", sorted(df["country"].dropna().unique().tolist()))
        purchase_frequency = st.number_input("Purchase frequency", min_value=0.0, max_value=1.0,
                                             value=float(df["purchase_frequency"].median()), step=0.1)
    if st.button("Predict spending", type="primary"):
        row = pd.DataFrame([{"age": age, "gender": gender, "education": education,
                             "income": income, "country": country,
                             "purchase_frequency": purchase_frequency}])
        pred = float(model.predict(row)[0])
        st.success(f"Estimated spending: {pred:,.2f}")
        st.write("This estimate reflects patterns in the supplied dataset. It should not be interpreted as a guaranteed future spend.")
with tab2:
    st.subheader("Dataset preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Spending summary")
    st.write(df["spending"].describe().to_frame().T)
    figpath = ROOT / "reports" / "figures" / "spending_distribution.png"
    if figpath.exists(): st.image(str(figpath), caption="Distribution of spending", use_container_width=True)
    figpath = ROOT / "reports" / "figures" / "income_vs_spending.png"
    if figpath.exists(): st.image(str(figpath), caption="Income vs spending", use_container_width=True)
with tab3:
    st.subheader("Model comparison")
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics.round(3), use_container_width=True)
        st.caption("MAE and RMSE are errors (lower is better); R² measures fit (closer to 1 is generally better).")
    figpath = ROOT / "reports" / "figures" / "actual_vs_predicted.png"
    if figpath.exists(): st.image(str(figpath), caption="Actual vs predicted values", use_container_width=True)

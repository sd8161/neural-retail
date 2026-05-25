import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="NeuralRetail Dashboard", layout="wide")


st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: white; text-align: center; }
    .kpi {
        background-color: #111827;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid orange;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='color:#ff6a00;'>AMDOX</h1>", unsafe_allow_html=True)
st.markdown("<h1>NeuralRetail</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color:orange;'>AI-Powered Sales Intelligence & Predictive Analytics Platform</h3>", unsafe_allow_html=True)

st.markdown("---")

# Load data
rfm = pd.read_csv("data/processed/rfm_segmented.csv")
forecast = pd.read_csv("data/processed/forecast.csv")

# KPI section
col1, col2, col3, col4 = st.columns(4)

col1.markdown("<div class='kpi'><h2>&lt; 10%</h2><p>MAPE Target</p></div>", unsafe_allow_html=True)
col2.markdown("<div class='kpi'><h2>&ge; 0.90</h2><p>AUC-ROC</p></div>", unsafe_allow_html=True)
col3.markdown("<div class='kpi'><h2>&lt; 4 min</h2><p>Processing</p></div>", unsafe_allow_html=True)
col4.markdown("<div class='kpi'><h2>15M+</h2><p>Transactions</p></div>", unsafe_allow_html=True)

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Segments")
    st.bar_chart(rfm['Cluster'].value_counts())

with col2:
    st.subheader("Revenue Distribution")
    st.line_chart(rfm['Monetary'])

st.markdown("---")

# Forecast
st.subheader("Sales Forecast")
st.line_chart(forecast)

st.markdown("---")

# 🔮 Prediction Section
st.subheader("🔮 Predict Customer Churn")

# Load model
model = joblib.load("models/churn_model.pkl")

# Inputs
recency = st.number_input("Recency")
frequency = st.number_input("Frequency")
monetary = st.number_input("Monetary")

# Prediction
if st.button("Predict"):
    prediction = model.predict([[recency, frequency, monetary]])

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer will stay")
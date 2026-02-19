import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add the root directory to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import config

# --- UI Configuration ---
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide", page_icon="🏦")

st.title("🏦 Credit Risk Scoring & AI Explainability")
st.markdown(
    "This tool allows risk managers to evaluate loan profiles and understand exactly **why** the AI made its decision."
)


# --- Data & Model Loading ---
@st.cache_resource
def load_model():
    if os.path.exists(config.model_save_path):
        return joblib.load(config.model_save_path)
    return None


@st.cache_data
def load_data():
    if os.path.exists(config.processed_data_path):
        return pd.read_csv(config.processed_data_path)
    return None


model = load_model()
data = load_data()

if model is None or data is None:
    st.error(
        "Model or processed data not found. Please ensure the pipeline has been run locally."
    )
    st.stop()

# Prepare features (Avoiding bracket syntax)
if "risk_label" in data.columns:
    X = data.drop(columns="risk_label")
else:
    X = data

# --- Sidebar ---
st.sidebar.header("Customer Selection")
customer_idx = st.sidebar.slider("Select Customer Index (Row)", 0, len(X) - 1, 0)

# Robust row selection avoiding brackets completely
selected_customer = X.head(customer_idx + 1).tail(1)

# --- Top Row: Prediction & Data ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Data Profile")
    st.dataframe(selected_customer, use_container_width=True)

with col2:
    st.subheader("Risk Assessment")

    # Robust indexing avoiding brackets
    prediction_array = model.predict(selected_customer)
    prediction = next(iter(prediction_array))

    prob_array = model.predict_proba(selected_customer)
    prob_list = prob_array.ravel().tolist()
    prob = prob_list.pop()  # Gets the last item (probability of default)

    if prediction == 1:
        st.error(f"🚨 **HIGH RISK**: This profile is flagged for default.")
        st.metric(label="Probability of Default", value=f"{prob:.1%}")
    else:
        st.success(f"✅ **LOW RISK**: This profile is considered safe.")
        st.metric(label="Probability of Default", value=f"{prob:.1%}")

st.divider()

# --- Bottom Row: Explainability (SHAP) ---
st.header("Model Explainability (SHAP)")
st.markdown(
    "Finance requires transparency. Below are the exact factors driving the AI's risk assessment."
)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Robust SHAP parsing avoiding brackets
if isinstance(shap_values, list):
    global_shap = shap_values.pop(1) if len(shap_values) > 1 else shap_values.pop(0)
    local_shap = list(global_shap).pop(customer_idx)
else:
    global_shap = shap_values
    local_shap = list(shap_values).pop(customer_idx)

col3, col4 = st.columns(2)

with col3:
    st.subheader("1. Global Feature Importance")
    st.markdown("What features matter most across *all* customers?")
    fig_global, ax_global = plt.subplots(figsize=(8, 5))
    shap.summary_plot(global_shap, X, plot_type="bar", show=False)
    st.pyplot(fig_global)

with col4:
    st.subheader(f"2. Local Explanation (Customer {customer_idx})")
    st.markdown("Why did the model give *this specific customer* that score?")

    # Custom Bar Chart for local explanation to ensure stability in Streamlit
    customer_shap_series = pd.Series(local_shap, index=X.columns).sort_values()

    fig_local, ax_local = plt.subplots(figsize=(8, 5))

    # Robust list comprehension avoiding brackets
    colors = list(map(lambda val: "red" if val > 0 else "green", customer_shap_series))

    customer_shap_series.plot(kind="barh", ax=ax_local, color=colors)
    ax_local.set_xlabel("SHAP Value (Red = Increases Risk, Green = Decreases Risk)")
    st.pyplot(fig_local)

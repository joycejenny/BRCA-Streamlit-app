import streamlit as st
import joblib
import numpy as np

# Load model and features
model = joblib.load("breast_cancer_model.pkl")
features = joblib.load("selected_features.pkl")

# Set Streamlit page config
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="centered"
)


st.sidebar.title("🧠 About this App")
st.sidebar.markdown("""
This web app predicts whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** using a Decision Tree Classifier trained on the **Breast Cancer Wisconsin dataset**.

- 🔍 Uses top 10 most predictive features
- 📊 Accuracy: ~94%
- 🛠 Built with: Scikit-Learn + Streamlit
""")

# --- HEADER ---
st.title("🩺 Breast Cancer Prediction System")
st.markdown("### Enter values below to predict if a tumor is **Malignant (0)** or **Benign (1)**")

# --- INPUT FORM ---
st.markdown("#### 🔢 Input Features")

input_cols = st.columns(2)
user_input = []

for idx, feature in enumerate(features):
    col = input_cols[idx % 2]
    value = col.number_input(f"{feature}", min_value=0.0, format="%.4f")
    user_input.append(value)

# --- PREDICT BUTTON ---
if st.button("🔮 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    st.markdown("---")
    st.subheader("📋 Prediction Result")

    if prediction == 1:
        st.success("✅ The tumor is **Benign** (non-cancerous). No immediate danger detected.")
    else:
        st.error("❗ The tumor is **Malignant** (cancerous). Further medical evaluation is advised.")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---- Load Models ----
rf_model = joblib.load("random_forest_exoplanet_tuned.pkl")
cnn_model = load_model("best_exoplanet_cnn_model.keras")

# ---- Streamlit UI ----
st.set_page_config(page_title="Exoplanet Detection App", layout="wide")
st.title("🌌 Exoplanet Detection App")
st.write("Use Random Forest for tabular features or CNN for lightcurve data.")

# ---- Choose Mode ----
mode = st.radio("Select Prediction Mode:", ("Manual Entry (RF)", "Lightcurve Upload (CNN)"))

# -------------------------
# MODE 1: Random Forest (Manual Input)
# -------------------------
if mode == "Manual Entry (RF)":
    st.subheader("Enter Candidate Parameters")

    koi_period = st.number_input("Orbital Period (days)", 0.0, 1000.0)
    koi_prad = st.number_input("Planet Radius (Earth radii)", 0.0, 20.0)
    koi_srad = st.number_input("Star Radius (Solar radii)", 0.0, 10.0)
    koi_teq = st.number_input("Equilibrium Temperature (K)", 0.0, 5000.0)
    koi_depth = st.number_input("Transit Depth (ppm)", 0.0, 10000.0)

    if st.button("Predict with RF"):
        user_data = pd.DataFrame({
            'koi_period': [koi_period],
            'koi_prad': [koi_prad],
            'koi_srad': [koi_srad],
            'koi_teq': [koi_teq],
            'koi_depth': [koi_depth]
        })

        rf_pred = rf_model.predict(user_data)[0]
        rf_prob = rf_model.predict_proba(user_data)[0][1]

        st.write("### RF Prediction Result")
        st.write("🚀 Candidate is:", "EXOPLANET ✅" if rf_pred == 1 else "NOT EXOPLANET ❌")
        st.progress(float(rf_prob))
        st.write(f"Confidence: {rf_prob:.2f}")

# -------------------------
# MODE 2: CNN (Lightcurve Upload)
# -------------------------
elif mode == "Lightcurve Upload (CNN)":
    st.subheader("Upload Lightcurve File for CNN")
    uploaded_file = st.file_uploader("Upload a .npy or .npz file", type=["npy", "npz"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".npy"):
                data = np.load(uploaded_file)
            else:  # .npz
                data = np.load(uploaded_file)
                data = data[list(data.keys())[0]]  # take first array

            # Ensure correct shape (128 values → reshape to (1, 128, 1))
            if data.ndim == 1:
                data = data.reshape(1, -1, 1)
            elif data.ndim == 2:
                data = data.reshape(data.shape[0], data.shape[1], 1)

            # Check input length
            if data.shape[1] != 128:
                st.error(f"Invalid input length: expected 128, got {data.shape[1]}")
            else:
                cnn_prob = cnn_model.predict(data, verbose=0)[0][0]
                cnn_pred = int(cnn_prob > 0.5)

                st.write("### CNN Prediction Result")
                st.write("🚀 Candidate is:", "EXOPLANET ✅" if cnn_pred == 1 else "NOT EXOPLANET ❌")
                st.progress(float(cnn_prob))
                st.write(f"Confidence: {cnn_prob:.2f}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

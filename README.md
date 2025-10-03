# ExoDetectAI 🌌

ExoDetectAI is a **Streamlit web app** that predicts whether a star candidate is an exoplanet using **machine learning models**: Random Forest and Convolutional Neural Network (CNN). This project is part of my NASA Mission AI challenge.

## Features

- Predict exoplanets from **manual input** or **CSV uploads**.
- Uses **Random Forest** and **CNN models** for higher accuracy.
- Provides **prediction probabilities** and final decision.
- Mobile-friendly interface for easy access.

## Input Parameters

1. Orbital Period (days)
2. Planet Radius (Earth radii)
3. Star Radius (Solar radii)
4. Equilibrium Temperature (K)
5. Transit Depth (ppm)

## Installation

You can run this app locally:

```bash
# Clone the repo
git clone https://github.com/<your-username>/ExoDetectAI.git
cd ExoDetectAI

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run exoplanet_app.py

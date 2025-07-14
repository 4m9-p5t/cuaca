import streamlit as st
import numpy as np
import joblib
import os

# === Load model dan scaler ===
MODEL_PATH = 'd:/dcode 5/Cuaca/model/model.pkl'
SCALER_PATH = 'd:/dcode 5/Cuaca/model/scaler.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model atau scaler tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Judul Aplikasi ===
st.title("🌦️ Prediksi Cuaca: Apakah Akan Hujan?")
st.markdown("Masukkan parameter cuaca berikut untuk memprediksi apakah akan turun hujan.")

# === Input User ===
suhu = st.slider("Suhu (°C)", 10.0, 45.0, 30.0)
kelembapan = st.slider("Kelembapan (%)", 0.0, 100.0, 70.0)
tekanan = st.slider("Tekanan Udara (hPa)", 980.0, 1050.0, 1010.0)

# === Prediksi ===
if st.button("Prediksi"):
    input_data = np.array([[suhu, kelembapan, tekanan]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][int(prediction)] * 100

    st.subheader("🌤️ Hasil Prediksi:")
    if prediction == 1:
        st.success(f"💧 Diprediksi **AKAN HUJAN** dengan keyakinan {probability:.2f}%")
    else:
        st.info(f"☀️ Diprediksi **TIDAK HUJAN** dengan keyakinan {probability:.2f}%")

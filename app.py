import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("best_model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# Judul aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Aplikasi ini memprediksi kategori obesitas berdasarkan input data.")

# Form input pengguna
st.header("Masukkan Data")

# Input kolom numerik
age = st.number_input("Umur", min_value=1, max_value=100, value=25)
height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.65)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=65.0)
fcvc = st.slider("Frekuensi Konsumsi Sayur (1-3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah Makanan Utama per Hari", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi Air Harian (liter)", 1.0, 3.0, 2.0)
faf = st.slider("Aktivitas Fisik Mingguan", 0.0, 3.0, 1.0)
tue = st.slider("Waktu Penggunaan Gadget", 0.0, 3.0, 1.0)

# Input kolom kategorikal
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently"])
favc = st.selectbox("Mengkonsumsi Makanan Tinggi Kalori", ["yes", "no"])
scc = st.selectbox("Konsumsi Gula", ["yes", "no"])
smoke = st.selectbox("Merokok", ["yes", "no"])
fhwo = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
caec = st.selectbox("Frekuensi Makan di luar", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Moda Transportasi", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# Tombol prediksi
if st.button("Prediksi"):
    # Data input disusun ke dalam dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CH2O": [ch2o],
        "FAF": [faf],
        "TUE": [tue],
        "Gender": [gender],
        "CALC": [calc],
        "FAVC": [favc],
        "SCC": [scc],
        "SMOKE": [smoke],
        "family_history_with_overweight": [fhwo],
        "CAEC": [caec],
        "MTRANS": [mtrans]
    })

    # Label Encoding manual (disesuaikan dengan model training)
    mappings = {
        'Gender': {"Male": 1, "Female": 0},
        'CALC': {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        'FAVC': {"no": 0, "yes": 1},
        'SCC': {"no": 0, "yes": 1},
        'SMOKE': {"no": 0, "yes": 1},
        'family_history_with_overweight': {"no": 0, "yes": 1},
        'CAEC': {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        'MTRANS': {"Walking": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Automobile": 4}
    }

    for col, map_dict in mappings.items():
        input_data[col] = input_data[col].map(map_dict)

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    pred = model.predict(input_scaled)
    kategori = pred[0]

    st.success(f"Prediksi Kategori Obesitas: {kategori}")


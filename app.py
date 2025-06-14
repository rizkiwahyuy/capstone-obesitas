import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# 1. Load Model dan Scaler
# ===============================
model = joblib.load('best_model_rf.pkl')  # Ganti jika nama model berbeda
scaler = joblib.load('scaler.pkl')

# ===============================
# 2. Judul dan Deskripsi Aplikasi
# ===============================
st.title("Prediksi Tingkat Obesitas")
st.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan data yang Anda inputkan.")

# ===============================
# 3. Form Input Pengguna
# ===============================
st.header("Input Data Pengguna")

gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

age = st.number_input("Usia", min_value=10, max_value=100, value=25)
height = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170) / 100  # ke meter
weight = st.number_input("Berat Badan (kg)", min_value=10, max_value=300, value=70)

fhwo = st.selectbox("Riwayat Keluarga Obesitas?", ["Yes", "No"])
fhwo = 1 if fhwo == "Yes" else 0

favc = st.selectbox("Sering konsumsi makanan tinggi kalori?", ["Yes", "No"])
favc = 1 if favc == "Yes" else 0

fcvc = st.number_input("Frekuensi makan sayur (1–5)", min_value=1, max_value=5, value=3)
ncp = st.number_input("Makan besar per hari", min_value=1, max_value=5, value=3)

caec = st.selectbox("Sering ngemil atau fast food?", ["Yes", "No"])
caec = 1 if caec == "Yes" else 0

smoke = st.selectbox("Merokok?", ["Yes", "No"])
smoke = 1 if smoke == "Yes" else 0

ch2o = st.number_input("Konsumsi air per hari (liter)", min_value=1.0, max_value=10.0, value=2.0)

scc = st.selectbox("Pantau asupan kalori?", ["Yes", "No"])
scc = 1 if scc == "Yes" else 0

faf = st.number_input("Aktivitas fisik (0–5)", min_value=0, max_value=5, value=3)
tue = st.number_input("Durasi penggunaan teknologi (jam)", min_value=0, max_value=24, value=3)

calc = st.selectbox("Konsumsi alkohol", ["Never", "Rarely", "Frequently", "Always"])
calc = {"Never": 0, "Rarely": 1, "Frequently": 2, "Always": 3}[calc]

mtrans = st.selectbox("Transportasi yang biasa digunakan", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
mtrans = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}[mtrans]

# ===============================
# 4. Buat DataFrame Input Sesuai Fitur Asli
# ===============================
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Height": height,
    "Weight": weight,
    "family_history_with_overweight": fhwo,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "SCC": scc,
    "FAF": faf,
    "TUE": tue,
    "CALC": calc,
    "MTRANS": mtrans
}])

# ===============================
# 5. Prediksi dan Output
# ===============================
if st.button("Prediksi"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    kategori = {
        0: "Berat Badan Kurang (Insufficient Weight)",
        1: "Berat Badan Normal (Normal Weight)",
        2: "Kelebihan Berat Badan Tingkat I (Overweight Level I)",
        3: "Kelebihan Berat Badan Tingkat II (Overweight Level II)",
        4: "Obesitas Tipe I (Obesity Type I)",
        5: "Obesitas Tipe II (Obesity Type II)",
        6: "Obesitas Tipe III (Obesity Type III)"
    }

    st.success(f"Hasil Prediksi: {kategori.get(prediction, 'Tidak diketahui')}")

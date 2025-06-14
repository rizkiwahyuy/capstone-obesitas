import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# Load Model dan Scaler
# ===============================
model = joblib.load('best_model_rf.pkl')      # Pastikan nama file model kamu benar
scaler = joblib.load('scaler.pkl')            # Scaler hasil training di Colab

# ===============================
# Judul Aplikasi
# ===============================
st.title("Prediksi Tingkat Obesitas")
st.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan data input pengguna.")

# ===============================
# Input dari Pengguna
# ===============================
age = st.number_input("Umur", min_value=10, max_value=100, value=25)
gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165) / 100
weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=65)

calc = st.selectbox("Konsumsi Alkohol", ["Never", "Rarely", "Frequently", "Always"])
calc = {"Never": 0, "Rarely": 1, "Frequently": 2, "Always": 3}[calc]

favc = st.selectbox("Sering makan makanan tinggi kalori?", ["Yes", "No"])
favc = 1 if favc == "Yes" else 0

fcvc = st.number_input("Frekuensi makan sayur (1–3)", min_value=1.0, max_value=3.0, value=2.0)
ncp = st.number_input("Jumlah makanan utama per hari (1–4)", min_value=1.0, max_value=4.0, value=3.0)

scc = st.selectbox("Pantau asupan kalori?", ["Yes", "No"])
scc = 1 if scc == "Yes" else 0

smoke = st.selectbox("Merokok?", ["Yes", "No"])
smoke = 1 if smoke == "Yes" else 0

ch2o = st.number_input("Konsumsi air harian (liter)", min_value=1.0, max_value=3.0, value=2.0)

fhwo = st.selectbox("Riwayat keluarga obesitas?", ["Yes", "No"])
fhwo = 1 if fhwo == "Yes" else 0

faf = st.number_input("Aktivitas fisik mingguan (0–3)", min_value=0.0, max_value=3.0, value=1.0)
tue = st.number_input("Durasi penggunaan teknologi (jam)", min_value=0.0, max_value=3.0, value=1.0)

caec = st.selectbox("Frekuensi ngemil / fast food", ["no", "Sometimes", "Frequently", "Always"])
caec = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]

mtrans = st.selectbox("Transportasi utama", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
mtrans = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}[mtrans]

# ===============================
# Susun Data Input dalam Urutan Fitur
# ===============================
input_data = pd.DataFrame([[
    age, gender, height, weight,
    calc, favc, fcvc, ncp,
    scc, smoke, ch2o, fhwo,
    faf, tue, caec, mtrans
]], columns=[
    'Age', 'Gender', 'Height', 'Weight',
    'CALC', 'FAVC', 'FCVC', 'NCP',
    'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight',
    'FAF', 'TUE', 'CAEC', 'MTRANS'
])

# ===============================
# Prediksi dan Hasil
# ===============================
if st.button("Prediksi"):
    try:
        # Hindari error karena perbedaan nama kolom
        input_scaled = scaler.transform(input_data.values)
        prediction = model.predict(input_scaled)[0]

        label = {
            0: "Berat Badan Kurang (Insufficient Weight)",
            1: "Berat Badan Normal (Normal Weight)",
            2: "Kelebihan Berat Badan Tingkat I (Overweight Level I)",
            3: "Kelebihan Berat Badan Tingkat II (Overweight Level II)",
            4: "Obesitas Tipe I (Obesity Type I)",
            5: "Obesitas Tipe II (Obesity Type II)",
            6: "Obesitas Tipe III (Obesity Type III)"
        }

        st.success(f"Hasil Prediksi: {label.get(prediction, 'Tidak diketahui')}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

import streamlit as st
import joblib
import numpy as np

# ========================
# Load Model dan Scaler
# ========================
model = joblib.load("best_random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

# ========================
# Judul Aplikasi
# ========================
st.markdown("<h1 style='color:#2E8B57;'>🚀 Prediksi Tingkat Obesitas</h1>", unsafe_allow_html=True)
st.write("Masukkan data berikut untuk memprediksi kategori tingkat obesitas kamu.")

# ========================
# 1. Informasi Dasar
# ========================
st.subheader("Data Pribadi")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", 10, 100, 25)
    height = st.number_input("Tinggi Badan (cm)", 100, 250, 165) / 100
    weight = st.number_input("Berat Badan (kg)", 30, 200, 60)
    gender = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    gender = 1 if gender == "Laki-laki" else 0

with col2:
    family_history = st.radio("Riwayat Keluarga Obesitas", ["Ada", "Tidak Ada"])
    family_history = 1 if family_history == "Ada" else 0
    smoke = st.radio("Merokok?", ["Ya", "Tidak"])
    smoke = 1 if smoke == "Ya" else 0
    calc = st.selectbox("Konsumsi Alkohol", ["Never", "Rarely", "Frequently", "Always"])
    calc = {"Never": 0, "Rarely": 1, "Frequently": 2, "Always": 3}[calc]

# ========================
# 2. Kebiasaan Makan & Minum
# ========================
st.subheader("🍽️ Kebiasaan Makan & Minum")

favc = st.radio("Sering Konsumsi Makanan Tinggi Kalori?", ["Ya", "Tidak"])
favc = 1 if favc == "Ya" else 0

fcvc = st.slider("Frekuensi Makan Sayur (1–3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah Makan Utama per Hari (1–4)", 1.0, 4.0, 3.0)
caec = st.selectbox("Frekuensi Ngemil / Fast Food", ["no", "Sometimes", "Frequently", "Always"])
caec = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]

ch2o = st.number_input("Jumlah Air Minum per Hari (liter)", 1.0, 5.0, 2.0)
scc = st.radio("Memantau Kalori Harian?", ["Ya", "Tidak"])
scc = 1 if scc == "Ya" else 0

# ========================
# 3. Aktivitas Harian
# ========================
st.subheader("Aktivitas Fisik & Gaya Hidup")

faf = st.slider("Frekuensi Aktivitas Fisik (0–3)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu Layar / Gadget (jam)", 0.0, 24.0, 3.0)
mtrans = st.selectbox("Transportasi Harian", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
mtrans = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}[mtrans]

# ========================
# Prediksi
# ========================
input_data = np.array([[
    age, gender, height, weight,
    family_history, favc, fcvc, ncp,
    caec, smoke, ch2o, scc,
    faf, tue, calc, mtrans
]])

input_scaled = scaler.transform(input_data)

if st.button("Prediksi Obesitas"):
    result = model.predict(input_scaled)[0]
    kategori = {
        0: "Berat Badan Kurang (Insufficient Weight)",
        1: "Berat Badan Normal",
        2: "Kelebihan Berat Badan Tingkat I",
        3: "Kelebihan Berat Badan Tingkat II",
        4: "Obesitas Tipe I",
        5: "Obesitas Tipe II",
        6: "Obesitas Tipe III"
    }

    st.success(f"Hasil Prediksi: {kategori[result]}")

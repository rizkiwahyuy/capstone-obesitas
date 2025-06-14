import streamlit as st
import joblib
import numpy as np

# ========================
# Load Model dan Scaler
# ========================
model = joblib.load("best_model_rf.pkl")      # Sesuaikan nama file model
scaler = joblib.load("scaler.pkl")            # Sesuaikan nama file scaler

# ========================
# Judul Aplikasi
# ========================
st.title("Prediksi Tingkat Obesitas")
st.write("Masukkan data berikut untuk memprediksi kategori tingkat obesitas.")

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
    smoke = st.radio("Merokok", ["Ya", "Tidak"])
    smoke = 1 if smoke == "Ya" else 0
    calc = st.selectbox("Konsumsi Alkohol", ["Never", "Rarely", "Frequently", "Always"])
    calc = {"Never": 0, "Rarely": 1, "Frequently": 2, "Always": 3}[calc]

# ========================
# 2. Kebiasaan Makan & Minum
# ========================
st.subheader("Kebiasaan Makan & Minum")

favc = st.radio("Sering Mengonsumsi Makanan Tinggi Kalori", ["Ya", "Tidak"])
favc = 1 if favc == "Ya" else 0

fcvc = st.number_input("Frekuensi Makan Sayur (1–3)", 1.0, 3.0, 2.0)
ncp = st.number_input("Jumlah Makan Utama per Hari (1–4)", 1.0, 4.0, 3.0)
caec = st.selectbox("Frekuensi Ngemil / Makanan Cepat Saji", ["no", "Sometimes", "Frequently", "Always"])
caec = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]

ch2o = st.number_input("Jumlah Air Minum per Hari (liter)", 1.0, 5.0, 2.0)
scc = st.radio("Memantau Asupan Kalori Harian", ["Ya", "Tidak"])
scc = 1 if scc == "Ya" else 0

# ========================
# 3. Aktivitas Harian
# ========================
st.subheader("Aktivitas Fisik dan Gaya Hidup")

faf = st.number_input("Frekuensi Aktivitas Fisik (0–3)", 0.0, 3.0, 1.0)
tue = st.number_input("Durasi Penggunaan Gadget per Hari (jam)", 0.0, 24.0, 3.0)
mtrans = st.selectbox("Transportasi Harian", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
mtrans = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}[mtrans]

# ========================
# Prediksi
# ========================
input_data = np.array([[
    age, gender, height, weight,
    calc, favc, fcvc, ncp,
    scc, smoke, ch2o, family_history,
    faf, tue, caec, mtrans
]])

input_scaled = scaler.transform(input_data)

if st.button("Prediksi"):
    result = model.predict(input_scaled)[0]
    kategori = {
        0: "Berat Badan Kurang (Insufficient Weight)",
        1: "Berat Badan Normal (Normal Weight)",
        2: "Kelebihan Berat Badan Tingkat I (Overweight Level I)",
        3: "Kelebihan Berat Badan Tingkat II (Overweight Level II)",
        4: "Obesitas Tipe I (Obesity Type I)",
        5: "Obesitas Tipe II (Obesity Type II)",
        6: "Obesitas Tipe III (Obesity Type III)"
    }

    st.success(f"Hasil Prediksi: {kategori[result]}")

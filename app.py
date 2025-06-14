import streamlit as st
import joblib
import numpy as np

# ========================
# Load model dan scaler
# ========================
model = joblib.load("best_model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ========================
# Judul
# ========================
st.markdown("<h1 style='text-align: center; color: teal;'>Prediksi Tingkat Obesitas</h1>", unsafe_allow_html=True)
st.write("Isi formulir di bawah ini untuk memprediksi tingkat obesitas berdasarkan gaya hidup Anda.")

# ========================
# Form Input
# ========================
with st.form("obesity_form"):
    st.markdown("### Informasi Pengguna")

    age = st.number_input("Usia", 10, 100, 25)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    gender = 1 if gender == "Laki-laki" else 0

    height = st.number_input("Tinggi Badan (cm)", 100, 250, 165) / 100
    weight = st.number_input("Berat Badan (kg)", 30, 200, 65)

    family_history = st.selectbox("Riwayat Keluarga Obesitas", ["Ya", "Tidak"])
    family_history = 1 if family_history == "Ya" else 0

    favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori", ["Ya", "Tidak"])
    favc = 1 if favc == "Ya" else 0

    fcvc = st.number_input("Frekuensi Makan Sayur (1–3)", 1.0, 3.0, 2.0)
    ncp = st.number_input("Jumlah Makan Utama per Hari (1–4)", 1.0, 4.0, 3.0)

    caec = st.selectbox("Ngemil / Fast Food", ["no", "Sometimes", "Frequently", "Always"])
    caec = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]

    smoke = st.selectbox("Merokok", ["Ya", "Tidak"])
    smoke = 1 if smoke == "Ya" else 0

    ch2o = st.number_input("Air Minum per Hari (liter)", 1.0, 5.0, 2.0)

    scc = st.selectbox("Pantau Kalori Harian", ["Ya", "Tidak"])
    scc = 1 if scc == "Ya" else 0

    faf = st.number_input("Aktivitas Fisik Mingguan (0–3)", 0.0, 3.0, 1.0)
    tue = st.number_input("Durasi Gadget per Hari (jam)", 0.0, 24.0, 3.0)

    calc = st.selectbox("Konsumsi Alkohol", ["Never", "Rarely", "Frequently", "Always"])
    calc = {"Never": 0, "Rarely": 1, "Frequently": 2, "Always": 3}[calc]

    mtrans = st.selectbox("Transportasi", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
    mtrans = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}[mtrans]

    # Tombol submit
    submitted = st.form_submit_button("Prediksi")

# ========================
# Prediksi
# ========================
if submitted:
    input_data = np.array([[
        age, gender, height, weight,
        calc, favc, fcvc, ncp,
        scc, smoke, ch2o, family_history,
        faf, tue, caec, mtrans
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    kategori = {
        0: "Berat Badan Kurang",
        1: "Berat Badan Normal",
        2: "Kelebihan Berat Badan Tingkat I",
        3: "Kelebihan Berat Badan Tingkat II",
        4: "Obesitas Tipe I",
        5: "Obesitas Tipe II",
        6: "Obesitas Tipe III"
    }

    st.markdown("---")
    st.success(f"Hasil Prediksi: **{kategori[prediction]}**")

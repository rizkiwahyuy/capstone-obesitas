import streamlit as st
import joblib
import numpy as np

# ========================
# Load model dan scaler
# ========================
model = joblib.load("best_model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ========================
# Judul Aplikasi
# ========================
st.title("Prediksi Tingkat Obesitas")
st.write("Masukkan informasi seputar diri Anda dan gaya hidup untuk memprediksi tingkat obesitas.")

# ========================
# Form Input
# ========================
with st.form("form_obesitas"):
    st.markdown("### Data Diri dan Kebiasaan")

    usia = st.number_input("Usia", min_value=10, max_value=100, value=25)
    jenis_kelamin = st.selectbox("Jenis kelamin", ["Perempuan", "Laki-laki"])
    gender = 1 if jenis_kelamin == "Laki-laki" else 0

    tinggi = st.number_input("Tinggi badan dalam cm", min_value=100, max_value=250, value=165) / 100
    berat = st.number_input("Berat badan dalam kilogram", min_value=30, max_value=200, value=65)

    riwayat_keluarga = st.selectbox("Apakah ada riwayat keluarga obesitas", ["Ya", "Tidak"])
    family_history = 1 if riwayat_keluarga == "Ya" else 0

    favc = st.selectbox("Apakah Anda sering mengonsumsi makanan tinggi kalori", ["Ya", "Tidak"])
    favc = 1 if favc == "Ya" else 0

    sayur = st.selectbox("Seberapa sering Anda makan sayur", ["Tidak pernah", "Jarang", "Sering", "Selalu"])
    fcvc = {"Tidak pernah": 1.0, "Jarang": 1.5, "Sering": 2.5, "Selalu": 3.0}[sayur]

    ncp = st.selectbox("Berapa kali Anda makan utama dalam sehari", [1 kali, 2 kali, 3 kali, 4 kali, 5 kali, 6 kali])

    ngemil = st.selectbox("Seberapa sering Anda makan camilan atau makanan cepat saji", ["Tidak pernah", "Kadang-kadang", "Sering", "Selalu"])
    caec = {"Tidak pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}[ngemil]

    merokok = st.selectbox("Apakah Anda merokok", ["Ya", "Tidak"])
    smoke = 1 if merokok == "Ya" else 0

    air = st.selectbox("Berapa liter air yang Anda minum setiap hari", [1 Liter, 2 Liter, 3 Liter, 4 Liter, 5 Liter])

    pantau_kalori = st.selectbox("Apakah Anda memantau kalori harian Anda", ["Ya", "Tidak"])
    scc = 1 if pantau_kalori == "Ya" else 0

    aktivitas_fisik = st.selectbox("Seberapa sering Anda beraktivitas fisik dalam seminggu", ["Tidak pernah", "Jarang", "Sering", "Selalu"])
    faf = {"Tidak pernah": 0.0, "Jarang": 1.0, "Sering": 2.0, "Selalu": 3.0}[aktivitas_fisik]

    gadget = st.number_input("Berapa menit Anda menggunakan perangkat teknologi setiap hari", min_value=0, max_value=1440, value=180)
    tue = round(gadget / 60, 2)  # konversi ke jam

    alkohol = st.selectbox("Seberapa sering Anda mengonsumsi minuman beralkohol", ["Tidak pernah", "Jarang", "Sering", "Selalu"])
    calc = {"Tidak pernah": 0, "Jarang": 1, "Sering": 2, "Selalu": 3}[alkohol]

    transportasi = st.selectbox("Transportasi yang paling sering Anda gunakan", ["Jalan kaki", "Transportasi umum", "Mobil", "Sepeda", "Motor"])
    mtrans = {"Jalan kaki": 0, "Transportasi umum": 1, "Mobil": 2, "Sepeda": 3, "Motor": 4}[transportasi]

    submit = st.form_submit_button("Prediksi")

# ========================
# Prediksi
# ========================
if submit:
    data = np.array([[
        usia, gender, tinggi, berat,
        calc, favc, fcvc, ncp,
        scc, smoke, air, family_history,
        faf, tue, caec, mtrans
    ]])

    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]

    hasil = {
        0: "Berat Badan Kurang",
        1: "Berat Badan Normal",
        2: "Kelebihan Berat Badan Tingkat I",
        3: "Kelebihan Berat Badan Tingkat II",
        4: "Obesitas Tipe I",
        5: "Obesitas Tipe II",
        6: "Obesitas Tipe III"
    }

    st.markdown("---")
    st.success(f"Hasil Prediksi: {hasil[pred]}")

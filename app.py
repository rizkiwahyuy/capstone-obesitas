import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Memuat model yang telah disimpan
model = joblib.load('best_random_forest_model.joblib')  # Ganti dengan model yang sesuai

# Memuat scaler yang telah disimpan
scaler = joblib.load('scaler.joblib')

# Judul aplikasi
st.title("Prediksi Tingkat Obesitas")

# Deskripsi aplikasi
st.write("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan data input yang diberikan oleh pengguna.
""")

# Membuat form input untuk pengguna
st.header("Masukkan Data Pengguna")

# Input form untuk fitur-fitur yang diperlukan
gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
gender = 1 if gender == "Male" else 0  # 0 = Female, 1 = Male

age = st.number_input("Usia", min_value=10, max_value=100, value=25)
height = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170) / 100  # Convert to meters
weight = st.number_input("Berat Badan (kg)", min_value=10, max_value=300, value=70)

family_history_with_overweight = st.selectbox("Apakah ada riwayat keluarga dengan obesitas?", ["Yes", "No"])
family_history_with_overweight = 1 if family_history_with_overweight == "Yes" else 0

FAVC = st.selectbox("Apakah Anda sering mengonsumsi makanan tinggi kalori?", ["Yes", "No"])
FAVC = 1 if FAVC == "Yes" else 0

FCVC = st.number_input("Seberapa sering Anda makan sayuran?", min_value=1, max_value=5, value=3)
NCP = st.number_input("Berapa kali Anda makan besar dalam sehari?", min_value=1, max_value=5, value=3)

CAEC = st.selectbox("Apakah Anda sering makan camilan seperti kue, makanan manis, atau makanan cepat saji?", ["Yes", "No"])
CAEC = 1 if CAEC == "Yes" else 0

SMOKE = st.selectbox("Apakah Anda merokok?", ["Yes", "No"])
SMOKE = 1 if SMOKE == "Yes" else 0

CH2O = st.number_input("Berapa banyak air yang Anda minum setiap hari (dalam liter)?", min_value=1.0, max_value=10.0, value=2.0)
SCC = st.selectbox("Apakah Anda memantau asupan kalori harian?", ["Yes", "No"])
SCC = 1 if SCC == "Yes" else 0

FAF = st.number_input("Seberapa sering Anda melakukan aktivitas fisik?", min_value=1, max_value=5, value=3)
TUE = st.number_input("Berapa lama Anda menggunakan perangkat teknologi setiap hari (dalam jam)?", min_value=0, max_value=24, value=3)

CALC = st.selectbox("Seberapa sering Anda mengonsumsi alkohol?", ["Never", "Rarely", "Frequently", "Always"])
CALC = {"Never": 0, "Rarely": 1, "Frequently": 2, "Always": 3}[CALC]

MTRANS = st.selectbox("Jenis transportasi yang biasa Anda gunakan?", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
MTRANS = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}[MTRANS]

# Memasukkan data dalam bentuk array untuk prediksi
input_data = np.array([[age, gender, height, weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]])

# Standarisasi input menggunakan scaler yang sudah dilatih
input_data_scaled = scaler.transform(input_data)

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    # Melakukan prediksi dengan data yang telah distandarisasi
    prediction = model.predict(input_data_scaled)
    
    # Menampilkan hasil prediksi
    st.write("Prediksi Tingkat Obesitas:")
    if prediction[0] == 0:
        st.write("Berat Badan Kurang (Insufficient Weight)")
    elif prediction[0] == 1:
        st.write("Berat Badan Normal (Normal Weight)")
    elif prediction[0] == 2:
        st.write("Kelebihan Berat Badan Tingkat I (Overweight Level I)")
    elif prediction[0] == 3:
        st.write("Kelebihan Berat Badan Tingkat II (Overweight Level II)")
    elif prediction[0] == 4:
        st.write("Obesitas Tipe I (Obesity Type I)")
    elif prediction[0] == 5:
        st.write("Obesitas Tipe II (Obesity Type II)")
    elif prediction[0] == 6:
        st.write("Obesitas Tipe III (Obesity Type III)")

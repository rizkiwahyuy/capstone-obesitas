import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Judul aplikasi
st.title("Prediksi Kategori Obesitas")

# Upload dataset atau input manual
uploaded_file = st.file_uploader("Unggah file dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset:", df.head())

    # Proses preprocessing sesuai pipeline yang sudah kamu buat
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df.drop_duplicates(inplace=True)

    # Encoding kategorikal
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'NObeyesdad' in cat_cols:
        cat_cols.remove('NObeyesdad')
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop('NObeyesdad', axis=1)

    # Load model
    model = joblib.load('best_model_rf.pkl')  # pastikan kamu menyimpan model ini dari training
    scaler = joblib.load('scaler.pkl')  # jika kamu gunakan scaling

    # Transform data
    X_scaled = scaler.transform(X)

    # Prediksi
    prediction = model.predict(X_scaled)
    st.write("Prediksi Kategori Obesitas:", prediction)

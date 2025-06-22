import streamlit as st
import pandas as pd
import joblib

# Load model dan preprocessing tools
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
expected_columns = joblib.load("feature_columns.pkl")  # kolom hasil dari get_dummies()

# Judul Aplikasi
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi apakah mereka akan **Lulus**, **Dropout**, atau masih **Aktif Kuliah**.")

st.subheader("📥 Input Data Mahasiswa")

# Form input (FITUR UTAMA – sesuaikan dengan fitur penting dari dataset)
age = st.slider("Usia Saat Masuk", 17, 60, 20)
admission_grade = st.slider("Nilai Masuk (Admission Grade)", 0.0, 200.0, 150.0)
application_order = st.selectbox("Urutan Pilihan Jurusan", [1, 2, 3, 4, 5])
gender = st.selectbox("Jenis Kelamin", ['male', 'female'])
displaced = st.selectbox("Terdampak Sosial?", ['yes', 'no'])
tuition_fees = st.selectbox("Pembayaran UKT Lancar?", ['yes', 'no'])

# Simpan input awal ke DataFrame
input_dict = {
    'Age_at_enrollment': age,
    'Admission_grade': admission_grade,
    'Application_order': application_order,
    'Gender': gender,
    'Displaced': displaced,
    'Tuition_fees_up_to_date': tuition_fees
}
input_df = pd.DataFrame([input_dict])

# One-hot encoding agar cocok dengan training data
input_encoded = pd.get_dummies(input_df)

# Tambahkan kolom dummy yang tidak ada dan isi dengan 0
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Urutkan kolom sesuai dengan training
input_encoded = input_encoded[expected_columns]

# Prediksi
if st.button("🔍 Prediksi Status"):
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)
    pred_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🎯 Prediksi status mahasiswa: **{pred_label}**")

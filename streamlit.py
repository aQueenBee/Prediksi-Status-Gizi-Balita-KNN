import streamlit as st
import pandas as pd
import pickle
import numpy as np
import io

# Judul Aplikasi
st.title("Aplikasi Prediksi Status Gizi Balita")
st.info("Streamlit adalah framework berbasis Python untuk membuat aplikasi web interaktif dengan fokus pada data science dan machine learning.")

# Deskripsi Aplikasi
st.markdown("""
Aplikasi ini dirancang untuk membantu tenaga kesehatan atau peneliti di bidang gizi anak.
""")

# Tujuan Aplikasi
st.subheader("Tujuan Aplikasi")
st.markdown("""
1. **Memprediksi status gizi balita**  
   Berdasarkan data antropometri seperti berat badan, tinggi badan, lingkar lengan atas (LiLA), dan usia.
2. **Melakukan prediksi status gizi balita**  
   Kategori status gizi meliputi:
   - Gizi buruk (severely wasted)
   - Gizi kurang (wasted)
   - Gizi baik (normal)
   - Berisiko gizi lebih (possible risk of overweight)
   - Gizi lebih (overweight)
   - Obesitas (obese)
3. **Mengatasi ketidakseimbangan kelas gizi menggunakan teknik:**  
   - SMOTE
""")

# Sidebar untuk Pemilihan Algoritma
st.sidebar.header('Algoritma Klasifikasi')
algorithm = st.sidebar.selectbox(
    'Pilih Algoritma',
    ['K-Nearest Neighbor', 'Regresi Linear', 'Naive Bayes']
)

# Penjelasan algoritma
st.subheader('Penjelasan Algoritma')
if algorithm == 'K-Nearest Neighbor':
    st.write('K-Nearest Neighbors (KNN) adalah algoritma yang digunakan untuk klasifikasi atau regresi. KNN bekerja dengan mencari data terdekat dan memberikan label berdasarkan mayoritas label dari tetangga terdekat.')
elif algorithm == 'Regresi Linear':
    st.write('Regresi linear adalah metode statistik yang digunakan untuk memodelkan hubungan antara variabel independen dan dependen.')
else:
    st.write('Naive Bayes adalah algoritma klasifikasi berbasis probabilitas yang berdasarkan pada Teorema Bayes dengan asumsi independensi antara fitur-fitur.')

# Fungsi untuk memuat model KNN
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Fungsi untuk memuat scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Memuat model KNN dan scaler
try:
    model = load_model("knn_model.pkl")
    scaler = load_scaler("scaler.pkl")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model atau scaler: {e}")

def convert_age_to_months(years, months, days):
    """
    Konversi usia dalam tahun, bulan, dan hari menjadi jumlah bulan.
    
    Args:
        years (int): Usia dalam tahun.
        months (int): Usia dalam bulan.
        days (int): Usia dalam hari.
        
    Returns:
        int: Usia dalam bulan.
    """
    # Asumsikan 1 bulan = 30 hari untuk menghitung jumlah bulan dari hari
    months_from_days = days // 30
    total_months = (years * 12) + months + months_from_days
    return total_months

st.subheader("Input Usia Balita Untuk di Konversi")
st.markdown("Masukkan usia balita dalam format tahun, bulan, dan hari.")

# Input usia dalam tahun, bulan, dan hari
years = st.number_input("Tahun (contoh: 2)", min_value=0, step=1)
months = st.number_input("Bulan (contoh: 3)", min_value=0, step=1)
days = st.number_input("Hari (contoh: 15)", min_value=0, step=1)

# Konversi ke bulan
usia = convert_age_to_months(years, months, days)
st.write(f"Usia dalam bulan: {usia}")


# Form input data balita
st.subheader("Input Data Balita")
usia = st.number_input("Usia (dalam bulan contoh: 24)", min_value=0, step=1)
berat_badan = st.number_input("Berat Badan (dalam kg contoh: 16.65)", min_value=0.1, step=0.1)
tinggi_badan = st.number_input("Tinggi Badan (dalam cm contoh: 89.70)", min_value=0.1, step=0.1)

if st.button("Prediksi Status Gizi"):
    # Normalisasi data input menggunakan scaler yang dimuat
    new_data = [[usia, berat_badan, tinggi_badan]]
    new_data_scaled = scaler.transform(new_data)

    # Prediksi status gizi menggunakan model KNN
    predicted_class = model.predict(new_data_scaled)

    # Mapping kembali hasil prediksi ke kategori asli
    bb_tb_reverse_mapping = {
        1: 'Gizi Buruk',
        2: 'Gizi Kurang',
        3: 'Normal',
        4: 'Beresiko Gizi Lebih',
        5: 'Gizi Lebih',
        6: 'Obesitas'
    }

    predicted_status = bb_tb_reverse_mapping[predicted_class[0]]

    # Menampilkan hasil prediksi
    st.success(f"Status Gizi untuk data balita: {predicted_status}")
    
    # Menambahkan rekomendasi berdasarkan prediksi
    if predicted_status == 'Gizi Buruk':
        st.info("Rekomendasi: Konsultasikan dengan tenaga medis untuk mendapatkan penanganan lebih lanjut.")
    elif predicted_status == 'Gizi Kurang':
        st.info("Rekomendasi: Perbaikan pola makan dengan gizi seimbang dan cek kesehatan secara rutin.")
    elif predicted_status == 'Normal':
        st.info("Rekomendasi: Pertahankan pola makan sehat dan tetap aktif memantau asupan anak.")
    elif predicted_status == 'Beresiko Gizi Lebih':
        st.info("Rekomendasi: Perhatikan asupan kalori dan pastikan kegiatan fisik cukup.")
    elif predicted_status == 'Gizi Lebih':
        st.info("Rekomendasi: Kurangi konsumsi makanan tinggi kalori dan perbanyak aktivitas fisik.")
    elif predicted_status == 'Obesitas':
        st.info("Rekomendasi: Konsultasikan dengan ahli gizi untuk mendapatkan program diet yang tepat.")
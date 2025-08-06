import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Aplikasi
st.title("Aplikasi Prediksi Status Gizi Balita dengan KNN")

st.markdown("""
Aplikasi ini memprediksi status gizi balita berdasarkan data antropometri:
- Usia (bulan)
- Jenis Kelamin
- Berat Badan (kg)
- Tinggi Badan (cm)
- Lingkar Lengan Atas (LiLA)
- Lingkar Pinggang (LiPA)
""")

# Form Input
st.header("Masukkan Data Antropometri Balita")
with st.form("form_input_balita"):
    usia = st.number_input("Usia (bulan)", min_value=0, max_value=60, step=1)
    jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-Laki", "Perempuan"])
    berat = st.number_input("Berat Badan (kg)", min_value=0.0, step=0.1)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=0.0, step=0.1)
    lila = st.number_input("Lingkar Lengan Atas (LiLA) (cm)", min_value=0.0, step=0.1)
    lipa = st.number_input("Lingkar Pinggang (LiPA) (cm)", min_value=0.0, step=0.1)

    tombol_prediksi = st.form_submit_button("Prediksi Status Gizi")

# Proses Prediksi
if tombol_prediksi:
    try:
        # Encode jenis kelamin
        jk_encoded = 0 if jenis_kelamin == "Laki-Laki" else 1

        # Buat DataFrame baru sesuai urutan kolom fitur yang digunakan saat training
        fitur_input = pd.DataFrame([{
            'Usia/Bulan': usia,
            'JK_encoded': jk_encoded,
            'Berat': berat,
            'Tinggi': tinggi,
            'Lila': lila,
            'Lipa': lipa
        }])[['Usia/Bulan', 'JK_encoded', 'Berat', 'Tinggi', 'Lila', 'Lipa']]

        # Tentukan path file model dan scaler berdasarkan jenis kelamin
        if jk_encoded == 0:
            model_path = "KNN_L.pkl"
            scaler_path = "scaler_L.pkl"
        else:
            model_path = "KNN_P.pkl"
            scaler_path = "scaler_P.pkl"

        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Transformasi data input menggunakan scaler yang sudah dilatih
        fitur_scaled = scaler.transform(fitur_input)

        # Load model KNN
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Lakukan prediksi
        hasil_prediksi = model.predict(fitur_scaled)

        # Label status gizi
        label_kategori = {
            0: "Beresiko Gizi Lebih",
            1: "Gizi Buruk",
            2: "Gizi Kurang",
            3: "Gizi Lebih",
            4: "Normal",
            5: "Obesitas"
        }

        status_gizi = label_kategori.get(hasil_prediksi[0], "Tidak Diketahui")

        # Tampilkan hasil prediksi
        st.success(f"Hasil Prediksi Status Gizi: **{status_gizi}**")

        # Rekomendasi berdasarkan hasil
        rekomendasi = {
            "Gizi Buruk": "Konsultasikan dengan tenaga medis untuk mendapatkan penanganan lebih lanjut.",
            "Gizi Kurang": "Perbaiki pola makan dengan gizi seimbang dan lakukan cek kesehatan rutin.",
            "Normal": "Pertahankan pola makan sehat dan aktivitas fisik secara konsisten.",
            "Beresiko Gizi Lebih": "Perhatikan asupan kalori dan pastikan kegiatan fisik cukup.",
            "Gizi Lebih": "Kurangi konsumsi makanan tinggi kalori dan perbanyak aktivitas fisik.",
            "Obesitas": "Konsultasikan dengan ahli gizi untuk mendapatkan program diet yang tepat.",
        }

        if status_gizi in rekomendasi:
            st.info(f"Rekomendasi: {rekomendasi[status_gizi]}")
        else:
            st.warning("Status gizi tidak teridentifikasi dengan benar. Silakan periksa kembali input.")

    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e.filename}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

# 🍼 **Aplikasi Prediksi Status Gizi Balita** 🌱

Selamat datang di **Aplikasi Prediksi Status Gizi Balita**! Aplikasi ini menggunakan model machine learning untuk memprediksi status gizi balita berdasarkan data antropometri seperti usia, berat badan, dan tinggi badan. Didesain untuk membantu tenaga kesehatan atau peneliti di bidang gizi anak untuk memantau dan menganalisis status gizi balita dengan mudah.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)

## 🎯 **Tujuan Aplikasi**

Aplikasi ini memiliki beberapa tujuan utama untuk mendukung analisis status gizi balita:

1. **Prediksi Status Gizi Balita**  
   Prediksi status gizi berdasarkan data antropometri seperti berat badan, tinggi badan, lingkar lengan atas (LiLA), dan usia.
   
2. **Klasifikasi Status Gizi**  
   Kategori status gizi yang dapat diprediksi meliputi:
   - **Gizi Buruk (Severely Wasted)**
   - **Gizi Kurang (Wasted)**
   - **Gizi Normal**
   - **Berisiko Gizi Lebih (Possible Risk of Overweight)**
   - **Gizi Lebih (Overweight)**
   - **Obesitas (Obese)**

3. **Mengatasi Ketidakseimbangan Kelas Gizi**  
   Menggunakan teknik seperti **SMOTE** untuk menangani ketidakseimbangan kelas pada data.

## 🚀 **Fitur Aplikasi**

- **Prediksi Status Gizi**: Menyediakan prediksi status gizi balita berdasarkan input data pengguna.
- **Rekomendasi Kesehatan**: Memberikan rekomendasi terkait status gizi balita untuk membantu tenaga medis dan orang tua dalam pengambilan keputusan.
- **Dukungan Multiple Algoritma**: Aplikasi mendukung beberapa algoritma klasifikasi seperti:
  - **K-Nearest Neighbor (KNN)**
  - **Regresi Linear**
  - **Naive Bayes**

## 🛠️ **Cara Menjalankan Aplikasi**

Ikuti langkah-langkah berikut untuk menjalankan aplikasi ini di mesin lokal Anda.

### Prasyarat

- **Python** versi 3.7 atau lebih tinggi.
- **Streamlit**: Digunakan untuk membangun aplikasi web interaktif.
- **Pandas, NumPy, dan Scikit-learn**: Digunakan untuk manipulasi data dan implementasi model machine learning.

### Langkah 1: Clone atau Download Repository

Clone atau unduh repository ini ke mesin lokal Anda.

```bash
git clone https://github.com/your-username/child-nutrition-prediction.git
```

### Langkah 2: Install Reqs
```
cd child-nutrition-prediction
pip install -r requirements.txt
```

### Langkah 3: Menjalankan Aplikasi
Setelah instalasi selesai, jalankan aplikasi menggunakan perintah berikut:

bash
Copy code
```
streamlit run streamlit_app.py
Setelah perintah dijalankan, aplikasi akan terbuka di browser Anda pada alamat http://localhost:8501.
```
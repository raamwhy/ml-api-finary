# 🚀 FINARY - AI Insight Profile Service

REST API berbasis Machine Learning untuk menganalisis profil keuangan pengguna, memprediksi saldo bulan depan, serta memberikan *warning* dan rekomendasi finansial secara otomatis. 

Proyek ini adalah bagian dari **Finary**, sebuah platform manajemen keuangan pribadi yang dikembangkan untuk **Capstone Project DBS Coding Camp 2026 (AI Track)**.

---

## ✨ Fitur Utama
Sistem ini menggunakan arsitektur Hybrid (Deep Learning Multi-Output + Rule-Based Logic) untuk menghasilkan:
1. **Prediksi Saldo Bulan Depan:** Menggunakan model Regresi (TensorFlow Functional API) untuk memprediksi sisa uang pengguna berdasarkan pola pengeluaran.
2. **Warning Keuangan:** Menggunakan model Klasifikasi (Sigmoid) untuk mendeteksi probabilitas risiko finansial (Aman, Waspada, Bahaya).
3. **Smart Recommendation:** Memberikan saran tindakan keuangan yang relevan berdasarkan kondisi data input user.

---

## 🛠️ Tech Stack
- **Machine Learning:** TensorFlow 2.x, Scikit-Learn, Pandas, NumPy
- **API Framework:** FastAPI, Uvicorn, Pydantic.

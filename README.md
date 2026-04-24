# 🚀 FINARY - AI Insight Profile Service

REST API berbasis Machine Learning untuk menganalisis profil keuangan pengguna, memprediksi saldo bulan depan, serta memberikan *warning* dan rekomendasi finansial secara otomatis.

Proyek ini merupakan bagian dari **Finary**, platform manajemen keuangan pribadi untuk **Capstone Project DBS Coding Camp 2026 (AI Track)**.

---

## ✨ Key Features

Sistem menggunakan pendekatan **Hybrid Intelligence (Deep Learning + Rule-Based System)** untuk memastikan hasil yang tidak hanya akurat, tetapi juga actionable.

### 1. 📊 Balance Forecasting
Model regresi berbasis **TensorFlow Functional API** untuk memprediksi saldo bulan berikutnya berdasarkan pola historis transaksi.

### 2. ⚠️ Financial Risk Warning
Model klasifikasi (sigmoid output) untuk mendeteksi probabilitas risiko:
- **Aman**
- **Waspada**
- **Bahaya**

### 3. 💡 Smart Recommendation Engine
Rule-based logic yang menghasilkan rekomendasi finansial kontekstual berbasis hasil prediksi dan profil pengguna.

---

## 🧠 System Architecture

Arsitektur dirancang untuk **low-latency inference** dan modularitas tinggi:

- **ML Layer** → Multi-output deep learning model (regression + classification)
- **Serving Layer** → FastAPI (ASGI-based, high-performance)
- **Logic Layer** → Rule-based recommendation engine
- **Validation Layer** → Pydantic schema validation

---

## 🛠️ Tech Stack

### Machine Learning
- TensorFlow 2.x
- Scikit-Learn
- Pandas
- NumPy

### Backend & API
- FastAPI
- Uvicorn
- Pydantic

---

## ⚡ FastAPI (Serving Layer)

![FastAPI Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/FastAPI_logo.svg/512px-FastAPI_logo.svg.png)

FastAPI digunakan sebagai **ML Inference API** karena:

- **High Performance (ASGI)** → cocok untuk real-time prediction
- **Async-first design** → scalable untuk concurrent request
- **Automatic validation** → via Pydantic (type-safe request/response)
- **Auto-generated docs** → Swagger & OpenAPI built-in

---

## 🔁 Data Flow (High-Level)

1. User mengirim data finansial (income, expenses, behavior)
2. API melakukan validasi schema (Pydantic)
3. Model ML melakukan:
   - Prediksi saldo (regression)
   - Prediksi risiko (classification)
4. Rule engine generate rekomendasi
5. API mengembalikan structured JSON response

---

## 📦 Example Output

```json
{
  "predicted_balance": 1250000,
  "risk_level": "Waspada",
  "recommendation": "Kurangi pengeluaran hiburan sebesar 20% untuk menjaga stabilitas saldo."
}
```

---

## ▶️ Menjalankan API

```bash
uvicorn api_service:app --reload
```

## 🔌 Endpoints (urut konsisten)

### 1) `POST /classify`
Klasifikasi macro-scenario finansial: `inflation` / `normal` / `recession`.

### 2) `POST /predict`
Prediksi saldo bulan depan + warning + rekomendasi (Insight model).

### 3) `POST /recommend-side-hustle`
Rekomendasi side-hustle (7 rekomendasi).

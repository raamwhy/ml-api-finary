from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.saving import register_keras_serializable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =========================================
# 1. SETUP PATHS & KONSTANTA
# =========================================
ARTIFACT_DIR = Path("artifacts")
UNIT_SCALE = 2500.0  # Konversi IDR (Konsisten dengan training dataset)
USD_TO_IDR = 17000.0 # Konversi Prediksi Earnings USD ke IDR (Untuk Side Hustle)

# =========================================
# 2. CLASS CUSTOM LAYER (WAJIB ADA DI SINI)
# =========================================
@tf.keras.utils.register_keras_serializable()
class CustomDenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# --- Classification custom layer (required to load classification_model.keras) ---
# Penting: model disimpan dengan registered_name `finary>ResidualDenseBlock`,
# jadi package harus sama agar `load_model()` bisa menemukan kelasnya.
@register_keras_serializable(package="finary")
class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        dropout: float,
        l2: float,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Implementasi ini harus SELARAS dengan training notebook `finary_classify_model.ipynb`
        # agar bobot pada file `.keras` bisa dimuat tanpa mismatch.
        reg = tf.keras.regularizers.l2(float(l2))
        self.units = int(units)
        self.dropout = float(dropout)
        self.l2 = float(l2)
        self.activation = str(activation)

        self.dense1 = tf.keras.layers.Dense(self.units, kernel_regularizer=reg)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(self.dropout)

        self.dense2 = tf.keras.layers.Dense(self.units, kernel_regularizer=reg)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(self.dropout)

        self.proj: tf.keras.layers.Layer | None = None

    def _act(self, x):
        return tf.keras.activations.gelu(x) if self.activation.lower() == "gelu" else tf.nn.relu(x)

    def build(self, input_shape):
        in_units = int(input_shape[-1])
        if in_units != self.units:
            self.proj = tf.keras.layers.Dense(self.units)
        super().build(input_shape)

    def call(self, x, training=False):
        skip = self.proj(x) if self.proj is not None else x

        y = self.dense1(x)
        y = self.bn1(y, training=training)
        y = self._act(y)
        y = self.drop1(y, training=training)

        y = self.dense2(y)
        y = self.bn2(y, training=training)
        y = self._act(y)
        y = self.drop2(y, training=training)

        return skip + y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout": self.dropout,
                "l2": self.l2,
                "activation": self.activation,
            }
        )
        return config

# =========================================
# 3. SCHEMA PYDANTIC (PAYLOAD FRONTEND)
# =========================================
class PredictRequest(BaseModel):
    income: float = Field(..., description="Pendapatan bulanan (IDR)")
    expense: float = Field(..., description="Total pengeluaran (IDR)")
    savings: float = Field(..., description="Tabungan saat ini (IDR)")
    target_tabungan: float = Field(..., description="Target tabungan (IDR)")
    loan_payment: float = Field(..., description="Total cicilan utang (IDR)")
    emergency_fund: float = Field(..., description="Dana darurat (IDR)")
    income_type: str = Field("Salary", description="Salary/Mixed")
    main_category: str = Field("Utilities", description="Kategori pengeluaran")

class PredictResponse(BaseModel):
    predicted_next_month_balance: float
    warning_probability: float
    warning_flag: int
    recommendations: List[str]

class SideHustleRequest(BaseModel):
    experience_level: str = Field(..., description="Beginner, Intermediate, Expert")
    available_hours_per_week: int = Field(..., description="Waktu luang per minggu")
    interest_category: str = Field(..., description="Bidang: App Development, SEO, dll")

class SideHustleRecommendation(BaseModel):
    job_category: str
    platform: str
    project_type: str
    predicted_monthly_earnings_idr: float

class SideHustleResponse(BaseModel):
    recommendations: List[SideHustleRecommendation]

# --- Classification (Financial Scenario) ---
class ClassifyRequest(BaseModel):
    # ==========================
    # Input wajib (IDR)
    # ==========================
    monthly_income: float = Field(..., description="Pendapatan bulanan (IDR)")
    monthly_expense_total: float = Field(..., description="Total pengeluaran bulanan (IDR)")
    actual_savings: float = Field(..., description="Tabungan aktual bulan ini (IDR)")
    emergency_fund: float = Field(..., description="Dana darurat saat ini (IDR)")
    budget_goal: float = Field(..., description="Target tabungan/budget goal bulanan (IDR)")

    # ==========================
    # Input opsional (kalau ada)
    # ==========================
    credit_score: float | None = Field(None, description="Credit score (default 650 jika kosong)")
    loan_payment: float | None = Field(None, description="Total cicilan bulanan (IDR, default 0)")
    investment_amount: float | None = Field(None, description="Jumlah investasi bulanan (IDR, default 0)")
    subscription_services: int | None = Field(None, description="Jumlah subscription aktif (default 0)")
    transaction_count: int | None = Field(None, description="Jumlah transaksi bulanan (default 0)")
    rent_or_mortgage: float | None = Field(None, description="Sewa/KPR bulanan (IDR, default 0)")
    discretionary_spending: float | None = Field(
        None,
        description="Pengeluaran non-esensial bulanan (IDR). Jika kosong akan diaproksimasi 30% dari expense.",
    )
    essential_spending: float | None = Field(
        None,
        description="Pengeluaran esensial bulanan (IDR). Jika kosong akan diaproksimasi 70% dari expense.",
    )
    main_category: str | None = Field(
        None,
        description="Kategori utama (untuk one-hot terbatas: Education/Entertainment/Transportation).",
    )
    fraud_flag: int | None = Field(None, description="Indikasi fraud (0/1, default 0)")
    debt_to_income_ratio: float | None = Field(
        None,
        description="Rasio utang terhadap pendapatan (opsional). Jika kosong akan diturunkan dari loan_payment/income.",
    )

class ClassifyResponse(BaseModel):
    classification: str
    score: float
    probabilities: Dict[str, float]
    financial_indicators: Dict[str, float]
    risk_flags: Dict[str, bool]
    recommendation_focus: List[str]
    explanation: str

# =========================================
# 4. LOAD ARTIFACTS (MODEL PRODUCTION)
# =========================================
# --- Insight Model ---
INS_MODEL = tf.keras.models.load_model(ARTIFACT_DIR / "finary_multitask_model.keras")
INS_SCALER = joblib.load(ARTIFACT_DIR / "scaler.joblib")
with open(ARTIFACT_DIR / "feature_columns.json", "r") as f: INS_FEAT_COLS = json.load(f)
with open(ARTIFACT_DIR / "target_stats.json", "r") as f: ins_stats = json.load(f)
INS_BAL_MIN, INS_BAL_MAX = float(ins_stats["balance_min"]), float(ins_stats["balance_max"])

# --- Side Hustle Model ---
SH_MODEL = tf.keras.models.load_model(
    ARTIFACT_DIR / "sh_model.keras",
    custom_objects={"CustomDenseBlock": CustomDenseBlock}
)
SH_SCALER = joblib.load(ARTIFACT_DIR / "sh_scaler.joblib")
with open(ARTIFACT_DIR / "sh_feature_columns.json", "r") as f: SH_FEAT_COLS = json.load(f)
with open(ARTIFACT_DIR / "sh_target_stats.json", "r") as f: sh_stats = json.load(f)
SH_EARN_MIN, SH_EARN_MAX = float(sh_stats["earn_min"]), float(sh_stats["earn_max"])

PLATFORMS = sh_stats["platforms"]
PROJECT_TYPES = sh_stats["project_types"]

# --- Classification Model ---
CLS_MODEL = tf.keras.models.load_model(
    ARTIFACT_DIR / "classification_model.keras",
    # Keras 3 menyimpan registered_name `finary>ResidualDenseBlock`, jadi kita map keduanya.
    custom_objects={
        "ResidualDenseBlock": ResidualDenseBlock,
        "finary>ResidualDenseBlock": ResidualDenseBlock,
    },
    compile=False,
)
CLS_SCALER = joblib.load(ARTIFACT_DIR / "classification_scaler.joblib")
with open(ARTIFACT_DIR / "classification_feature_columns.json", "r") as f: CLS_FEAT_COLS = json.load(f)
with open(ARTIFACT_DIR / "classification_label_mapping.json", "r") as f: CLS_LABEL_MAPPING = json.load(f)

# =========================================
# 5. APLIKASI FASTAPI
# =========================================
app = FastAPI(title="FINARY AI Microservices", version="2.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "message": "Classification, Insight, and Side Hustle models loaded."}

# -----------------------------------------
# ENDPOINT 1: CLASSIFICATION (FINANCIAL SCENARIO)
# -----------------------------------------
def build_classification_features(payload: ClassifyRequest) -> tuple[Dict[str, float], Dict[str, float], Dict[str, bool]]:
    # Implementasi ini dibuat agar 100% selaras dengan metode inference minimal
    # yang sudah dibangun di `finary_classify_model.ipynb`.

    def _num(v: Any, default: float) -> float:
        if v is None:
            return float(default)
        return float(v)

    # ==========================
    # 1) Normalisasi unit: IDR -> unit training
    # ==========================
    inc = float(payload.monthly_income) / UNIT_SCALE
    exp = float(payload.monthly_expense_total) / UNIT_SCALE
    savings = float(payload.actual_savings) / UNIT_SCALE
    emergency_fund = float(payload.emergency_fund) / UNIT_SCALE
    budget_goal = float(payload.budget_goal) / UNIT_SCALE

    loan_payment = _num(payload.loan_payment, 0.0) / UNIT_SCALE
    investment_amount = _num(payload.investment_amount, 0.0) / UNIT_SCALE
    rent = _num(payload.rent_or_mortgage, 0.0) / UNIT_SCALE

    discretionary_in = payload.discretionary_spending
    essential_in = payload.essential_spending
    if discretionary_in is None or essential_in is None:
        discretionary = 0.30 * exp
        essential = 0.70 * exp
    else:
        discretionary = float(discretionary_in) / UNIT_SCALE
        essential = float(essential_in) / UNIT_SCALE

    # ==========================
    # 2) Non-uang + default aman
    # ==========================
    credit_score = _num(payload.credit_score, 650.0)
    subscription_services = float(_num(payload.subscription_services, 0.0))
    transaction_count = float(_num(payload.transaction_count, 0.0))
    fraud_flag = float(int(_num(payload.fraud_flag, 0.0)))

    dti = _num(payload.debt_to_income_ratio, (loan_payment / inc if inc > 0 else 0.0))

    # ==========================
    # 3) Engineered features
    # ==========================
    net_cash_flow = inc - exp
    expense_ratio = exp / inc if inc > 0 else 0.0
    savings_rate = savings / inc if inc > 0 else 0.0
    financial_buffer = emergency_fund / exp if exp > 0 else 0.0

    savings_goal_met = 1.0 if savings >= budget_goal else 0.0

    debt_pressure = loan_payment * dti
    spending_efficiency = (essential / exp) if exp > 0 else 0.0
    lifestyle_burden = (discretionary / inc) if inc > 0 else 0.0
    saving_behavior = (savings / budget_goal) if budget_goal > 0 else 0.0

    credit_norm = float(np.clip((credit_score - 300.0) / 550.0, 0.0, 1.0))
    buffer_norm = float(np.clip(financial_buffer / 3.0, 0.0, 1.0))
    advice_score = float(
        np.clip(
            (
                0.35 * credit_norm
                + 0.35 * buffer_norm
                + 0.15 * spending_efficiency
                + 0.15 * float(np.clip(savings_rate / 0.5, 0.0, 1.0))
            )
            * 100.0,
            0.0,
            100.0,
        )
    )

    stress_medium = 1.0 if (expense_ratio > 0.9 or dti >= 0.35) else 0.0

    # One-hot category: hanya yang ada di 27 fitur terpilih
    main_category = (payload.main_category or "").strip().title()
    cat_education = 1.0 if main_category == "Education" else 0.0
    cat_entertainment = 1.0 if main_category == "Entertainment" else 0.0
    cat_transportation = 1.0 if main_category == "Transportation" else 0.0

    # ==========================
    # 4) Mapping ke schema 27 fitur (CLS_FEAT_COLS)
    # ==========================
    features = {col: 0.0 for col in CLS_FEAT_COLS}
    update_map = {
        "saving_behavior": saving_behavior,
        "expense_ratio": expense_ratio,
        "actual_savings": savings,
        "net_cash_flow": net_cash_flow,
        "monthly_income": inc,
        "monthly_expense_total": exp,
        "lifestyle_burden": lifestyle_burden,
        "savings_goal_met": savings_goal_met,
        "spending_efficiency": spending_efficiency,
        "financial_buffer": financial_buffer,
        "discretionary_spending": discretionary,
        "budget_goal": budget_goal,
        "savings_rate": savings_rate,
        "rent_or_mortgage": rent,
        "debt_pressure": debt_pressure,
        "emergency_fund": emergency_fund,
        "category_Education": cat_education,
        "financial_advice_score": advice_score,
        "credit_score": credit_score,
        "category_Entertainment": cat_entertainment,
        "investment_amount": investment_amount,
        "loan_payment": loan_payment,
        "subscription_services": subscription_services,
        "category_Transportation": cat_transportation,
        "financial_stress_level_Medium": stress_medium,
        "fraud_flag": fraud_flag,
        "transaction_count": transaction_count,
    }
    for k, v in update_map.items():
        if k in features:
            features[k] = float(v)

    # ==========================
    # 5) Indikator & risk flags (untuk response)
    # ==========================
    risk_flags = {
        "negative_cash_flow": net_cash_flow < 0,
        "high_expense_ratio": expense_ratio > 0.9,
        "high_debt_ratio": dti >= 0.35,
        "low_emergency_fund": financial_buffer < 1.0,
    }

    indicators = {
        "savings_rate": savings_rate,
        "expense_ratio": expense_ratio,
        "net_cash_flow": net_cash_flow,
        "debt_to_income_ratio": dti,
        "financial_buffer": financial_buffer,
    }

    return features, indicators, risk_flags

def build_classification_recommendations(
    classification: str,
    indicators: Dict[str, float],
    risk_flags: Dict[str, bool],
) -> List[str]:
    recs: List[str] = []

    # Label model klasifikasi: survival / stable / growth
    if classification == "survival":
        recs.extend(
            [
                "kurangi_pengeluaran_non_esensial",
                "buat_rencana_pemulihan_cashflow",
                "prioritaskan_pelunasan_utang",
            ]
        )
    elif classification == "stable":
        recs.extend(
            [
                "pertahankan_disiplin_anggaran",
                "tingkatkan_dana_darurat",
                "optimalkan_subscriptions",
            ]
        )
    elif classification == "growth":
        recs.extend(
            [
                "tingkatkan_investasi_berkala",
                "maksimalkan_tabungan_dan_goal",
                "evaluasi_target_keuangan",
            ]
        )

    if risk_flags.get("negative_cash_flow"):
        recs.append("perbaiki_cashflow_negatif")
    if risk_flags.get("high_expense_ratio"):
        recs.append("turunkan_expense_ratio")
    if risk_flags.get("high_debt_ratio"):
        recs.append("turunkan_rasio_utang")
    if risk_flags.get("low_emergency_fund"):
        recs.append("tingkatkan_dana_darurat")
    if indicators.get("savings_rate", 0.0) < 0.1:
        recs.append("tingkatkan_savings_rate")

    # de-dupe while preserving order
    return list(dict.fromkeys(recs))

def build_classification_explanation(
    classification: str,
    score: float,
    indicators: Dict[str, float],
    risk_flags: Dict[str, bool],
) -> str:
    reasons: List[str] = []

    if risk_flags.get("negative_cash_flow"):
        reasons.append("pengeluaran bulanan melebihi pendapatan bulanan")
    if risk_flags.get("high_expense_ratio"):
        reasons.append("expense ratio tergolong tinggi")
    if risk_flags.get("high_debt_ratio"):
        reasons.append("rasio utang terhadap pendapatan di atas ambang yang direkomendasikan")
    if risk_flags.get("low_emergency_fund"):
        reasons.append("cakupan dana darurat masih rendah")
    if not reasons:
        reasons.append("indikator inti berada pada rentang yang masih terkelola")

    return (
        f"Pengguna diklasifikasikan sebagai {classification} dengan tingkat keyakinan {score:.2f}. "
        f"Faktor utama: {', '.join(reasons)}."
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify_financial_scenario(payload: ClassifyRequest):
    try:
        features, indicators, risk_flags = build_classification_features(payload)

        df_input = pd.DataFrame([features]).reindex(columns=CLS_FEAT_COLS, fill_value=0.0)
        X_scaled = CLS_SCALER.transform(df_input.values)

        pred = CLS_MODEL.predict(X_scaled, verbose=0)[0]
        pred = np.clip(pred, 0.0, 1.0)

        class_id = int(np.argmax(pred))
        score = float(np.max(pred))
        classification = CLS_LABEL_MAPPING[str(class_id)]

        probabilities = {
            CLS_LABEL_MAPPING[str(i)]: round(float(pred[i]), 4)
            for i in range(len(pred))
        }

        recommendations = build_classification_recommendations(
            classification=classification,
            indicators=indicators,
            risk_flags=risk_flags,
        )

        explanation = build_classification_explanation(
            classification=classification,
            score=score,
            indicators=indicators,
            risk_flags=risk_flags,
        )

        return ClassifyResponse(
            classification=classification,
            score=round(score, 4),
            probabilities=probabilities,
            financial_indicators={k: round(float(v), 4) for k, v in indicators.items()},
            risk_flags=risk_flags,
            recommendation_focus=recommendations,
            explanation=explanation,
        )

    except Exception as exc:
        # Error 422 untuk payload yang tidak memenuhi kontrak input minimal
        if isinstance(exc, ValueError) and "Field wajib" in str(exc):
            raise HTTPException(status_code=422, detail=str(exc))
        raise HTTPException(status_code=500, detail=f"Classification inference error: {str(exc)}")

# -----------------------------------------
# ENDPOINT 2: INSIGHT & WARNING
# -----------------------------------------
def build_insight_recs(features_dict: Dict[str, float], warning_prob: float) -> list[str]:
    recs = []
    if features_dict.get("debt_ratio_flag", 0) == 1.0: recs.append("Prioritaskan pelunasan utang berbunga tinggi.")
    if features_dict.get("low_emergency_flag", 0) == 1.0: recs.append("Tingkatkan emergency fund.")
    if warning_prob > 0.7: recs.append("Warning tinggi: batasi transaksi non-esensial selama 2 minggu.")
    if not recs: recs.append("Profil keuangan sehat.")
    return recs

@app.post("/predict", response_model=PredictResponse)
def predict_insight(payload: PredictRequest):
    try:
        inc = payload.income / UNIT_SCALE
        exp = payload.expense / UNIT_SCALE
        sav = payload.savings / UNIT_SCALE
        tgt_sav = payload.target_tabungan / UNIT_SCALE
        loan = payload.loan_payment / UNIT_SCALE
        emg = payload.emergency_fund / UNIT_SCALE
        
        net_cf = inc - exp
        dti = loan / inc if inc > 0 else 0.0
        buffer = emg / exp if exp > 0 else 0.0

        features = {col: 0.0 for col in INS_FEAT_COLS}
        features.update({
            "monthly_income": inc, "monthly_expense_total": exp, "actual_savings": sav,
            "budget_goal": tgt_sav, "loan_payment": loan, "emergency_fund": emg,
            "net_cash_flow": net_cf, "savings_rate": sav / inc if inc > 0 else 0.0,
            "expense_ratio": exp / inc if inc > 0 else 0.0, "debt_to_income_ratio": dti,
            "financial_buffer": buffer, "savings_goal_met": 1.0 if sav >= tgt_sav else 0.0,
            "debt_ratio_flag": 1.0 if dti >= 0.35 else 0.0, "low_emergency_flag": 1.0 if buffer < 1.0 else 0.0
        })
        
        if f"income_type_{payload.income_type}" in features: features[f"income_type_{payload.income_type}"] = 1.0
        if f"category_{payload.main_category}" in features: features[f"category_{payload.main_category}"] = 1.0
        features["cash_flow_status_Positive"] = 1.0 if net_cf > 0 else 0.0
        features["cash_flow_status_Neutral"] = 1.0 if net_cf <= 0 else 0.0

        row_scaled = INS_SCALER.transform(pd.DataFrame([features])[INS_FEAT_COLS].values)
        pred_balance_norm, pred_warning_prob = INS_MODEL.predict(row_scaled, verbose=0)
        
        pred_bal_val = float(np.clip(pred_balance_norm[0][0], 0.0, None))
        pred_warn_val = float(np.clip(pred_warning_prob[0][0], 0.0, 1.0))

        predicted_balance_idr = (pred_bal_val * (INS_BAL_MAX - INS_BAL_MIN) + INS_BAL_MIN) * UNIT_SCALE

        return PredictResponse(
            predicted_next_month_balance=round(predicted_balance_idr, 2),
            warning_probability=round(pred_warn_val, 4),
            warning_flag=int(pred_warn_val >= 0.5),
            recommendations=build_insight_recs(features, pred_warn_val)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# -----------------------------------------
# ENDPOINT 3: SIDE HUSTLE RECOMMENDATION (7 Rekomendasi & Variasi Platform)
# -----------------------------------------
@app.post("/recommend-side-hustle", response_model=SideHustleResponse)
def recommend_side_hustle(payload: SideHustleRequest):
    try:
        # 1. Normalisasi Input
        exp_input = payload.experience_level.strip().title()
        interest_input = payload.interest_category.strip().title()
        
        # 2. Penentuan Rate Berdasarkan Level (Sesuai Standar Freelance)
        rate_map = {"Beginner": 10.0, "Intermediate": 15.0, "Expert": 25.0}
        target_hourly_rate_usd = rate_map.get(exp_input, 15.0)

        # 3. Hitung Total Jam & Durasi Kerja
        total_hours_per_month = payload.available_hours_per_week * 4
        duration_days = total_hours_per_month / 8.0 
        
        # Bobot Platform sesuai tren di dataset (Toptal paling tinggi, Fiverr paling rendah)
        plat_weights = {
            "Toptal": 1.25, "Upwork": 1.15, "Freelancer": 1.0, 
            "PeoplePerHour": 0.95, "Fiverr": 0.85
        }
        # Bobot Project Type (Fixed biasanya memiliki sedikit premi harga)
        type_weights = {"Fixed": 1.1, "Hourly": 1.0}

        simulations = []
        sim_metadata = []
        
        for plat in PLATFORMS:
            for ptype in PROJECT_TYPES:
                feat_map = {col: 0.0 for col in SH_FEAT_COLS}
                
                if "Hourly_Rate" in feat_map: feat_map["Hourly_Rate"] = float(target_hourly_rate_usd)
                if "Job_Duration_Days" in feat_map: feat_map["Job_Duration_Days"] = float(duration_days)
                
                if f"Experience_Level_{exp_input}" in feat_map: feat_map[f"Experience_Level_{exp_input}"] = 1.0
                if f"Job_Category_{interest_input}" in feat_map: feat_map[f"Job_Category_{interest_input}"] = 1.0
                if f"Platform_{plat}" in feat_map: feat_map[f"Platform_{plat}"] = 1.0
                if f"Project_Type_{ptype}" in feat_map: feat_map[f"Project_Type_{ptype}"] = 1.0
                    
                simulations.append(feat_map)
                sim_metadata.append({"platform": plat, "project_type": ptype})

        # --- PROTEKSI ERROR INDEX & CONSISTENCY ---
        df_sim = pd.DataFrame(simulations)
        df_sim = df_sim.reindex(columns=SH_FEAT_COLS, fill_value=0.0)
        
        X_sim_scaled = SH_SCALER.transform(df_sim.values).astype(np.float32)
        tensor_input = tf.constant(X_sim_scaled)
        _, pred_succ_prob = SH_MODEL(tensor_input, training=False)
        
        results = []
        for i, meta in enumerate(sim_metadata):
            succ_prob = float(np.clip(pred_succ_prob[i][0], 0.0, 1.0))
            
            # 4. PERHITUNGAN GAJI BERVARIASI (Berdasarkan Platform & Project Type)
            p_mul = plat_weights.get(meta["platform"], 1.0)
            t_mul = type_weights.get(meta["project_type"], 1.0)
            
            earn_usd = total_hours_per_month * target_hourly_rate_usd * p_mul * t_mul
            earn_idr = earn_usd * USD_TO_IDR
            
            results.append({
                "job_category": interest_input,
                "platform": meta["platform"],
                "project_type": meta["project_type"],
                "predicted_monthly_earnings_idr": round(earn_idr, 2),
                "score": succ_prob 
            })
            
        # Urutkan berdasarkan peluang sukses tertinggi (AI Ranking)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Ambil 7 rekomendasi terbaik
        top_7 = results[:7]
        
        for item in top_7: 
            if "score" in item:
                del item["score"]

        return SideHustleResponse(recommendations=top_7)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(exc)}")
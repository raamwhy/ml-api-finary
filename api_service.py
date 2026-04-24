from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
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
@tf.keras.utils.register_keras_serializable()
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
        self.units = int(units)
        self.dropout = float(dropout)
        self.l2 = float(l2)
        self.activation = str(activation)

        reg = tf.keras.regularizers.l2(self.l2)
        self.d1 = tf.keras.layers.Dense(self.units, kernel_regularizer=reg)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dp1 = tf.keras.layers.Dropout(self.dropout)

        self.d2 = tf.keras.layers.Dense(self.units, kernel_regularizer=reg)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dp2 = tf.keras.layers.Dropout(self.dropout)

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

        y = self.d1(x)
        y = self.bn1(y, training=training)
        y = self._act(y)
        y = self.dp1(y, training=training)

        y = self.d2(y)
        y = self.bn2(y, training=training)
        y = self._act(y)
        y = self.dp2(y, training=training)

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
    monthly_income: float = Field(..., description="Monthly income (IDR)")
    monthly_expense_total: float = Field(..., description="Total monthly expense (IDR)")
    budget_goal: float = Field(0.0, description="Monthly budget or saving goal (IDR)")
    credit_score: float = Field(0.0, description="Credit score")
    debt_to_income_ratio: float = Field(0.0, description="Debt-to-income ratio")
    loan_payment: float = Field(0.0, description="Monthly loan payment (IDR)")
    investment_amount: float = Field(0.0, description="Monthly investment amount (IDR)")
    subscription_services: int = Field(0, description="Number of active subscription services")
    emergency_fund: float = Field(0.0, description="Emergency fund amount (IDR)")
    transaction_count: int = Field(0, description="Monthly transaction count")
    discretionary_spending: float = Field(0.0, description="Non-essential monthly spending (IDR)")
    essential_spending: float = Field(0.0, description="Essential monthly spending (IDR)")
    rent_or_mortgage: float = Field(0.0, description="Monthly rent or mortgage payment (IDR)")
    actual_savings: float = Field(0.0, description="Actual monthly savings (IDR)")
    income_type: str = Field("Salary", description="Salary or Mixed")
    main_category: str = Field("Utilities", description="Main spending category")

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
    custom_objects={"ResidualDenseBlock": ResidualDenseBlock},
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
    # IDR -> training unit scale (konsisten dengan /predict)
    inc = float(payload.monthly_income) / UNIT_SCALE
    exp = float(payload.monthly_expense_total) / UNIT_SCALE
    savings = float(payload.actual_savings) / UNIT_SCALE

    budget_goal = float(payload.budget_goal) / UNIT_SCALE
    loan_payment = float(payload.loan_payment) / UNIT_SCALE
    investment_amount = float(payload.investment_amount) / UNIT_SCALE
    emergency_fund = float(payload.emergency_fund) / UNIT_SCALE
    discretionary = float(payload.discretionary_spending) / UNIT_SCALE
    essential = float(payload.essential_spending) / UNIT_SCALE
    rent = float(payload.rent_or_mortgage) / UNIT_SCALE

    credit_score = float(payload.credit_score)
    dti = float(payload.debt_to_income_ratio)
    subscription_services = float(payload.subscription_services)
    transaction_count = float(payload.transaction_count)

    net_cash_flow = inc - exp
    expense_ratio = exp / inc if inc > 0 else 0.0
    saving_rate = savings / inc if inc > 0 else 0.0
    financial_buffer = emergency_fund / exp if exp > 0 else 0.0

    savings_goal_met = 1.0 if savings >= budget_goal else 0.0
    debt_ratio_flag = 1.0 if dti >= 0.35 else 0.0
    low_emergency_flag = 1.0 if financial_buffer < 1.0 else 0.0

    # Additional dataset features used by the model (best-effort derivations)
    # Keep semantics consistent with dataset scale:
    # - debt_pressure: loan_payment * debt_to_income_ratio
    # - spending_efficiency: essential_spending / monthly_expense_total
    # - lifestyle_burden: discretionary_spending / monthly_income
    # - saving_behavior: actual_savings / budget_goal (proxy for goal progress)
    debt_pressure = loan_payment * dti
    spending_efficiency = (essential / exp) if exp > 0 else 0.0
    lifestyle_burden = (discretionary / inc) if inc > 0 else 0.0
    saving_behavior = (savings / budget_goal) if budget_goal > 0 else 0.0

    # financial_advice_score in dataset is on ~0-100 scale (heuristic proxy)
    # This avoids leaving an always-zero feature that the model expects.
    credit_norm = np.clip((credit_score - 300.0) / 550.0, 0.0, 1.0)  # 300..850 -> 0..1
    buffer_norm = np.clip(financial_buffer / 3.0, 0.0, 1.0)          # cap at 3 months
    advice_score = float(np.clip((0.35 * credit_norm + 0.35 * buffer_norm + 0.15 * spending_efficiency + 0.15 * np.clip(saving_rate / 0.5, 0.0, 1.0)) * 100.0, 0.0, 100.0))

    # fraud_flag is not inferable from request; default to 0 (no fraud)
    fraud_flag = 0.0

    risk_flags = {
        "negative_cash_flow": net_cash_flow < 0,
        "high_expense_ratio": expense_ratio > 0.9,
        "high_debt_ratio": dti >= 0.35,
        "low_emergency_fund": financial_buffer < 1.0,
    }

    indicators = {
        "saving_rate": saving_rate,
        "expense_ratio": expense_ratio,
        "net_cash_flow": net_cash_flow,
        "debt_to_income_ratio": dti,
        "financial_buffer": financial_buffer,
    }

    features = {col: 0.0 for col in CLS_FEAT_COLS}
    update_map = {
        "monthly_income": inc,
        "monthly_expense_total": exp,
        "savings_rate": saving_rate,
        "budget_goal": budget_goal,
        "credit_score": credit_score,
        "debt_to_income_ratio": dti,
        "loan_payment": loan_payment,
        "investment_amount": investment_amount,
        "subscription_services": subscription_services,
        "emergency_fund": emergency_fund,
        "transaction_count": transaction_count,
        "fraud_flag": fraud_flag,
        "discretionary_spending": discretionary,
        "essential_spending": essential,
        "rent_or_mortgage": rent,
        "financial_advice_score": advice_score,
        "actual_savings": savings,
        "net_cash_flow": net_cash_flow,
        "expense_ratio": expense_ratio,
        "debt_pressure": debt_pressure,
        "financial_buffer": financial_buffer,
        "saving_behavior": saving_behavior,
        "spending_efficiency": spending_efficiency,
        "lifestyle_burden": lifestyle_burden,
        "savings_goal_met": savings_goal_met,
        "debt_ratio_flag": debt_ratio_flag,
        "low_emergency_flag": low_emergency_flag,
    }

    for k, v in update_map.items():
        if k in features:
            features[k] = float(v)

    income_type_col = f"income_type_{payload.income_type}"
    if income_type_col in features:
        features[income_type_col] = 1.0

    category_col = f"category_{payload.main_category}"
    if category_col in features:
        features[category_col] = 1.0

    if "cash_flow_status_Positive" in features:
        features["cash_flow_status_Positive"] = 1.0 if net_cash_flow > 0 else 0.0
    if "cash_flow_status_Neutral" in features:
        features["cash_flow_status_Neutral"] = 1.0 if net_cash_flow <= 0 else 0.0

    # Stress level one-hot (best-effort rule)
    # Medium if high expense ratio OR high debt ratio; else Low.
    if "financial_stress_level_Medium" in features:
        features["financial_stress_level_Medium"] = 1.0 if (expense_ratio > 0.9 or dti >= 0.35) else 0.0
    if "financial_stress_level_Low" in features:
        features["financial_stress_level_Low"] = 0.0 if features.get("financial_stress_level_Medium", 0.0) == 1.0 else 1.0

    return features, indicators, risk_flags

def build_classification_recommendations(
    classification: str,
    indicators: Dict[str, float],
    risk_flags: Dict[str, bool],
) -> List[str]:
    recs: List[str] = []

    if classification == "recession":
        recs.extend([
            "reduce_non_essential_expenses",
            "build_cashflow_recovery_plan",
            "prioritize_debt_repayment",
        ])
    elif classification == "inflation":
        recs.extend([
            "review_price_sensitive_expenses",
            "control_lifestyle_spending",
            "optimize_subscription_services",
        ])
    elif classification == "normal":
        recs.extend([
            "maintain_budget_discipline",
            "increase_investment_allocation",
        ])

    if risk_flags.get("negative_cash_flow"):
        recs.append("fix_negative_cash_flow")
    if risk_flags.get("high_expense_ratio"):
        recs.append("reduce_expense_ratio")
    if risk_flags.get("high_debt_ratio"):
        recs.append("lower_debt_to_income_ratio")
    if risk_flags.get("low_emergency_fund"):
        recs.append("increase_emergency_fund")
    if indicators.get("saving_rate", 0.0) < 0.1:
        recs.append("improve_saving_rate")

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
        reasons.append("monthly expenses exceed monthly income")
    if risk_flags.get("high_expense_ratio"):
        reasons.append("expense ratio is high")
    if risk_flags.get("high_debt_ratio"):
        reasons.append("debt-to-income ratio is above the recommended threshold")
    if risk_flags.get("low_emergency_fund"):
        reasons.append("emergency fund coverage is low")
    if not reasons:
        reasons.append("core financial indicators are within a manageable range")

    return (
        f"The user is classified as {classification} with confidence {score:.2f}. "
        f"Key factors: {', '.join(reasons)}."
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
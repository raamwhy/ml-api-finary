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

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "finary_multitask_model.keras"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.json"
TARGET_STATS_PATH = ARTIFACT_DIR / "target_stats.json"

UNIT_SCALE = 2500.0

class PredictRequest(BaseModel):
    income: float = Field(..., description="Pendapatan bulanan (IDR)")
    expense: float = Field(..., description="Total pengeluaran (IDR)")
    savings: float = Field(..., description="Tabungan saat ini (IDR)")
    target_tabungan: float = Field(..., description="Target tabungan (IDR)")
    loan_payment: float = Field(..., description="Total cicilan utang (IDR)")
    emergency_fund: float = Field(..., description="Dana darurat saat ini (IDR)")
    income_type: str = Field("Salary", description="Tipe pendapatan (Salary/Mixed)")
    main_category: str = Field("Utilities", description="Kategori pengeluaran terbesar")

class PredictResponse(BaseModel):
    predicted_next_month_balance: float
    warning_probability: float
    warning_flag: int
    recommendations: List[str]

def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    with open(TARGET_STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return model, scaler, feature_columns, float(stats["balance_min"]), float(stats["balance_max"])

MODEL, SCALER, FEATURE_COLUMNS, BALANCE_MIN, BALANCE_MAX = load_artifacts()

def build_recommendations(features_dict: Dict[str, float], warning_prob: float) -> list[str]:
    recs = []
    if features_dict.get("debt_ratio_flag", 0) == 1.0:
        recs.append("Prioritaskan pelunasan utang berbunga tinggi dengan metode snowball/avalanche.")
    if features_dict.get("low_emergency_flag", 0) == 1.0:
        recs.append("Tingkatkan emergency fund hingga 3-6x pengeluaran bulanan.")
    if features_dict.get("savings_goal_met", 1) == 0.0:
        recs.append("Aktifkan auto-transfer tabungan di awal bulan agar target tabungan tercapai.")
    if features_dict.get("expense_ratio", 0.0) > 0.9:
        recs.append("Turunkan discretionary spending 10-15% untuk memperbaiki rasio pengeluaran.")
    if warning_prob > 0.7:
        recs.append("Warning tinggi: batasi transaksi non-esensial selama 2 minggu.")
    if not recs:
        recs.append("Profil keuangan sehat. Pertimbangkan peningkatan porsi investasi jangka panjang.")
    return recs

app = FastAPI(title="FINARY Insight API", version="1.0.0")

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": "loaded"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
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

        features = {col: 0.0 for col in FEATURE_COLUMNS}
        
        features["monthly_income"] = inc
        features["monthly_expense_total"] = exp
        features["actual_savings"] = sav
        features["budget_goal"] = tgt_sav
        features["loan_payment"] = loan
        features["emergency_fund"] = emg
        features["net_cash_flow"] = net_cf
        features["savings_rate"] = sav / inc if inc > 0 else 0.0
        features["expense_ratio"] = exp / inc if inc > 0 else 0.0
        features["debt_to_income_ratio"] = dti
        features["debt_pressure"] = dti * loan
        features["financial_buffer"] = buffer
        features["saving_behavior"] = sav / tgt_sav if tgt_sav > 0 else 0.0
        features["savings_goal_met"] = 1.0 if sav >= tgt_sav else 0.0
        features["debt_ratio_flag"] = 1.0 if dti >= 0.35 else 0.0
        features["low_emergency_flag"] = 1.0 if buffer < 1.0 else 0.0
        
        if f"income_type_{payload.income_type}" in features: 
            features[f"income_type_{payload.income_type}"] = 1.0
        if f"category_{payload.main_category}" in features: 
            features[f"category_{payload.main_category}"] = 1.0
        if net_cf > 0: 
            features["cash_flow_status_Positive"] = 1.0
        else: 
            features["cash_flow_status_Neutral"] = 1.0

        row_df = pd.DataFrame([features])[FEATURE_COLUMNS].astype(np.float32)
        row_scaled = SCALER.transform(row_df.values)

        pred_balance_norm, pred_warning_prob = MODEL.predict(row_scaled, verbose=0)
        
        pred_balance_norm_val = float(np.clip(pred_balance_norm[0][0], 0.0, None)) # Ganti None
        pred_warning_prob_val = float(np.clip(pred_warning_prob[0][0], 0.0, 1.0))

        predicted_balance_idr = (pred_balance_norm_val * (BALANCE_MAX - BALANCE_MIN) + BALANCE_MIN) * UNIT_SCALE
        warning_flag = int(pred_warning_prob_val >= 0.5)

        return PredictResponse(
            predicted_next_month_balance=round(predicted_balance_idr, 2),
            warning_probability=round(pred_warning_prob_val, 4),
            warning_flag=warning_flag,
            recommendations=build_recommendations(features, pred_warning_prob_val)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
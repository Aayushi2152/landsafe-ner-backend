"""
LandSafe NER — AI-Based Landslide Risk Monitoring & Early Warning System
Backend (FastAPI + scikit-learn Random Forest + SQLite)

Run:
    python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

First-time / whenever you change train_model.py:
    python train_model.py     # trains and saves model/*.pkl

Endpoints (see /docs for interactive Swagger UI):
    GET  /                          -> dashboard (index.html)
    GET  /health                    -> service + model status
    POST /api/predict               -> AI risk prediction for arbitrary coords
    GET  /api/demo/live             -> simulated live sensor reading + prediction
    GET  /api/zones                 -> list monitored zones (+ latest risk)
    POST /api/zones                 -> add a new monitored zone
    GET  /api/zones/{id}/history    -> risk history for one zone (for charts)
    GET  /api/alerts                -> list generated alerts
    POST /api/alerts/{id}/ack       -> acknowledge an alert
    POST /api/reports               -> submit a citizen/field hazard report
    GET  /api/reports               -> list field reports
    PATCH /api/reports/{id}         -> update report status (authority side)
"""

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import db

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"

app = FastAPI(title="LandSafe NER", version="2.0",
              description="AI-based early warning and landslide risk monitoring backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
MODEL_LOADED = False
clf = reg = scaler = encoder = None
FEATURE_COLUMNS = ["rainfall_mm", "rainfall_3day_mm", "soil_moisture_pct", "slope_deg", "elevation_m"]

try:
    clf = joblib.load(MODEL_DIR / "risk_classifier.pkl")
    reg = joblib.load(MODEL_DIR / "risk_regressor.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    FEATURE_COLUMNS = joblib.load(MODEL_DIR / "feature_columns.pkl")
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False  # falls back to the transparent formula below


def formula_fallback_risk(rainfall, rainfall_3day, soil_moisture, slope, elevation):
    """Transparent backup scorer, used only if model/*.pkl hasn't been trained yet."""
    rain = min(rainfall / 220, 1)
    antecedent = min(rainfall_3day / 500, 1)
    moisture = soil_moisture / 100
    slope_t = min(slope / 45, 1)
    elevation_t = min(max((elevation - 300) / 2200, 0), 1)
    score = 0.30 * rain + 0.20 * antecedent + 0.25 * moisture + 0.20 * slope_t + 0.05 * elevation_t
    return round(score * 100, 1)


def bucket(p):
    if p >= 80:
        return "CRITICAL"
    elif p >= 60:
        return "HIGH"
    elif p >= 30:
        return "MEDIUM"
    return "LOW"


def run_model(rainfall_mm, rainfall_3day_mm, soil_moisture_pct, slope_deg, elevation_m):
    """Returns (risk_probability_pct, risk_level, source) where source is
    'ml_model' if the trained Random Forest was used, or 'fallback_formula'."""
    if MODEL_LOADED:
        row = pd.DataFrame(
            [[rainfall_mm, rainfall_3day_mm, soil_moisture_pct, slope_deg, elevation_m]],
            columns=FEATURE_COLUMNS,
        )
        row_s = scaler.transform(row)
        level = encoder.inverse_transform(clf.predict(row_s))[0]
        prob = float(np.clip(reg.predict(row_s)[0], 0, 100))
        return round(prob, 1), level, "ml_model"
    else:
        prob = formula_fallback_risk(rainfall_mm, rainfall_3day_mm, soil_moisture_pct, slope_deg, elevation_m)
        return prob, bucket(prob), "fallback_formula"


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class RiskInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    rainfall_mm: float = Field(..., ge=0, description="24-hour rainfall (mm)")
    rainfall_3day_mm: Optional[float] = Field(None, ge=0, description="3-day cumulative rainfall (mm). Estimated from rainfall_mm if omitted.")
    soil_moisture_pct: float = Field(..., ge=0, le=100)
    slope_deg: float = Field(..., ge=0, le=90)
    elevation_m: float = Field(..., ge=0)
    zone_id: Optional[int] = Field(None, description="Attach this reading to an existing monitored zone")


class ZoneCreate(BaseModel):
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class ReportCreate(BaseModel):
    reporter_name: Optional[str] = None
    hazard_type: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    description: Optional[str] = None
    photo_url: Optional[str] = None


class ReportStatusUpdate(BaseModel):
    status: str  # NEW, REVIEWING, VERIFIED, RESOLVED, DISMISSED


ALERT_THRESHOLD = 80.0  # risk_probability_pct at/above which an alert is auto-generated

# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    db.init_db(seed=True)


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "LandSafe NER",
        "model_loaded": MODEL_LOADED,
        "prediction_source": "ml_model" if MODEL_LOADED else "fallback_formula (run train_model.py to enable ML)",
    }


@app.post("/api/predict")
def predict(x: RiskInput):
    rainfall_3day = x.rainfall_3day_mm if x.rainfall_3day_mm is not None else round(x.rainfall_mm * 1.8, 1)
    prob, level, source = run_model(x.rainfall_mm, rainfall_3day, x.soil_moisture_pct, x.slope_deg, x.elevation_m)

    zone_name = None
    if x.zone_id is not None:
        zones = {z["id"]: z for z in db.list_zones()}
        if x.zone_id not in zones:
            raise HTTPException(404, f"zone_id {x.zone_id} not found")
        zone_name = zones[x.zone_id]["name"]

    features = {
        "rainfall_mm": x.rainfall_mm, "rainfall_3day_mm": rainfall_3day,
        "soil_moisture_pct": x.soil_moisture_pct, "slope_deg": x.slope_deg,
        "elevation_m": x.elevation_m,
    }
    db.save_prediction(x.zone_id, x.latitude, x.longitude, features, prob, level)

    alert_created = False
    if prob >= ALERT_THRESHOLD:
        msg = f"Risk reached {prob}% ({level}) — immediate field inspection recommended."
        db.create_alert(x.zone_id, zone_name, x.latitude, x.longitude, prob, level, msg)
        alert_created = True

    return {
        "location": {"latitude": x.latitude, "longitude": x.longitude},
        "inputs": {**features},
        "prediction": {"risk_probability_pct": prob, "risk_level": level, "source": source},
        "alert_required": prob >= ALERT_THRESHOLD,
        "alert_created": alert_created,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/demo/live")
def demo_live():
    """Simulates a live sensor feed for demo purposes (no hardware required)."""
    rainfall = round(random.uniform(20, 260), 1)
    rainfall_3day = round(rainfall * random.uniform(1.2, 2.6), 1)
    x = RiskInput(
        latitude=27.33, longitude=88.61,
        rainfall_mm=rainfall, rainfall_3day_mm=rainfall_3day,
        soil_moisture_pct=round(random.uniform(25, 95), 1),
        slope_deg=round(random.uniform(10, 48), 1),
        elevation_m=round(random.uniform(500, 2500), 0),
    )
    prob, level, source = run_model(x.rainfall_mm, x.rainfall_3day_mm, x.soil_moisture_pct, x.slope_deg, x.elevation_m)
    return {
        "location": {"latitude": x.latitude, "longitude": x.longitude},
        "environment": {
            "rainfall_mm": x.rainfall_mm, "rainfall_3day_mm": x.rainfall_3day_mm,
            "soil_moisture_pct": x.soil_moisture_pct, "slope_deg": x.slope_deg,
            "elevation_m": x.elevation_m,
        },
        "prediction": {"risk_probability_pct": prob, "risk_level": level, "source": source},
        "alert_required": prob >= ALERT_THRESHOLD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# Zones (GIS dashboard backing data)
# --------------------------------------------------------------------------
@app.get("/api/zones")
def get_zones():
    zones = db.list_zones()
    for z in zones:
        latest = db.zone_latest_prediction(z["id"])
        z["latest_prediction"] = latest
    return {"zones": zones, "count": len(zones)}


@app.post("/api/zones")
def add_zone(z: ZoneCreate):
    zone_id = db.create_zone(z.name, z.latitude, z.longitude)
    return {"id": zone_id, "name": z.name, "latitude": z.latitude, "longitude": z.longitude}


@app.get("/api/zones/{zone_id}/history")
def zone_history(zone_id: int, limit: int = 50):
    return {"zone_id": zone_id, "history": db.prediction_history(zone_id=zone_id, limit=limit)}


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------
@app.get("/api/alerts")
def get_alerts(unacknowledged_only: bool = False):
    return {"alerts": db.list_alerts(only_unacknowledged=unacknowledged_only)}


@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    ok = db.acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(404, "alert not found")
    return {"id": alert_id, "acknowledged": True}


# --------------------------------------------------------------------------
# Citizen / field reports
# --------------------------------------------------------------------------
@app.post("/api/reports")
def submit_report(r: ReportCreate):
    report_id = db.create_report(r.reporter_name, r.hazard_type, r.latitude, r.longitude, r.description, r.photo_url)
    return {"id": report_id, "status": "NEW", "message": "Report received"}


@app.get("/api/reports")
def get_reports():
    return {"reports": db.list_reports()}


@app.patch("/api/reports/{report_id}")
def patch_report(report_id: int, body: ReportStatusUpdate):
    valid = {"NEW", "REVIEWING", "VERIFIED", "RESOLVED", "DISMISSED"}
    if body.status not in valid:
        raise HTTPException(400, f"status must be one of {sorted(valid)}")
    ok = db.update_report_status(report_id, body.status)
    if not ok:
        raise HTTPException(404, "report not found")
    return {"id": report_id, "status": body.status}

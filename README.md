# LandSafe NER — Backend (v2)

AI-based landslide risk monitoring backend for SIH 2026 (PS ID 26001), built with
**FastAPI + scikit-learn (Random Forest) + SQLite**.

This upgrades your original prototype (a hand-written linear scoring formula) into
a proper ML pipeline plus the data layer your PPT's "Proposed Solution" slide
promises: AI risk prediction, a GIS-ready zones API, automatic alerting, and
citizen/field reporting — all backed by a real (embedded) database.

## What changed vs. your original app.py

| Before | Now |
|---|---|
| Linear formula (`0.4*rain + 0.3*moisture + ...`) | Trained `RandomForestClassifier` (risk level) + `RandomForestRegressor` (risk %) |
| No persistence | SQLite (`landsafe.db`) — zones, prediction history, alerts, field reports |
| Single demo endpoint | Full API: zones, zone history, alerts, citizen reports |
| No historical/antecedent rainfall | Adds 3-day antecedent rainfall (a key driver per your literature review) |

The original `/api/predict` and `/api/demo/live` response shapes are preserved
(with extra fields added), so your existing `index.html` keeps working
unmodified — it'll just be getting real ML predictions instead of the formula.

## Setup

```bash
pip install -r requirements.txt
python train_model.py     # trains the model, saves to model/*.pkl (~10s)
python -m uvicorn app:app --reload --port 8000
```

Or on Windows, just double-click `START_SERVER.bat` — it installs deps, trains
the model on first run, and starts the server.

Visit:
- `http://127.0.0.1:8000` — dashboard
- `http://127.0.0.1:8000/docs` — interactive Swagger API docs (great for your demo)

## About the training data

There's no clean public historical landslide dataset for NER readily available,
so `train_model.py` generates a **physics-informed synthetic dataset**: labels
come from the same rainfall + antecedent-moisture + slope + elevation
relationships your literature review cites (Abraham et al. 2021, Gupta & Satyam
2024, etc.), with noise and a slope×saturation interaction term added so the
model has to genuinely learn the pattern, not just memorize a formula.
Validation accuracy is ~83% (see console output when you run `train_model.py`).

**Say this plainly to the judges**: the ML pipeline (feature engineering,
train/test split, Random Forest, evaluation metrics) is real and production-shaped;
only the training *labels* are synthetic pending a real historical dataset. Swapping
in real data later means only editing `train_model.py` — the rest of the backend
doesn't change. This is a completely normal, honest thing to say in an SIH prototype.

### To plug in real data later
1. Put a CSV at `data/historical_landslides.csv` with columns:
   `rainfall_mm, rainfall_3day_mm, soil_moisture_pct, slope_deg, elevation_m, risk_level`
   (or `risk_probability_pct` — see `train_model.py`).
2. Set `USE_SYNTHETIC = False` in `train_model.py`.
3. Re-run `python train_model.py`.

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | service + model status |
| POST | `/api/predict` | AI risk prediction for given coordinates/conditions |
| GET | `/api/demo/live` | simulated live sensor reading (no hardware needed for demo) |
| GET | `/api/zones` | list monitored zones with each zone's latest prediction |
| POST | `/api/zones` | register a new monitored zone |
| GET | `/api/zones/{id}/history` | risk history for one zone (for a trend chart) |
| GET | `/api/alerts` | list generated alerts (auto-created when risk ≥ 80%) |
| POST | `/api/alerts/{id}/ack` | mark an alert as acknowledged |
| POST | `/api/reports` | submit a citizen/field hazard report |
| GET | `/api/reports` | list field reports |
| PATCH | `/api/reports/{id}` | update a report's status (authority workflow) |

## Files

```
app.py            FastAPI app — all routes
db.py             SQLite data layer (zones, predictions, alerts, reports)
train_model.py    Dataset generation + Random Forest training
model/            Saved model artifacts (created by train_model.py)
data/             Generated synthetic dataset (for transparency/inspection)
index.html        Your existing dashboard (unchanged, still works)
requirements.txt
START_SERVER.bat
```
## Screenshots

### Dashboard Overview
![Dashboard](pictorials/dashboard.png)

### GIS Risk Map
![GIS Risk Map](pictorials/GIS.png)

### Zone Details
![Zone Details](pictorials/zone_details.png)

### Alert Center
![Alerts](pictorials/Alert.png)

### Recent Alerts (Dashboard Widget)
![Recent Alerts](pictorials/alerts.png)

### Field / Citizen Report
![Field Report](pictorials/Report.png)

### System Architecture
![System Architecture](pictorials/Architecture.png)

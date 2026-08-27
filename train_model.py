"""
LandSafe NER — Model Training Script
=====================================
Trains a Random Forest classifier that predicts landslide risk level
(LOW / MEDIUM / HIGH / CRITICAL) from environmental features.

WHY SYNTHETIC DATA:
Public, ready-to-use historical landslide datasets for the NER region are not
freely available in a clean tabular form. For the SIH prototype we generate a
physics-informed synthetic dataset: labels are derived from a domain-based
rule (grounded in the geotechnical relationships cited in your PPT's
"Research & References" slide — rainfall, antecedent soil moisture, slope,
elevation), then Gaussian noise + random feature interactions are added so
the model has to *learn* the pattern rather than memorize a formula.

>>> HOW TO SWITCH TO REAL DATA LATER <<<
If your team gets a real historical landslide CSV (e.g. from GSI Bhukosh,
NRSC Bhuvan, or state disaster management data), just:
  1. Put it at data/historical_landslides.csv with columns matching
     FEATURE_COLUMNS below (rename as needed).
  2. Set USE_SYNTHETIC = False.
  3. Re-run: python train_model.py
No other file needs to change — the FastAPI backend only depends on the
saved model.pkl/scaler.pkl/encoder.pkl artifacts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

USE_SYNTHETIC = True
N_SAMPLES = 6000
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "rainfall_mm",          # 24h rainfall
    "rainfall_3day_mm",     # antecedent (3-day cumulative) rainfall
    "soil_moisture_pct",
    "slope_deg",
    "elevation_m",
]


def generate_synthetic_dataset(n=N_SAMPLES, seed=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rainfall = rng.gamma(shape=2.0, scale=45, size=n).clip(0, 400)
    rainfall_3day = (rainfall * rng.uniform(1.2, 2.8, n) + rng.normal(0, 20, n)).clip(0, 900)
    soil_moisture = (30 + 0.15 * rainfall_3day + rng.normal(0, 12, n)).clip(5, 100)
    slope = rng.uniform(0, 60, n)
    elevation = rng.uniform(200, 3000, n)

    # Physics-informed latent risk score (mirrors geotechnical drivers cited
    # in the literature review: rainfall + antecedent moisture + slope are the
    # dominant triggers; elevation is a smaller proxy for terrain exposure).
    rain_term = np.minimum(rainfall / 220, 1.0)
    antecedent_term = np.minimum(rainfall_3day / 500, 1.0)
    moisture_term = soil_moisture / 100
    slope_term = np.minimum(slope / 45, 1.0)
    elevation_term = np.clip((elevation - 300) / 2200, 0, 1)

    latent = (
        0.30 * rain_term
        + 0.20 * antecedent_term
        + 0.25 * moisture_term
        + 0.20 * slope_term
        + 0.05 * elevation_term
    )
    # Nonlinear interaction: saturated soil + steep slope compounds risk
    latent += 0.15 * (moisture_term > 0.75) * (slope_term > 0.6)
    latent += rng.normal(0, 0.06, n)  # measurement/model noise
    latent = np.clip(latent, 0, 1.3)

    risk_pct = np.clip(latent * 100, 0, 100)

    def bucket(p):
        if p >= 80:
            return "CRITICAL"
        elif p >= 60:
            return "HIGH"
        elif p >= 30:
            return "MEDIUM"
        return "LOW"

    labels = np.array([bucket(p) for p in risk_pct])

    df = pd.DataFrame({
        "rainfall_mm": rainfall.round(1),
        "rainfall_3day_mm": rainfall_3day.round(1),
        "soil_moisture_pct": soil_moisture.round(1),
        "slope_deg": slope.round(1),
        "elevation_m": elevation.round(0),
        "risk_probability_pct": risk_pct.round(1),
        "risk_level": labels,
    })
    return df


def load_dataset() -> pd.DataFrame:
    csv_path = DATA_DIR / "historical_landslides.csv"
    if not USE_SYNTHETIC and csv_path.exists():
        print(f"Loading real dataset from {csv_path}")
        return pd.read_csv(csv_path)
    print("Generating physics-informed synthetic training dataset...")
    df = generate_synthetic_dataset()
    df.to_csv(DATA_DIR / "synthetic_training_data.csv", index=False)
    return df


def train():
    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y_class = df["risk_level"]
    y_reg = df["risk_probability_pct"]

    encoder = LabelEncoder()
    y_class_enc = encoder.fit_transform(y_class)

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_class_enc, y_reg, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Classifier: predicts risk level bucket
    clf = RandomForestClassifier(
        n_estimators=120, max_depth=9, min_samples_leaf=5,
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train_s, yc_train)

    # Regressor: predicts continuous risk probability (0-100) for finer-grained UI display
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(
        n_estimators=120, max_depth=9, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    reg.fit(X_train_s, yr_train)

    preds = clf.predict(X_test_s)
    acc = accuracy_score(yc_test, preds)
    print(f"\nValidation accuracy: {acc:.3f}\n")
    print(classification_report(yc_test, preds, target_names=encoder.classes_))

    importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.round(3)))
    print("Feature importances:", importances)

    joblib.dump(clf, MODEL_DIR / "risk_classifier.pkl", compress=3)
    joblib.dump(reg, MODEL_DIR / "risk_regressor.pkl", compress=3)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl", compress=3)
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl", compress=3)
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_columns.pkl", compress=3)

    print(f"\nSaved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    train()

"""
LandSafe NER — Database layer (SQLite, zero external services needed).
Stores: monitored zones, prediction history, alerts, citizen/field reports.
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

# On Vercel (and most serverless platforms) the deployed code directory is
# READ-ONLY at runtime — only /tmp is writable, and it's wiped between cold
# starts (so data won't persist across invocations there). Locally, we keep
# using a normal file next to this script so your data persists as before.
if os.environ.get("VERCEL"):
    DB_PATH = Path("/tmp/landsafe.db")
else:
    DB_PATH = Path(__file__).parent / "landsafe.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    rainfall_mm REAL,
    rainfall_3day_mm REAL,
    soil_moisture_pct REAL,
    slope_deg REAL,
    elevation_m REAL,
    risk_probability_pct REAL NOT NULL,
    risk_level TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (zone_id) REFERENCES zones (id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER,
    zone_name TEXT,
    latitude REAL,
    longitude REAL,
    risk_probability_pct REAL NOT NULL,
    risk_level TEXT NOT NULL,
    message TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS field_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reporter_name TEXT,
    hazard_type TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    description TEXT,
    photo_url TEXT,
    status TEXT NOT NULL DEFAULT 'NEW',
    created_at TEXT NOT NULL
);
"""

SEED_ZONES = [
    ("NH-10 Corridor", 27.02, 88.26),
    ("Gangtok Ridge", 27.33, 88.61),
    ("Kalimpong Slope", 27.06, 88.47),
    ("Zone 04 - Mangan Road", 27.51, 88.53),
    ("Rangpo River Bend", 27.17, 88.53),
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(seed: bool = True):
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        if seed:
            existing = conn.execute("SELECT COUNT(*) c FROM zones").fetchone()["c"]
            if existing == 0:
                for name, lat, lon in SEED_ZONES:
                    conn.execute(
                        "INSERT INTO zones (name, latitude, longitude, created_at) VALUES (?,?,?,?)",
                        (name, lat, lon, now_iso()),
                    )


# ---------- Zones ----------

def list_zones():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM zones ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def create_zone(name, latitude, longitude):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO zones (name, latitude, longitude, created_at) VALUES (?,?,?,?)",
            (name, latitude, longitude, now_iso()),
        )
        return cur.lastrowid


def zone_latest_prediction(zone_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE zone_id=? ORDER BY id DESC LIMIT 1",
            (zone_id,),
        ).fetchone()
        return dict(row) if row else None


# ---------- Predictions ----------

def save_prediction(zone_id, latitude, longitude, features: dict, risk_pct, risk_level):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
            (zone_id, latitude, longitude, rainfall_mm, rainfall_3day_mm,
             soil_moisture_pct, slope_deg, elevation_m,
             risk_probability_pct, risk_level, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                zone_id, latitude, longitude,
                features.get("rainfall_mm"), features.get("rainfall_3day_mm"),
                features.get("soil_moisture_pct"), features.get("slope_deg"),
                features.get("elevation_m"),
                risk_pct, risk_level, now_iso(),
            ),
        )
        return cur.lastrowid


def prediction_history(zone_id=None, limit=50):
    with get_conn() as conn:
        if zone_id is not None:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE zone_id=? ORDER BY id DESC LIMIT ?",
                (zone_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


# ---------- Alerts ----------

def create_alert(zone_id, zone_name, latitude, longitude, risk_pct, risk_level, message):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO alerts
            (zone_id, zone_name, latitude, longitude, risk_probability_pct,
             risk_level, message, acknowledged, created_at)
            VALUES (?,?,?,?,?,?,?,0,?)""",
            (zone_id, zone_name, latitude, longitude, risk_pct, risk_level, message, now_iso()),
        )
        return cur.lastrowid


def list_alerts(only_unacknowledged=False, limit=100):
    with get_conn() as conn:
        q = "SELECT * FROM alerts"
        if only_unacknowledged:
            q += " WHERE acknowledged=0"
        q += " ORDER BY id DESC LIMIT ?"
        rows = conn.execute(q, (limit,)).fetchall()
        return [dict(r) for r in rows]


def acknowledge_alert(alert_id):
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
        return conn.execute("SELECT changes()").fetchone()[0] > 0


# ---------- Field reports ----------

def create_report(reporter_name, hazard_type, latitude, longitude, description, photo_url=None):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO field_reports
            (reporter_name, hazard_type, latitude, longitude, description, photo_url, status, created_at)
            VALUES (?,?,?,?,?,?,'NEW',?)""",
            (reporter_name, hazard_type, latitude, longitude, description, photo_url, now_iso()),
        )
        return cur.lastrowid


def list_reports(limit=100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM field_reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def update_report_status(report_id, status):
    with get_conn() as conn:
        conn.execute("UPDATE field_reports SET status=? WHERE id=?", (status, report_id))
        return conn.execute("SELECT changes()").fetchone()[0] > 0
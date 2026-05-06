#!/usr/bin/env python3
"""
Import existing CSV telemetry data into the SQLite database.

Reads all CSV files from data/, creates vehicle_sessions and
telemetry_readings, then runs the Isolation Forest anomaly
detector to populate diagnostic_reports.

Usage:
    python scripts/import_csv_to_db.py
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.database import (
    Base,
    VehicleSession,
    TelemetryReading,
    DiagnosticReport,
    init_db,
    get_engine,
    get_session_factory,
)

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "telemetry.db"


def import_csv_files():
    """Import all CSV files from the data directory into the database."""
    # Initialize database
    print("=" * 60)
    print("CSV → SQLite Import")
    print("=" * 60)

    # Remove old database if it exists (fresh import)
    if DB_PATH.exists():
        os.remove(DB_PATH)
        print(f"  Removed existing database: {DB_PATH}")

    engine = init_db(str(DB_PATH))
    Session = get_session_factory(engine)
    db = Session()

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("  No CSV files found in data/")
        return

    print(f"  Found {len(csv_files)} CSV file(s)")

    total_readings = 0
    total_sessions = 0

    for csv_file in csv_files:
        print(f"\n  Processing: {csv_file.name}")

        # Group rows by session UUID
        sessions_data = defaultdict(list)

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sessions_data[row["vehicle_session_id"]].append(row)

        print(f"    Sessions found: {len(sessions_data)}")

        for session_uuid, rows in sessions_data.items():
            timestamps = [
                datetime.fromisoformat(r["timestamp"]) for r in rows
            ]

            # Create session record
            session = VehicleSession(
                session_uuid=session_uuid,
                start_time=min(timestamps),
                end_time=max(timestamps),
                reading_count=len(rows),
            )
            db.add(session)
            db.flush()  # get the auto-generated ID

            # Create telemetry readings
            for row in rows:
                reading = TelemetryReading(
                    session_id=session.id,
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    pid=row["pid"],
                    name=row["name"],
                    value=float(row["value"]),
                    unit=row["unit"],
                )
                db.add(reading)

            total_readings += len(rows)
            total_sessions += 1

    db.commit()
    print(f"\n  ✓ Imported {total_sessions} sessions, {total_readings:,} readings")

    # ── Run anomaly detection and save results ──
    print("\n  Running anomaly detection on each session...")
    run_anomaly_detection(db)

    db.close()
    engine.dispose()

    # Print database size
    db_size = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  Database: {DB_PATH}")
    print(f"  Size: {db_size:.1f} MB")
    print("=" * 60)


def run_anomaly_detection(db):
    """Run Isolation Forest on each session and save DiagnosticReport records."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    sessions = db.query(VehicleSession).all()
    features = ["engine_rpm", "coolant_temp", "throttle_pos", "engine_load", "vehicle_speed"]

    # Load ALL readings for training the model
    all_readings = db.query(TelemetryReading).all()

    # Pivot into wide format
    rows_by_ts = defaultdict(dict)
    for r in all_readings:
        rows_by_ts[(r.session_id, r.timestamp.isoformat())][r.name] = r.value

    all_data = []
    for key, vals in rows_by_ts.items():
        if all(f in vals for f in features):
            all_data.append([vals[f] for f in features])

    if not all_data:
        print("    No complete feature rows found, skipping ML.")
        return

    X_all = np.array(all_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Train model
    model = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    model.fit(X_scaled)

    # Score each session
    for session in sessions:
        readings = db.query(TelemetryReading).filter_by(session_id=session.id).all()

        # Pivot this session's readings
        session_rows_by_ts = defaultdict(dict)
        for r in readings:
            session_rows_by_ts[r.timestamp.isoformat()][r.name] = r.value

        session_data = []
        for ts, vals in session_rows_by_ts.items():
            if all(f in vals for f in features):
                session_data.append([vals[f] for f in features])

        if not session_data:
            continue

        X_session = scaler.transform(np.array(session_data))
        scores = model.decision_function(X_session)
        preds = model.predict(X_session)

        anomaly_pct = (preds == -1).sum() / len(preds) * 100
        avg_score = float(scores.mean())

        if anomaly_pct > 15:
            severity = "HIGH"
        elif anomaly_pct > 5:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        n_anomalies = int((preds == -1).sum())
        summary = (
            f"Analyzed {len(session_data)} readings across 5 PIDs. "
            f"Detected {n_anomalies} anomalous samples ({anomaly_pct:.1f}%). "
            f"Mean anomaly score: {avg_score:.4f}. "
            f"Severity: {severity}."
        )

        report = DiagnosticReport(
            session_id=session.id,
            anomaly_score=round(avg_score, 6),
            severity=severity,
            summary=summary,
            model_used="isolation_forest_v1",
        )
        db.add(report)
        print(f"    Session {session.session_uuid[:8]}…: {severity} ({anomaly_pct:.1f}% anomalous)")

    db.commit()
    print(f"  ✓ Created {len(sessions)} diagnostic reports")


if __name__ == "__main__":
    import_csv_files()

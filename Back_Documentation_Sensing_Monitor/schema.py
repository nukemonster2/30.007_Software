"""
Database schema for the ESP32 palpation-device pipeline.

This was missing entirely from the original codebase — get_connection()
and init_indexes() assumed the tables already existed, but there was no
CREATE TABLE anywhere, so the pipeline crashed on a fresh database with
'no such table: patients'.

Call init_db() once at startup (esp32_receiver.py does this) before any
other db.py function is used.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    patient_id       TEXT NOT NULL,
    session_date     TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    ended_at         TEXT,
    administered_by  TEXT,
    composite_score  REAL,
    photo_path       TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS readings (
    reading_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                 TEXT NOT NULL,
    taken_at                   TEXT NOT NULL,
    landmark                   TEXT NOT NULL,
    side                       TEXT NOT NULL,
    frequency                  REAL,
    stiffness                  REAL,
    asymmetry_idx_frequency    REAL,
    asymmetry_idx_stiffness    REAL,
    flagged                    INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_readings_session ON readings(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_readings_unique
    ON readings(session_id, taken_at, landmark, side);

CREATE TABLE IF NOT EXISTS reports (
    report_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL UNIQUE,
    generated_at TEXT DEFAULT (datetime('now')),
    heatmap_path TEXT,
    pdf_path     TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()
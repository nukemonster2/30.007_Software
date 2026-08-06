"""
Database schema for the ESP32 palpation-device pipeline.

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
    photo_path       TEXT,
    composite_score  REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS readings (
    reading_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                TEXT NOT NULL,
    taken_at                  TEXT NOT NULL,
    landmark                  TEXT NOT NULL,
    side                      TEXT NOT NULL,
    frequency                  REAL,
    stiffness                  REAL,
    acceleration                REAL,
    asymmetry_idx_frequency    REAL,
    asymmetry_idx_stiffness    REAL,
    flagged                   INTEGER DEFAULT 0,
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
    delta_path   TEXT,
    pdf_path     TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

# Columns added after this schema was first deployed. executescript()'s
# "CREATE TABLE IF NOT EXISTS" only helps on a brand-new database -- an
# EXISTING database (e.g. one already holding real session data) needs
# these added explicitly, or code expecting them (heatmap.py, the new
# pipeline.py) will hit "no such column" instead of just working.
_MIGRATIONS = {
    "sessions": [
        ("photo_path", "TEXT"),
    ],
    "readings": [
        ("frequency", "REAL"),
        ("stiffness", "REAL"),
        ("acceleration", "REAL"),
        ("asymmetry_idx_frequency", "REAL"),
        ("asymmetry_idx_stiffness", "REAL"),
    ],
    "reports": [
        ("delta_path", "TEXT"),
    ],
}


def _apply_migrations(conn):
    for table, columns in _MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column_name, column_type in columns:
            if column_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")

    # CREATE TABLE IF NOT EXISTS can't retroactively add a UNIQUE constraint
    # to a table that already exists without one (unlike ADD COLUMN, SQLite
    # has no ALTER TABLE ADD UNIQUE) -- a CREATE UNIQUE INDEX is the
    # equivalent fix, and is what report.py's ON CONFLICT(session_id)
    # upsert actually needs to work.
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_session ON reports(session_id)")


def init_db(conn):
    conn.executescript(SCHEMA)
    _apply_migrations(conn)
    conn.commit()
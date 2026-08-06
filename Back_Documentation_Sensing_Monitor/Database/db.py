import sqlite3

from pathlib import Path

from Database.schema import init_db

DEBUG = True


def _debug(message):
    if DEBUG:
        print(f"[DEBUG] {message}")

# Always resolve to the real database at the project root, regardless of
# what folder you happen to run a command from. (Database/db.py -> parent
# of Database -> project root -> 30.007.db)
DB_NAME = str(Path(__file__).resolve().parent.parent / "30.007.db")

print("Using database:")
print(Path(DB_NAME).resolve())

def _ensure_schema(conn):
    try:
        init_db(conn)
    except Exception:
        pass

    cols = {row[1] for row in conn.execute("PRAGMA table_info(readings)").fetchall()}
    if "asymmetry_idx_frequency" not in cols:
        conn.execute("ALTER TABLE readings ADD COLUMN asymmetry_idx_frequency REAL")
    if "asymmetry_idx_stiffness" not in cols:
        conn.execute("ALTER TABLE readings ADD COLUMN asymmetry_idx_stiffness REAL")
    if "flagged" not in cols:
        conn.execute("ALTER TABLE readings ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()


def get_connection():
    _debug("get_connection: opening database")
    conn = sqlite3.connect(DB_NAME, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    _ensure_schema(conn)
    return conn


def init_indexes():
    """Ensure the unique index needed for ON CONFLICT upserts exists.
    Call this once at startup (e.g. in esp32_receiver.py)."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_readings_unique
            ON readings(session_id, taken_at, landmark, side)
        """)
        conn.commit()
    finally:
        conn.close()


def patient_exists(patient_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT 1 FROM patients WHERE patient_id = ?",
        (patient_id,)
    )

    exists = cur.fetchone() is not None
    conn.close()
    return exists


def ensure_patient(patient_id):
    conn = get_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT OR IGNORE INTO patients(patient_id, name)
                VALUES (?, ?)
            """, (patient_id, patient_id))
        except sqlite3.OperationalError:
            cur.execute("""
                INSERT OR IGNORE INTO patients(patient_id)
                VALUES (?)
            """, (patient_id,))
        conn.commit()
    finally:
        conn.close()


def next_session_number(patient_id):
    """Returns the next session number for this patient (count of existing
    sessions + 1). Used to auto-generate a session_id at start_session()."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions WHERE patient_id = ?", (patient_id,))
        return cur.fetchone()[0] + 1
    finally:
        conn.close()


def get_latest_session_id(patient_id=None):
    """
    Returns the most recently started session_id, or None if there are no
    matching sessions. Scoped to patient_id if given, otherwise the most
    recent session across the whole database (any patient) -- which will
    pick up test/demo sessions if they happen to have a more recent
    started_at than your real data, so pass patient_id to avoid that.
    """
    conn = get_connection()
    try:
        if patient_id:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE patient_id = ? ORDER BY started_at DESC LIMIT 1",
                (patient_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def save_session(payload):
    """
    Insert or update a session row. ended_at, administered_by, and
    photo_path are all optional -- this supports the two-phase lifecycle:
    start_session() creates the row early (photo_path set, ended_at still
    None), and the later finalize call (once readings arrive) fills in
    the rest. COALESCE means a call that omits one of these fields won't
    blow away a value set by an earlier call.
    """
    _debug(f"save_session: {payload.get('session_id')} for patient {payload.get('patient_id')}")
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO sessions(
                session_id,
                patient_id,
                session_date,
                started_at,
                ended_at,
                administered_by,
                photo_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                patient_id      = excluded.patient_id,
                session_date    = excluded.session_date,
                started_at      = excluded.started_at,
                ended_at        = COALESCE(excluded.ended_at, sessions.ended_at),
                administered_by = COALESCE(excluded.administered_by, sessions.administered_by),
                photo_path      = COALESCE(excluded.photo_path, sessions.photo_path)
        """, (
            payload["session_id"],
            payload["patient_id"],
            payload["started_at"][:10],
            payload["started_at"],
            payload.get("ended_at"),
            payload.get("administered_by"),
            payload.get("photo_path")
        ))

        conn.commit()
    finally:
        conn.close()


def start_session_record(session_id, patient_id, administered_by, photo_path, started_at):
    """Creates the session row at the moment the 'start session' button is
    pressed -- before any readings exist. ended_at stays None until the
    session is later finalized via process_session()."""
    save_session({
        "session_id": session_id,
        "patient_id": patient_id,
        "started_at": started_at,
        "ended_at": None,
        "administered_by": administered_by,
        "photo_path": photo_path,
    })


def save_readings(payload):
    _debug(f"save_readings: writing {len(payload.get('readings', []))} readings for {payload.get('session_id')}")
    conn = get_connection()
    try:
        cur = conn.cursor()

        for r in payload["readings"]:
            cur.execute("""
                INSERT INTO readings(
                    session_id,
                    taken_at,
                    landmark,
                    side,
                    frequency,
                    stiffness,
                    acceleration,
                    asymmetry_idx_frequency,
                    asymmetry_idx_stiffness,
                    flagged
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, taken_at, landmark, side) DO UPDATE SET
                    frequency = excluded.frequency,
                    stiffness = excluded.stiffness,
                    acceleration = excluded.acceleration
            """, (
                payload["session_id"],
                r["taken_at"],
                r["landmark"],
                r["side"],
                r["frequency"],
                r["stiffness"],
                r.get("acceleration"),
                None,
                None,
                0
            ))

        conn.commit()
    finally:
        conn.close()
import os
import sys
from datetime import datetime

# camera.py lives at the project root, one level up from this Database
# folder -- add it to sys.path so it's importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.db import (
    save_session, save_readings, ensure_patient, patient_exists,
    next_session_number, start_session_record,
)
from Database.compute import compute_session
from Database.heatmap import generate_heatmap, generate_delta_heatmap
from Database.report import generate_and_save_report
from camera import capture_frame, capture_back_image


def start_session(patient_id, administered_by, session_id=None,
                   camera_index=0, auto_register=False):
    """
    Call this the moment the "start session" button is pressed -- before
    any probe alignment or readings happen. Captures the reference back
    photo immediately and creates the session row so later steps
    (compute_session -> heatmap -> report) can find it.

    Returns (session_id, photo_path).
    """
    if auto_register:
        ensure_patient(patient_id)
    elif not patient_exists(patient_id):
        raise ValueError(f"Patient {patient_id} does not exist")

    if session_id is None:
        session_id = f"{patient_id}_S{next_session_number(patient_id):03d}"

    frame = capture_frame(camera_index)
    photo_path = capture_back_image(frame, session_id)

    started_at = datetime.now().isoformat(timespec="seconds")
    start_session_record(session_id, patient_id, administered_by, photo_path, started_at)

    return session_id, photo_path


def process_session(payload, auto_register=False):
    """
    Call this once the full session payload (readings etc.) has arrived,
    e.g. via the ESP32 HTTP endpoint. The session row -- including
    photo_path -- already exists from start_session(); this fills in the
    rest and runs computation, heatmap generation, and reporting.
    """
    if auto_register:
        ensure_patient(payload["patient_id"])
    elif not patient_exists(payload["patient_id"]):
        raise ValueError(f"Patient {payload['patient_id']} does not exist")

    save_session(payload)
    save_readings(payload)
    compute_session(payload["session_id"])

    heatmap_path = generate_heatmap(payload["session_id"])
    delta_path = generate_delta_heatmap(payload["session_id"])
    generate_and_save_report(payload["session_id"], heatmap_path=heatmap_path, delta_path=delta_path)

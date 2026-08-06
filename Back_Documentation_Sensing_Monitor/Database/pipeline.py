import os
import sys
from datetime import datetime
from pathlib import Path

DEBUG = True


def _debug(message):
    if DEBUG:
        print(f"[DEBUG] {message}")

# camera.py lives at the project root, one level up from this Database
# folder -- add it to sys.path so it's importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.db import (
    save_session, save_readings, ensure_patient, patient_exists,
    next_session_number, start_session_record, get_latest_session_id,
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
    _debug(f"start_session: patient={patient_id}, session_id={session_id}, auto_register={auto_register}")

    if auto_register:
        _debug("start_session: ensuring patient")
        ensure_patient(patient_id)
    elif not patient_exists(patient_id):
        _debug("start_session: patient not found")
        raise ValueError(f"Patient {patient_id} does not exist")

    if session_id is None:
        session_id = f"{patient_id}_S{next_session_number(patient_id):03d}"
        _debug(f"start_session: generated session_id={session_id}")

    _debug("start_session: capturing frame")
    frame = capture_frame(camera_index)
    _debug("start_session: capturing back image")
    photo_path = capture_back_image(frame, session_id)

    started_at = datetime.now().isoformat(timespec="seconds")
    _debug(f"start_session: writing session record at {started_at}")
    start_session_record(session_id, patient_id, administered_by, photo_path, started_at)

    _debug(f"start_session: completed for {session_id}")
    return session_id, photo_path


def process_session(payload, auto_register=False):
    """
    Call this once the full session payload (readings etc.) has arrived,
    e.g. via the ESP32 HTTP endpoint. The session row -- including
    photo_path -- already exists from start_session(); this fills in the
    rest and runs computation, heatmap generation, and reporting.
    """
    _debug(f"process_session: start for session={payload.get('session_id')} patient={payload.get('patient_id')}")

    if auto_register:
        _debug("process_session: ensuring patient")
        ensure_patient(payload["patient_id"])
    elif not patient_exists(payload["patient_id"]):
        _debug("process_session: patient not found")
        raise ValueError(f"Patient {payload['patient_id']} does not exist")

    _debug("process_session: saving session")
    save_session(payload)
    _debug("process_session: saving readings")
    save_readings(payload)
    _debug("process_session: computing session")
    compute_session(payload["session_id"])

    _debug("process_session: generating heatmap")
    heatmap_path = generate_heatmap(payload["session_id"])
    _debug("process_session: generating delta heatmap")
    delta_path = generate_delta_heatmap(payload["session_id"])
    _debug("process_session: generating report")
    generate_and_save_report(payload["session_id"], heatmap_path=heatmap_path, delta_path=delta_path)
    _debug("process_session: completed")


if __name__ == "__main__":
    # No more fake demo data -- this regenerates the heatmap, delta
    # heatmap, and report for whichever session is most recent, using
    # real data that's already there. Delta heatmap comparison
    # (S00n vs S00n-1) is handled automatically by generate_delta_heatmap(),
    # which looks up the same patient's most recent PRIOR session by
    # started_at.
    #
    # By default "most recent" is across the WHOLE database (any patient)
    # -- pass a patient_id to scope it to just that patient instead, e.g.
    # if old test/demo sessions have a more recent timestamp than your
    # real data and keep getting picked up unintentionally:
    #     python pipeline.py P001
    _debug("pipeline.py executed directly")

    patient_filter = sys.argv[1] if len(sys.argv) > 1 else None
    session_id = get_latest_session_id(patient_filter)

    if session_id is None:
        scope = f"for patient {patient_filter!r}" if patient_filter else "at all"
        _debug(f"No sessions found {scope} -- nothing to generate.")
    else:
        _debug(f"Regenerating heatmap/delta/report for latest session: {session_id}")
        try:
            compute_session(session_id)
            heatmap_path = generate_heatmap(session_id)
            delta_path = generate_delta_heatmap(session_id)
            generate_and_save_report(session_id, heatmap_path=heatmap_path, delta_path=delta_path)
            _debug(f"pipeline.py finished successfully for {session_id}")
        except Exception as exc:
            _debug(f"pipeline.py failed: {exc}")
            import traceback
            traceback.print_exc()
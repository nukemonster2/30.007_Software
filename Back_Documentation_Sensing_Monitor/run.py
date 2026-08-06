"""
Sensing Monitor -- unified CLI launcher.

Run this from the project root (the folder containing Database/, new_Aruco/,
camera.py, pose_landmarker.task, and 30.007.db):

    python run.py

Presents a menu:
    0. Start probe targeting (ArUco probe -> muscle group guidance)
    1. Start ESP32 receiver server
    2. Run pipeline for a patient (prompts for patient name/ID)
    q. Quit
"""

import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _script_path(*parts):
    path = PROJECT_ROOT.joinpath(*parts)
    if not path.exists():
        print(f"WARNING: expected file not found: {path}")
    return path


def run_targeting():
    """
    Launches new_Aruco/targeting.py as its own process -- live probe (X)
    vs muscle-group target (O) guidance. Runs until 'q' is pressed in the
    camera window, then returns control to this menu.
    """
    try:
        script = _script_path("new_Aruco", "targeting.py")
        model = _script_path("pose_landmarker.task")
        print(f"Starting probe targeting...")
        print(f"  script: {script}")
        print(f"  model:  {model}")
        print("(Camera window will open. Press 'q' in it to return to this menu.)")
        subprocess.run([sys.executable, str(script), "--model", str(model)])
    except Exception as exc:
        print(f"Failed to start probe targeting: {exc}")
        traceback.print_exc()


def run_esp32_receiver():
    """
    Launches Database/esp32_receiver.py as its own process -- the Flask
    server that accepts session payloads from the measurement device.
    Runs until interrupted (Ctrl+C), then returns control to this menu.
    """
    script = _script_path("Database", "esp32_receiver.py")
    print(f"Starting ESP32 receiver...")
    print(f"  script: {script}")
    print("(Server will run until you press Ctrl+C. That returns you to this menu.)")
    try:
        subprocess.run([sys.executable, str(script)])
    except KeyboardInterrupt:
        print("\nESP32 receiver stopped.")
    except Exception as exc:
        print(f"Failed to start ESP32 receiver: {exc}")
        traceback.print_exc()


def run_pipeline_for_patient():
    """
    Prompts for a patient name/ID, finds that patient's most recent
    session, recomputes its asymmetry indices, and regenerates the
    heatmap, delta heatmap, and PDF report for it.
    """
    try:
        from Database.db import get_latest_session_id
        from Database.compute import compute_session
        from Database.heatmap import generate_heatmap, generate_delta_heatmap
        from Database.report import generate_and_save_report

        patient_id = input("Patient name/ID: ").strip()
        if not patient_id:
            print("No patient ID entered -- cancelled.")
            return

        session_id = get_latest_session_id(patient_id)
        if session_id is None:
            print(f"No sessions found for patient {patient_id!r}.")
            return

        print(f"Latest session for {patient_id!r}: {session_id}")
        print("Recomputing asymmetry indices and regenerating outputs...")

        compute_session(session_id)
        heatmap_path = generate_heatmap(session_id)
        delta_path = generate_delta_heatmap(session_id)
        pdf_path = generate_and_save_report(session_id, heatmap_path=heatmap_path, delta_path=delta_path)

        print(f"Done.")
        print(f"  heatmap: {heatmap_path}")
        print(f"  delta:   {delta_path}")
        print(f"  report:  {pdf_path}")
    except Exception as exc:
        print(f"Failed: {exc}")
        traceback.print_exc()
        print("(Returning to menu -- nothing else was affected.)")


MENU = """
=== Sensing Monitor ===
0. Start probe targeting (ArUco probe -> muscle group)
1. Start ESP32 receiver server
2. Run pipeline for a patient (prompts for patient name)
q. Quit
"""


def main():
    actions = {
        "0": run_targeting,
        "1": run_esp32_receiver,
        "2": run_pipeline_for_patient,
    }

    while True:
        print(MENU)
        choice = input("Select an option: ").strip().lower()

        if choice == "q":
            print("Goodbye.")
            break

        action = actions.get(choice)
        if action is None:
            print(f"Invalid choice: {choice!r}. Enter 0, 1, 2, or q.")
            continue

        action()


if __name__ == "__main__":
    main()
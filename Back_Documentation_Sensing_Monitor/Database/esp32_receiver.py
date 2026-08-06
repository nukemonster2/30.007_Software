import logging
import os

from flask import Flask, jsonify, request
from pathlib import Path

from pipeline import process_session
from db import init_indexes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimal shared-secret check. This is NOT production-grade auth (no
# rotation, no per-device keys, sent in a plain header) but it stops the
# endpoint being wide open to anyone who can reach it. Set this env var
# on both the server and the ESP32 firmware.
API_KEY = os.environ.get("ESP32_API_KEY")

init_indexes()
app = Flask(__name__)

SAVE_FILE = Path(__file__).resolve().parent / "received_sessions.jsonl"


def validate_payload(payload):
    required_top_level = [
        "session_id",
        "patient_id",
        "started_at",
        "ended_at",
        "administered_by",
        "readings",
    ]

    for field in required_top_level:
        if field not in payload:
            return False, f"Missing field: {field}"

    if not isinstance(payload["readings"], list) or not payload["readings"]:
        return False, "'readings' must be a non-empty list"

    for i, reading in enumerate(payload["readings"]):
        required_reading_fields = [
            "landmark",
            "side",
            "taken_at",
            "frequency",
            "stiffness",
        ]

        for field in required_reading_fields:
            if field not in reading:
                return False, f"Reading {i} missing field: {field}"

    return True, "OK"


@app.route("/esp32/session", methods=["POST"])
def receive_session():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if not request.is_json:
        return jsonify({"status": "error", "message": "Request body must be JSON"}), 400

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "JSON body must be an object"}), 400

    is_valid, message = validate_payload(payload)
    if not is_valid:
        return jsonify({"status": "error", "message": message}), 400

    # Log metadata only — NOT the raw payload, which contains patient data.
    logger.info(
        "Received session %s for patient %s (%d readings)",
        payload.get("session_id"), payload.get("patient_id"), len(payload["readings"]),
    )

    try:
        process_session(payload)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    return jsonify({
        "status": "success",
        "message": "Session payload received and stored",
        "session_id": payload["session_id"],
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

import os
import sys
import math
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

import Database.targeting_config as cfg

# Ensure repo root (parent of this Database folder) is on sys.path so
# sibling package `new_Aruco` can be imported when running this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_Aruco.targeting import get_pose_points, compute_targets
from Database.db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
SESSION_IMAGES_DIR = PROJECT_ROOT / "session_images"


def _resolve_output_dir(output_dir):
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


FLAG_THRESHOLD = 15.0
SEVERITY_CAP = 40.0  # AI% at which the "current" color maxes out fully red


def _severity_color(ai_value):
    """
    Map an AI percentage to a BGR color on a green -> yellow -> red scale.
    None / missing AI -> grey (no data).
    """
    if ai_value is None:
        return (160, 160, 160)  # grey

    t = max(0.0, min(1.0, ai_value / SEVERITY_CAP))
    if t < 0.5:
        # green -> yellow
        local_t = t / 0.5
        b, g, r = 0, 200, int(200 * local_t)
    else:
        # yellow -> red
        local_t = (t - 0.5) / 0.5
        b, g, r = 0, int(200 * (1 - local_t)), 200
    return (b, g, r)


def _delta_color(current, previous):
    """
    BGR color for the change in AI between sessions.
    Red = got worse (higher AI now), blue = improved (lower AI now),
    grey = no data for one side of the comparison.
    """
    if current is None or previous is None:
        return (160, 160, 160)  # grey - no comparison possible

    delta = current - previous
    cap = 20.0  # +/- AI points at which the color fully saturates
    t = max(-1.0, min(1.0, delta / cap))

    if t >= 0:
        # worse -> red
        return (0, int(200 * (1 - t)), 200)
    else:
        # better -> blue
        t = -t
        return (200, int(200 * (1 - t)), 0)


def _auto_marker_radii(targets, max_radius=22, min_radius=4, gap_px=2):
    """
    Per-point radius: half of that point's distance to its nearest
    neighbor (minus a small gap so circles never touch), capped at
    max_radius.

    A single fixed radius overlaps badly in this photo because the lumbar
    cluster (erector_spinae_lumbar, iliocostalis_lumborum,
    thoracolumbar_fascia, multifidus) sits much closer together than
    latissimus_dorsi_lower or gluteus_medius -- e.g. erector_spinae_lumbar
    and iliocostalis_lumborum can be as little as ~8px apart at typical
    photo scale, versus 22px+ for the well-separated points. Scaling per
    point (rather than shrinking everything to fit the tightest pair)
    means crowded regions shrink on their own while spaced-out points
    still get the full max_radius.

    targets: {(muscle, side): (x, y)}
    Returns: {(muscle, side): radius_px}
    """
    keys = list(targets.keys())
    positions = list(targets.values())
    n = len(keys)
    radii = {}
    for i in range(n):
        nearest = min(
            (math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
             for j in range(n) if j != i),
            default=max_radius * 2,
        )
        radius = max(min_radius, min(max_radius, nearest / 2 - gap_px))
        radii[keys[i]] = radius
    return radii


def _fetch_landmark_ai(session_id):
    """Returns {landmark: {'ai_freq':.., 'ai_stiff':.., 'flagged':..}}."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT landmark, side, asymmetry_idx_frequency, asymmetry_idx_stiffness, flagged
            FROM readings
            WHERE session_id = ?
        """, (session_id,))
        rows = cur.fetchall()
    finally:
        conn.close()

    result = {}
    for landmark, side, ai_freq, ai_stiff, flagged in rows:
        # readings table has one row per (landmark, side); both L/R rows
        # carry the same computed AI values in principle, but if one side
        # hasn't been measured yet this session its row will be NULL --
        # don't let a later NULL row silently overwrite an earlier real
        # value for the same landmark.
        existing = result.get(landmark)
        if existing is not None and (existing["ai_freq"] is not None or existing["ai_stiff"] is not None):
            if ai_freq is None and ai_stiff is None:
                continue  # keep the earlier non-null entry
        result[landmark] = {
            "ai_freq": ai_freq,
            "ai_stiff": ai_stiff,
            "flagged": bool(flagged),
        }
    return result


def _worst_ai(info):
    """Picks the worse of frequency/stiffness AI from a _fetch_landmark_ai entry."""
    if info is None:
        return None
    candidates = [v for v in (info["ai_freq"], info["ai_stiff"]) if v is not None]
    return max(candidates) if candidates else None


def _fetch_previous_session_id(session_id):
    """Returns the patient's most recent prior session_id, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT patient_id, started_at FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row is None:
            return None
        patient_id, started_at = row

        prev = conn.execute("""
            SELECT session_id FROM sessions
            WHERE patient_id = ? AND session_id != ? AND started_at < ?
            ORDER BY started_at DESC LIMIT 1
        """, (patient_id, session_id, started_at)).fetchone()
        return prev[0] if prev else None
    finally:
        conn.close()


_pose_landmarker = None


def _resolve_model_path(model_path):
    if not model_path:
        return None

    candidate = Path(model_path)
    if candidate.is_absolute():
        return candidate

    for base in (PROJECT_DIR, PROJECT_ROOT, Path.cwd()):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved

    return (PROJECT_DIR / candidate).resolve()


def _get_pose_landmarker(model_path):
    global _pose_landmarker
    if _pose_landmarker is None:
        resolved_model_path = _resolve_model_path(model_path)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        _pose_landmarker = PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def _resolve_photo_path(session_id, provided_path=None):
    candidates = []
    if provided_path:
        candidates.append(Path(provided_path))
    if session_id:
        candidates.append(SESSION_IMAGES_DIR / f"session_{session_id}_back.png")
        candidates.append(SESSION_IMAGES_DIR / f"{session_id}.png")
        candidates.append(SESSION_IMAGES_DIR / f"{session_id}.jpg")
    candidates.append(SESSION_IMAGES_DIR / "preview.png")
    candidates.append(PROJECT_ROOT / "30.007" / "session_images" / "preview.png")

    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.exists():
            return str(candidate)
    return None


def _load_photo_and_targets(session_id, model_path, mirror):
    """
    Shared setup for both heatmap functions: loads the session's photo,
    runs pose detection, and computes muscle-group pixel positions.
    Returns (frame, targets) or (None, None) if anything is missing.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT photo_path FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    finally:
        conn.close()

    photo_path = _resolve_photo_path(session_id, row[0] if row else None)
    if not photo_path:
        return None, None

    frame = cv2.imread(photo_path)
    if frame is None:
        return None, None

    height, width = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    landmarker = _get_pose_landmarker(model_path)
    pose_result = landmarker.detect(mp_image)
    pose_points = get_pose_points(pose_result, width, height)

    if pose_points is None:
        return None, None  # no person detected in the stored photo

    targets = compute_targets(pose_points, mirror=mirror)
    return frame, targets


def generate_heatmap(
    session_id,
    model_path="pose_landmarker.task",
    output_dir="heatmaps",
    mirror=False,
    marker_radius=22,
):
    """
    Build the tension heatmap PNG for a session and return its path.
    Falls back to a simple placeholder image when there is no usable photo
    or pose data so the report pipeline can still complete.
    """
    output_dir_path = _resolve_output_dir(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = str(output_dir_path / f"{session_id}_heatmap.png")

    frame, targets = _load_photo_and_targets(session_id, model_path, mirror)
    if frame is None or targets is None:
        blank = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "Heatmap unavailable", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        cv2.imwrite(out_path, blank)
        return out_path

    landmark_ai = _fetch_landmark_ai(session_id)
    radii = _auto_marker_radii(targets, max_radius=marker_radius)

    overlay = frame.copy()
    for (muscle, side), (x, y) in targets.items():
        ai_value = _worst_ai(landmark_ai.get(muscle))
        color = _severity_color(ai_value)
        cv2.circle(overlay, (int(x), int(y)), int(round(radii[(muscle, side)])),
                   color, -1, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.imwrite(out_path, blended)
    return out_path


def generate_delta_heatmap(
    session_id,
    model_path="pose_landmarker.task",
    output_dir="heatmaps",
    mirror=False,
    marker_radius=22,
):
    """
    Overlays the CURRENT session's photo with markers colored by the
    change in asymmetry (worse of frequency/stiffness AI) versus the
    patient's previous session. Returns None if there's no previous
    session, or no photo, to compare against.
    """
    output_dir_path = _resolve_output_dir(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = str(output_dir_path / f"{session_id}_delta_heatmap.png")

    prev_session_id = _fetch_previous_session_id(session_id)
    if prev_session_id is None:
        blank = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "Delta unavailable", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        cv2.imwrite(out_path, blank)
        return out_path

    frame, targets = _load_photo_and_targets(session_id, model_path, mirror)
    if frame is None or targets is None:
        blank = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "Delta unavailable", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        cv2.imwrite(out_path, blank)
        return out_path

    current_ai = _fetch_landmark_ai(session_id)
    previous_ai = _fetch_landmark_ai(prev_session_id)
    radii = _auto_marker_radii(targets, max_radius=marker_radius)

    overlay = frame.copy()
    for (muscle, side), (x, y) in targets.items():
        cur_val = _worst_ai(current_ai.get(muscle))
        prev_val = _worst_ai(previous_ai.get(muscle))
        color = _delta_color(cur_val, prev_val)
        cv2.circle(overlay, (int(x), int(y)), int(round(radii[(muscle, side)])),
                   color, -1, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.imwrite(out_path, blended)
    return out_path


import argparse
import math
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

import targeting_config as cfg


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _scale(a, s):
    return (a[0] * s, a[1] * s)


def _norm(a):
    return math.hypot(a[0], a[1])


def _unit(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n) if n else (0.0, 0.0)


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ---------------------------------------------------------------------------
# ArUco probe detection
# ---------------------------------------------------------------------------

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICTIONARY))
_ARUCO_DETECTOR = (
    cv2.aruco.ArucoDetector(_ARUCO_DICT, cv2.aruco.DetectorParameters())
    if hasattr(cv2.aruco, "ArucoDetector") else None
)


def detect_probe(frame_bgr, marker_id):
    """Returns (x, y) pixel position of the probe's marker in this frame, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if _ARUCO_DETECTOR is not None:
        corners, ids, _rejected = _ARUCO_DETECTOR.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _rejected = cv2.aruco.detectMarkers(gray, _ARUCO_DICT, parameters=params)

    if ids is None:
        return None
    for marker_corners, mid in zip(corners, ids.flatten()):
        if int(mid) == marker_id:
            pts = marker_corners.reshape(4, 2)
            center = pts.mean(axis=0)
            return (float(center[0]), float(center[1]))
    return None


# ---------------------------------------------------------------------------
# MediaPipe pose -> muscle-group target positions
# ---------------------------------------------------------------------------

def get_pose_points(pose_result, width, height):
    """Returns pixel-space shoulder/hip points from a PoseLandmarker result, or None."""
    if not pose_result.pose_landmarks:
        return None
    landmarks = pose_result.pose_landmarks[0]

    def px(idx):
        lm = landmarks[idx]
        return (lm.x * width, lm.y * height)

    return {
        "shoulder_l": px(cfg.POSE_LEFT_SHOULDER), "shoulder_r": px(cfg.POSE_RIGHT_SHOULDER),
        "hip_l": px(cfg.POSE_LEFT_HIP), "hip_r": px(cfg.POSE_RIGHT_HIP),
    }


def compute_targets(pose_points, mirror=False):
    """
    Build a spine-aligned frame from shoulder/hip points and place every
    muscle group per targeting_config.MUSCLE_GROUP_RATIOS.

    Returns {(muscle_group, side): (x, y)}.
    """
    shoulder_l, shoulder_r = pose_points["shoulder_l"], pose_points["shoulder_r"]
    hip_l, hip_r = pose_points["hip_l"], pose_points["hip_r"]

    shoulder_mid = _midpoint(shoulder_l, shoulder_r)
    hip_mid = _midpoint(hip_l, hip_r)
    shoulder_width = _norm(_sub(shoulder_r, shoulder_l))
    hip_width = _norm(_sub(hip_r, hip_l))

    spine_vec = _sub(hip_mid, shoulder_mid)
    spine_unit = _unit(spine_vec)

    # "Rightward" reference direction: perpendicular component of the
    # shoulder line relative to the spine (robust to a slightly tilted pose).
    lateral_ref = _sub(shoulder_r, shoulder_l)
    proj_len = lateral_ref[0] * spine_unit[0] + lateral_ref[1] * spine_unit[1]
    lateral_unit = _unit(_sub(lateral_ref, _scale(spine_unit, proj_len)))
    if mirror:
        lateral_unit = _scale(lateral_unit, -1)

    targets = {}
    for muscle, (t, lateral_frac, width_ref) in cfg.MUSCLE_GROUP_RATIOS.items():
        base = _add(shoulder_mid, _scale(spine_vec, t))
        width = shoulder_width if width_ref == "shoulder" else hip_width
        for side, sign in (("R", 1.0), ("L", -1.0)):
            offset = _scale(lateral_unit, sign * lateral_frac * width)
            targets[(muscle, side)] = _add(base, offset)

    return targets


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_x(frame, center, size=16, color=cfg.COLOR_PROBE, thickness=3):
    x, y = int(center[0]), int(center[1])
    cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)


def draw_o(frame, center, radius=cfg.CAPTURE_RADIUS_PX, color=cfg.COLOR_TARGET, thickness=3):
    cv2.circle(frame, (int(center[0]), int(center[1])), radius, color, thickness, cv2.LINE_AA)


def draw_dwell_progress(frame, center, radius, progress, color=cfg.COLOR_DWELLING):
    cv2.circle(frame, (int(center[0]), int(center[1])), radius, color, 2, cv2.LINE_AA)
    angle = int(360 * min(progress, 1.0))
    cv2.ellipse(frame, (int(center[0]), int(center[1])), (radius + 4, radius + 4),
                -90, 0, angle, color, 4, cv2.LINE_AA)


def get_capture_radius(muscle_group, base_radius):
    """Scales the base capture radius per muscle group per
    targeting_config.MUSCLE_GROUP_RADIUS_SCALE — bigger muscles get a
    larger margin of error, smaller/deeper ones get a tighter one."""
    scale = cfg.MUSCLE_GROUP_RADIUS_SCALE.get(muscle_group, 1.0)
    return base_radius * scale


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(model_path, camera_index, probe_marker_id,
        capture_radius=cfg.CAPTURE_RADIUS_PX, dwell_seconds=cfg.DWELL_SECONDS, mirror=False):
    target_keys = [(muscle, side) for muscle in cfg.MUSCLE_GROUP_RATIOS for side in cfg.SIDES]
    confirmed = set()
    dwell_start = {}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    print(f"Fully automatic — no calibration step. {len(target_keys)} targets, "
          f"recomputed live every frame.")
    print(f"Hold the probe steady inside a target for {dwell_seconds:.1f}s to confirm it.")
    print("Keys: [ / ] = prev/next probe marker ID, r = reset, m = toggle mirror, q = quit")
    print(f"Looking for probe marker ID {probe_marker_id}.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            pose_result = landmarker.detect(mp_image)
            pose_points = get_pose_points(pose_result, width, height)

            probe_pos = detect_probe(frame, probe_marker_id)
            now = time.time()

            if pose_points is None:
                cv2.putText(frame, "No person detected", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2, cv2.LINE_AA)
                dwell_start.clear()
            else:
                targets = compute_targets(pose_points, mirror=mirror)
                target_radius = {key: get_capture_radius(key[0], capture_radius) for key in target_keys}

                # Only the SINGLE nearest unconfirmed target is allowed to
                # dwell at a time -- otherwise, when targets sit closer
                # together than 2x their capture radius, one held position
                # could fall inside several targets' radius at once and
                # they'd all confirm together. "Nearest" here means closest
                # relative to each target's own (possibly different) radius,
                # so a big target doesn't unfairly out-compete a small one
                # just for being more forgiving.
                nearest_key = None
                if probe_pos is not None:
                    remaining = [k for k in target_keys if k not in confirmed]
                    in_range = [
                        (k, _norm(_sub(probe_pos, targets[k])) / target_radius[k])
                        for k in remaining
                    ]
                    in_range = [(k, ratio) for k, ratio in in_range if ratio <= 1.0]
                    if in_range:
                        nearest_key = min(in_range, key=lambda kr: kr[1])[0]

                for key in target_keys:
                    if key in confirmed:
                        continue
                    pos = targets[key]
                    radius = target_radius[key]
                    if key == nearest_key:
                        if key not in dwell_start:
                            dwell_start[key] = now
                        elapsed = now - dwell_start[key]
                        if elapsed >= dwell_seconds:
                            confirmed.add(key)
                            dwell_start.pop(key, None)
                            print(f"Confirmed: {key[0]} ({key[1]})  ({len(confirmed)}/{len(target_keys)})")
                            draw_o(frame, pos, radius=radius, color=cfg.COLOR_CONFIRMED)
                        else:
                            draw_dwell_progress(frame, pos, radius, elapsed / dwell_seconds)
                    else:
                        dwell_start.pop(key, None)
                        draw_o(frame, pos, radius=radius, color=cfg.COLOR_TARGET)

                probe_status = f"probe ID {probe_marker_id} tracked" if probe_pos is not None \
                    else f"probe marker (ID {probe_marker_id}) not detected"
                cv2.putText(frame, probe_status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

            if probe_pos is not None:
                draw_x(frame, probe_pos)

            done_label = f"Confirmed: {len(confirmed)}/{len(target_keys)}"
            cv2.putText(frame, done_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        cfg.COLOR_CONFIRMED if len(confirmed) == len(target_keys) else (255, 255, 255),
                        2, cv2.LINE_AA)
            if len(confirmed) == len(target_keys):
                cv2.putText(frame, "ALL TARGETS CONFIRMED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, cfg.COLOR_CONFIRMED, 2, cv2.LINE_AA)

            cv2.imshow("Probe targeting (X=probe, O=remaining targets)", frame)
            key_pressed = cv2.waitKey(1) & 0xFF

            if key_pressed == ord("q"):
                break
            elif key_pressed == ord("r"):
                confirmed.clear()
                dwell_start.clear()
                print("Reset: all targets restored.")
            elif key_pressed == ord("m"):
                mirror = not mirror
                print(f"Mirror -> {mirror}")
            elif key_pressed == ord("["):
                probe_marker_id = max(0, probe_marker_id - 1)
                print(f"Probe marker ID -> {probe_marker_id}")
            elif key_pressed == ord("]"):
                probe_marker_id += 1
                print(f"Probe marker ID -> {probe_marker_id}")

    cap.release()
    cv2.destroyAllWindows()
    return confirmed, target_keys


def main():
    parser = argparse.ArgumentParser(description="Live probe-to-muscle-group targeting")
    parser.add_argument("--model", default="pose_landmarker.task",
                         help="Path to your MediaPipe Pose Landmarker .task file")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--probe-marker-id", type=int, default=cfg.PROBE_MARKER_ID)
    parser.add_argument("--capture-radius", type=int, default=cfg.CAPTURE_RADIUS_PX)
    parser.add_argument("--dwell-seconds", type=float, default=cfg.DWELL_SECONDS)
    parser.add_argument("--mirror", action="store_true",
                         help="Swap L/R if the camera faces the patient instead of being behind them")
    args = parser.parse_args()

    confirmed, targets = run(args.model, args.camera, args.probe_marker_id,
                              capture_radius=args.capture_radius,
                              dwell_seconds=args.dwell_seconds, mirror=args.mirror)
    print(f"Session complete: {len(confirmed)}/{len(targets)} confirmed.")
    missing = [k for k in targets if k not in confirmed]
    if missing:
        print(f"Not confirmed: {missing}")


if __name__ == "__main__":
    main()
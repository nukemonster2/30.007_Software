"""
Run from your project's 30.007/ folder:
    python diagnose_delta2.py SESSION_ID

Pinpoints exactly where photo/pose loading is failing for a session --
photo resolution, image loading, or MediaPipe pose detection -- and
checks whether the regular (non-delta) heatmap has the same problem.
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from db import get_connection
from heatmap import (
    _resolve_photo_path, _get_pose_landmarker, generate_heatmap,
    generate_delta_heatmap, PROJECT_ROOT, SESSION_IMAGES_DIR,
)
from new_Aruco.targeting import get_pose_points

if len(sys.argv) < 2:
    sys.exit("Usage: python diagnose_delta2.py SESSION_ID")

session_id = sys.argv[1]

print(f"PROJECT_ROOT resolves to: {PROJECT_ROOT}")
print(f"SESSION_IMAGES_DIR resolves to: {SESSION_IMAGES_DIR}")
print()

conn = get_connection()
row = conn.execute("SELECT photo_path FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
conn.close()
stored_photo_path = row[0] if row else None
print(f"photo_path stored in DB for {session_id}: {stored_photo_path!r}")

resolved = _resolve_photo_path(session_id, stored_photo_path)
print(f"_resolve_photo_path() found: {resolved!r}")

if resolved is None:
    print()
    print(">>> PROBLEM FOUND: no photo file could be located at all.")
    print(">>> Check that the file actually exists at one of these paths:")
    print(f"      {stored_photo_path}")
    print(f"      {SESSION_IMAGES_DIR / f'session_{session_id}_back.png'}")
    print(f"      {SESSION_IMAGES_DIR / 'preview.png'}")
    sys.exit(1)

frame = cv2.imread(resolved)
if frame is None:
    print()
    print(">>> PROBLEM FOUND: the file exists at that path but OpenCV couldn't read it")
    print(">>> (corrupted file, unsupported format, or a permissions issue).")
    sys.exit(1)

print(f"Image loaded OK, shape: {frame.shape}")
print()

print("Running MediaPipe pose detection...")
height, width = frame.shape[:2]
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
landmarker = _get_pose_landmarker("pose_landmarker.task")
pose_result = landmarker.detect(mp_image)
pose_points = get_pose_points(pose_result, width, height)

if pose_points is None:
    print()
    print(">>> PROBLEM FOUND: MediaPipe did not detect a person in this photo.")
    print(">>> This is why both generate_heatmap() and generate_delta_heatmap()")
    print(">>> fall back to the 'unavailable' placeholder -- it's not a delta-")
    print(">>> specific issue, the current session's photo itself isn't usable.")
    sys.exit(1)

print(f"Pose detected OK: {pose_points}")
print()
print("Photo + pose both work -- generating both heatmaps now to compare directly...")

hpath = generate_heatmap(session_id)
himg = cv2.imread(hpath)
print(f"generate_heatmap() -> {hpath}, brightness={himg.mean():.1f}")

dpath = generate_delta_heatmap(session_id)
dimg = cv2.imread(dpath)
print(f"generate_delta_heatmap() -> {dpath}, brightness={dimg.mean():.1f}")

if dimg.mean() < 10:
    print()
    print(">>> Delta heatmap brightness is near-zero -- it's still hitting the")
    print(">>> placeholder path despite pose/photo working. This means")
    print(">>> _fetch_previous_session_id worked (confirmed earlier) AND")
    print(">>> _load_photo_and_targets worked (confirmed above) -- something")
    print(">>> is throwing inside generate_delta_heatmap's drawing loop itself.")
    print(">>> Please share this full output so we can look at that directly.")
else:
    print()
    print(">>> Delta heatmap generated successfully with real content.")
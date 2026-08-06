
# ---------------------------------------------------------------------------
# Muscle groups
# ---------------------------------------------------------------------------

UPPER_BACK_LANDMARKS = [
    "erector_spinae_thoracic",
]

LOWER_BACK_LANDMARKS = [
    "erector_spinae_lumbar",
    "multifidus",
    "quadratus_lumborum",
    "iliocostalis_lumborum",
    "longissimus_thoracis",
    "latissimus_dorsi_lower",
    "thoracolumbar_fascia",
    "gluteus_medius",
]

ALL_LANDMARKS = UPPER_BACK_LANDMARKS + LOWER_BACK_LANDMARKS  # 10 muscle groups
SIDES = ("L", "R")

# ---------------------------------------------------------------------------
# ArUco marker assignment — direct 1:1, one marker per O
# ---------------------------------------------------------------------------
# Each marker maps to exactly one (landmark, side) — i.e. one O — with no
# mirroring or computation involved. Stick the marker exactly where that O
# belongs; calibrate_landmarks.py reads its position directly.
#
# Convention within each pair: lower ID = L, higher ID = R.
ARUCO_DICTIONARY = "DICT_5X5_50"

ARUCO_MARKER_MAP = {
    2: ("erector_spinae_thoracic", "L"),
    3: ("erector_spinae_thoracic", "R"),
    4: ("latissimus_dorsi_lower", "L"),
    5: ("latissimus_dorsi_lower", "R"),
    6: ("longissimus_thoracis", "L"),
    7: ("longissimus_thoracis", "R"),
    8: ("quadratus_lumborum", "L"),
    9: ("quadratus_lumborum", "R"),
    # Not yet covered (add IDs here as they become available):
    #   erector_spinae_lumbar L/R, multifidus L/R,
    #   iliocostalis_lumborum L/R, thoracolumbar_fascia L/R, gluteus_medius L/R
}
# Reverse lookup: (landmark, side) -> marker_id
LANDMARK_TO_MARKER_ID = {v: k for k, v in ARUCO_MARKER_MAP.items()}

# Dropping trapezius freed IDs 0 and 1. Probe takes 0; 1 is spare — assign
# it to another landmark above (e.g. erector_spinae_lumbar L) whenever you
# print more markers or want partial coverage of a 9th group.
ARUCO_PROBE_MARKER_ID = 0

DEFAULT_CALIBRATION_PATH = "landmark_calibration.json"

# ---------------------------------------------------------------------------
# Anatomical ratios for MediaPipe-based calibration
# ---------------------------------------------------------------------------
# Each muscle group's position is expressed relative to a spine-aligned
# coordinate frame built from four MediaPipe Pose landmarks: left/right
# shoulder and left/right hip. See pose_calibration.py for how this frame
# is built from a photo.
#
#   t             : 0.0 = shoulder line, 1.0 = hip line, along the spine
#   lateral_frac  : how far off the spine midline, as a fraction of the
#                   reference width (shoulder or hip width, half-width == 1.0
#                   would be exactly at the shoulder/hip tip)
#   width_ref     : "shoulder" or "hip" — which width lateral_frac scales by
#
# These are starting-point clinical estimates, not measured facts — treat
# them as a first pass. Verify with pose_calibration.py's --preview output,
# and if a point looks off for a given body type, either hand-edit the
# resulting landmark_calibration.json (it's plain pixel coordinates) or
# adjust the ratio here.
POSE_LANDMARK_RATIOS = {
    "erector_spinae_thoracic":  (0.35, 0.15, "shoulder"),
    "latissimus_dorsi_lower":   (0.50, 0.40, "shoulder"),
    "longissimus_thoracis":     (0.55, 0.18, "hip"),
    "quadratus_lumborum":       (0.60, 0.25, "hip"),
    "erector_spinae_lumbar":    (0.65, 0.15, "hip"),
    "iliocostalis_lumborum":    (0.68, 0.22, "hip"),
    "thoracolumbar_fascia":     (0.72, 0.12, "hip"),
    "multifidus":               (0.78, 0.08, "hip"),
    "gluteus_medius":           (0.95, 0.30, "hip"),
}

# MediaPipe Pose landmark indices (subject-relative left/right, not image
# left/right). See pose_calibration.py for the assumption this relies on
# (camera behind the patient, patient facing away from camera).
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_HIP, POSE_RIGHT_HIP = 23, 24
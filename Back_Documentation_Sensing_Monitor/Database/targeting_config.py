SIDES = ("L", "R")

# Each muscle group's position, expressed relative to a hip-only frame
# (see targeting.compute_targets): only the hip landmarks are used now,
# since every target we track is a lower-back / lumbar-region muscle and
# shoulder tracking added noise without adding accuracy for this region.
#   t_above_hip   : distance above the hip line, as a fraction of hip
#                   width (0.0 = right on the hip line, larger = higher
#                   up the back)
#   lateral_frac  : how far off the spine midline, as a fraction of hip
#                   width (1.0 would be exactly at the hip tip)
#
# These are re-derived from the old shoulder-to-hip ratios using an
# approximate torso-height : hip-width ratio of 1.5 (TORSO_HIP_RATIO
# below) -- still starting-point clinical estimates, not measured facts.
# If a point looks consistently off for a given body type, adjust the
# numbers here (or adjust TORSO_HIP_RATIO to rescale all of them at once).
TORSO_HIP_RATIO = 1.5

MUSCLE_GROUP_RATIOS = {
    "latissimus_dorsi_lower":   (0.75, 0.46),
    "longissimus_thoracis":     (0.68, 0.18),
    "quadratus_lumborum":       (0.60, 0.25),
    "erector_spinae_lumbar":    (0.53, 0.15),
    "iliocostalis_lumborum":    (0.48, 0.22),
    "thoracolumbar_fascia":     (0.42, 0.12),
    "multifidus":               (0.33, 0.08),
    "gluteus_medius":           (0.08, 0.30),
}

# ---------------------------------------------------------------------------
# MediaPipe Pose
# ---------------------------------------------------------------------------
# Landmark indices (subject-relative left/right, not image left/right).
# Only the hips are used now (see compute_targets). ASSUMPTION: camera is
# behind the patient, facing the same direction they are, so image-left =
# the patient's own left. Pass --mirror if your setup is reversed (e.g.
# camera facing the patient).
POSE_LEFT_HIP, POSE_RIGHT_HIP = 23, 24

# ---------------------------------------------------------------------------
# ArUco (probe only — muscle groups have no physical markers)
# ---------------------------------------------------------------------------
ARUCO_DICTIONARY = "DICT_5X5_50"
PROBE_MARKER_ID = 0

# ---------------------------------------------------------------------------
# Muscle location IDs (for the measurement device / marker workflow)
# ---------------------------------------------------------------------------
# The measurement probe carries an ArUco marker (PROBE_MARKER_ID above).
# Separately, each of the 16 muscle locations (8 groups x L/R) has its own
# numeric ID, 1-16, matching 1:1 to an ArUco marker ID of the same number
# via MUSCLE_ID_TO_ARUCO_MARKER_ID -- e.g. muscle_id 1 <-> marker 1 <->
# erector_spinae_lumbar (L).
#
# NOTE: muscle_id 1's marker (1) is the SAME number as PROBE_MARKER_ID.
# This is intentional, not a collision to fix -- confirmed directly. Since
# the two are never relevant at the same moment (PROBE_MARKER_ID identifies
# the probe during live targeting; MUSCLE_ID_TO_ARUCO_MARKER_ID identifies
# a location during the device/marker workflow), sharing the number 1 is
# fine. Don't "fix" this by reassigning one of them without checking back
# on the actual workflow first.
MUSCLE_ID_TO_LOCATION = {
    1:  ("erector_spinae_lumbar", "L"),
    2:  ("erector_spinae_lumbar", "R"),
    3:  ("multifidus", "L"),
    4:  ("multifidus", "R"),
    5:  ("quadratus_lumborum", "L"),
    6:  ("quadratus_lumborum", "R"),
    7:  ("iliocostalis_lumborum", "L"),
    8:  ("iliocostalis_lumborum", "R"),
    9:  ("longissimus_thoracis", "L"),
    10: ("longissimus_thoracis", "R"),
    11: ("latissimus_dorsi_lower", "L"),
    12: ("latissimus_dorsi_lower", "R"),
    13: ("thoracolumbar_fascia", "L"),
    14: ("thoracolumbar_fascia", "R"),
    15: ("gluteus_medius", "L"),
    16: ("gluteus_medius", "R"),
}

# Identity mapping (muscle_id N -> marker ID N), kept as its own dict
# since that's the direct lookup the device/marker workflow needs, and
# it's the exact name the test suite expects.
MUSCLE_ID_TO_ARUCO_MARKER_ID = {muscle_id: muscle_id for muscle_id in MUSCLE_ID_TO_LOCATION}

# Reverse lookup: (muscle_group, side) -> muscle_id
LOCATION_TO_MUSCLE_ID = {v: k for k, v in MUSCLE_ID_TO_LOCATION.items()}

# ---------------------------------------------------------------------------
# Targeting behavior
# ---------------------------------------------------------------------------
CAPTURE_RADIUS_PX = 28   # base radius; scaled per muscle group below
DWELL_SECONDS = 0.5      # how long the probe must stay there, continuously, to confirm

# Bigger/more superficial muscles are easier to land on and don't need
# pinpoint accuracy, so they get a larger capture radius (more forgiving);
# smaller or deeper muscles need the probe closer to the real anatomical
# point, so they get a smaller radius (less forgiving). These are relative
# size judgments, not measurements — adjust if a particular group feels
# too easy or too fussy in practice.
MUSCLE_GROUP_RADIUS_SCALE = {
    "latissimus_dorsi_lower":   1.4,   # large, superficial
    "gluteus_medius":           1.3,   # large
    "longissimus_thoracis":     1.1,   # broad
    "thoracolumbar_fascia":     1.1,   # broad connective sheet
    "erector_spinae_lumbar":    1.0,   # medium (baseline)
    "iliocostalis_lumborum":    1.0,   # medium
    "quadratus_lumborum":       0.8,   # smaller, deep
    "multifidus":               0.7,   # small, narrow, deep
}

# ---------------------------------------------------------------------------
# Colors (BGR, for OpenCV)
# ---------------------------------------------------------------------------
COLOR_PROBE = (255, 120, 0)     # X
COLOR_TARGET = (0, 200, 255)    # O, untouched
COLOR_DWELLING = (0, 165, 255)  # O, probe currently holding inside
COLOR_CONFIRMED = (0, 220, 0)   # O, just confirmed (green flash before it disappears)
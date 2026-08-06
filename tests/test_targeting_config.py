import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "30.007"))

from Database import targeting_config as cfg


def test_muscle_id_1_maps_only_to_marker_1():
    assert cfg.MUSCLE_ID_TO_ARUCO_MARKER_ID[1] == 1
    assert cfg.PROBE_MARKER_ID == 1

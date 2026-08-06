"""
Thin wrapper around cv2.aruco so the rest of the codebase doesn't care
which OpenCV version is installed. OpenCV >=4.7 replaced the old free
functions (cv2.aruco.detectMarkers) with an ArucoDetector class; this
module picks whichever is available.
"""

import cv2
import numpy as np

from new_Aruco.landmark_config import ARUCO_DICTIONARY


def _get_dictionary():
    dict_const = getattr(cv2.aruco, ARUCO_DICTIONARY)
    return cv2.aruco.getPredefinedDictionary(dict_const)


def _get_detector():
    aruco_dict = _get_dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(aruco_dict, params)
    return None  # signals caller to use the legacy function-based API


_DETECTOR = _get_detector()


def detect_markers(frame_bgr):
    """
    Detect ArUco markers in a BGR image/frame.

    Returns: dict of {marker_id (int): (center_x, center_y) in pixels}
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if _DETECTOR is not None:
        corners, ids, _rejected = _DETECTOR.detectMarkers(gray)
    else:
        aruco_dict = _get_dictionary()
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    positions = {}
    if ids is None:
        return positions

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners.reshape(4, 2)
        center = pts.mean(axis=0)
        positions[int(marker_id)] = (float(center[0]), float(center[1]))

    return positions


def draw_marker_outlines(frame_bgr, marker_positions):
    """Debug helper: draw a small dot + ID label at each detected marker."""
    for marker_id, (x, y) in marker_positions.items():
        pt = (int(round(x)), int(round(y)))
        cv2.circle(frame_bgr, pt, 4, (0, 255, 255), -1)
        cv2.putText(frame_bgr, str(marker_id), (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr

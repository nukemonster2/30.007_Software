import cv2
import os
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parent.parent / "session_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def capture_frame(camera_index=0, warmup_frames=5):
    """
    Grabs a single frame from the given camera and releases it.

    Reads and discards a few warmup frames first -- many webcams return a
    dark/uninitialized frame on the very first read after opening.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    frame = None
    try:
        for _ in range(warmup_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from camera")
    finally:
        cap.release()

    return frame


def capture_back_image(frame, session_id):
    """
    Saves the current camera frame as the reference back image.

    Parameters
    ----------
    frame : np.ndarray
        Current OpenCV frame.
    session_id : int
        Current session ID.

    Returns
    -------
    str
        Path to saved image.
    """

    filename = f"session_{session_id}_back.png"
    path = str(IMAGE_DIR / filename)

    cv2.imwrite(path, frame)

    print(f"Saved reference image: {path}")

    return path
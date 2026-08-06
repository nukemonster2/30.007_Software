


import argparse
import sys

import cv2

from aruco_utils import detect_markers, draw_marker_outlines


def run_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera_index}")

    print("Showing live marker detection. Press 'q' to quit.")
    last_ids = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        positions = detect_markers(frame)
        ids = sorted(positions.keys())

        if ids != last_ids:
            print(f"Detected marker IDs: {ids or 'none'}")
            last_ids = ids

        draw_marker_outlines(frame, positions)
        cv2.putText(frame, f"IDs: {ids or 'none'}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("ArUco detector (q = quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"Could not read image: {image_path}")

    positions = detect_markers(frame)

    if not positions:
        print("No markers detected.")
    else:
        print(f"Detected {len(positions)} marker(s):")
        for marker_id, (x, y) in sorted(positions.items()):
            print(f"  ID {marker_id}: ({x:.1f}, {y:.1f})")

    if output_path:
        annotated = draw_marker_outlines(frame.copy(), positions)
        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect ArUco markers in a camera feed or image")
    parser.add_argument("image_path", nargs="?", help="Path to an image. Omit if using --camera.")
    parser.add_argument("--camera", type=int, default=None,
                         help="Camera index for a live feed (e.g. 0). Overrides image_path if given.")
    parser.add_argument("--output", default=None,
                         help="(Image mode only) save an annotated copy here")
    args = parser.parse_args()

    if args.camera is not None:
        run_camera(args.camera)
    elif args.image_path:
        run_image(args.image_path, args.output)
    else:
        sys.exit("Provide an image_path or --camera")


if __name__ == "__main__":
    main()
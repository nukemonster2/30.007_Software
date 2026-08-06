import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Database.heatmap import generate_heatmap


def test_generate_heatmap_uses_available_photo(tmp_path):
    out_dir = tmp_path / "heatmaps"
    out_path = generate_heatmap("demo_session", output_dir=str(out_dir))

    assert Path(out_path).exists(), "heatmap file should be written"

    img = cv2.imread(out_path)
    assert img is not None, "heatmap image should be readable"

    # A real heatmap overlaid on the back photo should be visually non-trivial.
    # The placeholder path creates a nearly black image with only text.
    assert img.mean() > 40, "heatmap should contain real image content, not a placeholder"

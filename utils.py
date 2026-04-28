"""
Utility helpers for drawing and video processing.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from config import BOX_COLOR


def draw_boxes(frame: np.ndarray, result) -> Tuple[np.ndarray, int]:
    """
    Draw green bounding boxes and labels on a frame.
    Returns annotated frame and pothole count.
    """
    count = 0
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, "pothole")

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            # Draw filled label background so confidence text is readable.
            label = f"{class_name} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_top = max(y1 - th - baseline - 8, 0)
            y_bottom = y_top + th + baseline + 8
            x_right = min(x1 + tw + 10, frame.shape[1] - 1)

            cv2.rectangle(frame, (x1, y_top), (x_right, y_bottom), BOX_COLOR, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, y_bottom - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )
            count += 1

    return frame, count


def draw_count(frame: np.ndarray, count: int) -> np.ndarray:
    """
    Draw pothole count on top-left corner.
    """
    cv2.putText(
        frame,
        f"Potholes: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        BOX_COLOR,
        2,
    )
    return frame


def create_video_writer(
    output_path: Path, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    """
    Create OpenCV video writer for mp4 output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if fps <= 0:
        fps = 30.0
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

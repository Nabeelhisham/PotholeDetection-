"""
Run pothole detection on image, video, or webcam.
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    BEST_MODEL_PATH,
    CONF_THRESHOLD,
    OUTPUT_VIDEO_PATH,
    SAMPLE_VIDEO_PATH,
    TRAIN_WEIGHTS_DIR,
)
from utils import create_video_writer, draw_boxes, draw_count


def detect_image(model: YOLO, source: Path, output: Path) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    frame, count = draw_boxes(frame, result)
    frame = draw_count(frame, count)

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    cv2.imshow("Pothole Detection - Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Image result saved: {output}")


def detect_video_stream(model: YOLO, source, output: Path | None = None) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")

    writer = None
    if output is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = create_video_writer(output, width, height, fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        frame, count = draw_boxes(frame, result)
        frame = draw_count(frame, count)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Pothole Detection - Video/Webcam", frame)
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if output is not None:
        print(f"Video result saved: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Pothole Detection")
    parser.add_argument(
        "--mode",
        choices=["image", "video", "webcam"],
        required=True,
        help="Run mode: image, video, or webcam",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(SAMPLE_VIDEO_PATH),
        help="Path to image/video source",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(BEST_MODEL_PATH),
        help="Path to trained model (best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_VIDEO_PATH),
        help="Output file path (image/video)",
    )
    return parser.parse_args()


def resolve_model_path(user_model_path: str) -> str | None:
    """
    Resolve model path robustly:
    1) Try provided path (best.pt)
    2) Fallback to sibling last.pt
    3) Fallback to training weights dir best.pt/last.pt
    """
    requested = Path(user_model_path).expanduser()
    requested_abs = os.path.abspath(str(requested))
    requested_last_abs = os.path.abspath(str(requested.with_name("last.pt")))

    fallback_best = os.path.abspath(str(TRAIN_WEIGHTS_DIR / "best.pt"))
    fallback_last = os.path.abspath(str(TRAIN_WEIGHTS_DIR / "last.pt"))

    candidates = [
        requested_abs,
        requested_last_abs,
        fallback_best,
        fallback_last,
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> None:
    args = parse_args()

    print(f"Current working directory: {os.getcwd()}")
    resolved_model_path = resolve_model_path(args.model)
    print(f"Full model path checked: {os.path.abspath(args.model)}")

    if resolved_model_path is None:
        print("Model not found. Please train the model first.")
        return

    print(f"Using model: {resolved_model_path}")
    model = YOLO(resolved_model_path)

    if args.mode == "image":
        detect_image(model, Path(args.source), Path(args.output))
    elif args.mode == "video":
        detect_video_stream(model, str(Path(args.source)), Path(args.output))
    else:
        # Webcam source 0
        detect_video_stream(model, 0, None)


if __name__ == "__main__":
    main()

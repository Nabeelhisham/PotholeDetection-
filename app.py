"""
Streamlit web app for pothole detection (image and video).
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO
import av

from config import CONF_THRESHOLD, OUTPUTS_DIR, TRAIN_WEIGHTS_DIR
from utils import draw_boxes, draw_count


def resolve_model_path(user_model_path: str) -> str | None:
    """
    Resolve model path robustly:
    1) Requested best.pt
    2) Requested sibling last.pt
    3) Default training best.pt
    4) Default training last.pt
    """
    requested = Path(user_model_path).expanduser()
    requested_abs = os.path.abspath(str(requested))
    requested_last_abs = os.path.abspath(str(requested.with_name("last.pt")))

    fallback_best = os.path.abspath(str(TRAIN_WEIGHTS_DIR / "best.pt"))
    fallback_last = os.path.abspath(str(TRAIN_WEIGHTS_DIR / "last.pt"))

    for candidate in [requested_abs, requested_last_abs, fallback_best, fallback_last]:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def process_image(model: YOLO, image_bytes: bytes, conf_threshold: float) -> tuple[np.ndarray, int]:
    file_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode uploaded image.")

    result = model(frame, conf=conf_threshold, verbose=False)[0]
    frame, count = draw_boxes(frame, result)
    frame = draw_count(frame, count)
    return frame, count


def process_video(model: YOLO, video_bytes: bytes, conf_threshold: float) -> tuple[Path, int, str]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_bytes)
        temp_input_path = Path(tmp_in.name)

    cap = cv2.VideoCapture(str(temp_input_path))
    if not cap.isOpened():
        temp_input_path.unlink(missing_ok=True)
        raise ValueError("Could not open uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Try browser-friendly mp4 codecs first.
    writer = None
    output_path = OUTPUTS_DIR / "streamlit_output.mp4"
    output_mime = "video/mp4"
    for codec in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if candidate_writer.isOpened():
            writer = candidate_writer
            break
        candidate_writer.release()

    # Fallback to AVI if mp4 writers are unavailable on this machine.
    if writer is None:
        output_path = OUTPUTS_DIR / "streamlit_output.avi"
        output_mime = "video/x-msvideo"
        for codec in ["MJPG", "XVID"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if candidate_writer.isOpened():
                writer = candidate_writer
                break
            candidate_writer.release()

    if writer is None:
        cap.release()
        temp_input_path.unlink(missing_ok=True)
        raise ValueError("Could not create output video writer. Try installing FFmpeg-enabled OpenCV.")

    max_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=conf_threshold, verbose=False)[0]
        frame, count = draw_boxes(frame, result)
        frame = draw_count(frame, count)
        max_count = max(max_count, count)
        writer.write(frame)

    cap.release()
    writer.release()
    temp_input_path.unlink(missing_ok=True)
    return output_path, max_count, output_mime


def fit_for_display(frame: np.ndarray, max_width: int = 960, max_height: int = 540) -> np.ndarray:
    """
    Resize frame for clearer on-screen viewing without stretching.
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return frame
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_webrtc_processor(model: YOLO, display_scale: float, conf_threshold: float):
    class PotholeVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            result = model(img_bgr, conf=conf_threshold, verbose=False)[0]
            annotated, count = draw_boxes(img_bgr, result)
            annotated = draw_count(annotated, count)
            # Optional downscale for easier viewing in browser.
            if display_scale < 1.0:
                h, w = annotated.shape[:2]
                annotated = cv2.resize(
                    annotated,
                    (int(w * display_scale), int(h * display_scale)),
                    interpolation=cv2.INTER_AREA,
                )
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    return PotholeVideoProcessor


def main() -> None:
    st.set_page_config(page_title="Pothole Detection", layout="wide")
    st.title("Pothole Detection System (YOLOv8)")
    st.caption("Upload an image or video and get pothole detections with counts.")

    st.write(f"Current working directory: `{os.getcwd()}`")

    model_input = st.text_input(
        "Model path (best.pt preferred)",
        value=str(Path("models") / "best.pt"),
    )
    resolved_model = resolve_model_path(model_input)
    st.write(f"Full model path checked: `{os.path.abspath(model_input)}`")

    if resolved_model is None:
        st.error("Model not found. Please train the model first.")
        st.stop()

    st.success(f"Using model: `{resolved_model}`")
    model = load_model(resolved_model)

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=float(CONF_THRESHOLD),
        step=0.05,
        help="Increase this to reduce false detections and make confidence labels clearer.",
    )

    tab_image, tab_video, tab_webcam = st.tabs(["Image", "Video", "Webcam"])

    with tab_image:
        uploaded_image = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_uploader"
        )
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            output_bgr, count = process_image(model, image_bytes, conf_threshold)
            output_bgr_display = fit_for_display(output_bgr)
            output_rgb = cv2.cvtColor(output_bgr_display, cv2.COLOR_BGR2RGB)
            st.image(output_rgb, caption=f"Detected potholes: {count}")

            success, encoded = cv2.imencode(".jpg", output_bgr)
            if success:
                st.download_button(
                    "Download annotated image",
                    data=encoded.tobytes(),
                    file_name="output_image.jpg",
                    mime="image/jpeg",
                )

    with tab_video:
        uploaded_video = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader"
        )
        if "video_data" not in st.session_state:
            st.session_state.video_data = None
            st.session_state.video_mime = None
            st.session_state.video_name = None
            st.session_state.video_count = None

        process_now = st.button("Process uploaded video", key="process_video_btn")
        if uploaded_video is not None and process_now:
            with st.spinner("Processing video..."):
                output_path, max_count, output_mime = process_video(
                    model, uploaded_video.read(), conf_threshold
                )
            with open(output_path, "rb") as f:
                st.session_state.video_data = f.read()
            st.session_state.video_mime = output_mime
            st.session_state.video_name = output_path.name
            st.session_state.video_count = max_count

        if st.session_state.video_data is not None:
            st.success(f"Video processed. Max potholes in a frame: {st.session_state.video_count}")
            if st.session_state.video_mime == "video/mp4":
                st.video(st.session_state.video_data, format="video/mp4")
            else:
                st.warning("Preview may not play in browser for AVI codec. Download the file to view.")

            st.download_button(
                "Download annotated video",
                data=st.session_state.video_data,
                file_name=st.session_state.video_name,
                mime=st.session_state.video_mime,
            )

    with tab_webcam:
        st.write("Start webcam for live pothole detection.")
        st.info("If webcam permission prompt appears, allow access in browser.")
        display_scale = st.slider(
            "Webcam display scale",
            min_value=0.5,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Lower value makes webcam preview less zoomed and easier to view.",
        )
        webrtc_streamer(
            key="pothole-webcam",
            video_processor_factory=create_webrtc_processor(model, display_scale, conf_threshold),
            media_stream_constraints={
                "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                "audio": False,
            },
        )


if __name__ == "__main__":
    main()

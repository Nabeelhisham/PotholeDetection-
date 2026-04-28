"""
Central configuration for pothole detection project.
"""

from pathlib import Path

# Project root directory: .../DIP/pothole_detection
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset is already present outside this project folder.
DATASET_PATH = Path("G:/Nabeel/SRM/Courses/Subjects/Projects/DIP/dataset")
DATA_YAML = DATASET_PATH / "data.yaml"
SAMPLE_VIDEO_PATH = DATASET_PATH / "sample_video.mp4"

# Output folders/files
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRAIN_RUN_NAME = "pothole_yolov8n"
TRAIN_WEIGHTS_DIR = MODELS_DIR / TRAIN_RUN_NAME / "weights"
TRAINED_BEST_MODEL = TRAIN_WEIGHTS_DIR / "best.pt"
BEST_MODEL_PATH = MODELS_DIR / "best.pt"
OUTPUT_VIDEO_PATH = OUTPUTS_DIR / "output.mp4"

# Model and inference settings
BASE_MODEL = "yolov8n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
BOX_COLOR = (0, 255, 0)  # Green in BGR

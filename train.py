"""
Train YOLOv8 model on pothole dataset.
"""

import shutil

from ultralytics import YOLO

from config import (
    BASE_MODEL,
    BEST_MODEL_PATH,
    DATA_YAML,
    EPOCHS,
    IMAGE_SIZE,
    MODELS_DIR,
    TRAIN_RUN_NAME,
    TRAINED_BEST_MODEL,
)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    print("Starting training...")
    print(f"Dataset YAML: {DATA_YAML}")

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(MODELS_DIR),
        name=TRAIN_RUN_NAME,
        exist_ok=True,
    )

    if not TRAINED_BEST_MODEL.exists():
        raise FileNotFoundError(f"Trained best model not found: {TRAINED_BEST_MODEL}")

    # Copy best model to a stable path: models/best.pt
    shutil.copy2(TRAINED_BEST_MODEL, BEST_MODEL_PATH)
    print(f"Training complete. Best model saved at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()

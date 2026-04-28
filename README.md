# Pothole Detection System (YOLOv8)

Complete end-to-end pothole detection project using YOLOv8 (Ultralytics), OpenCV, and NumPy.

## Project Structure

```text
pothole_detection/
│── models/
│── outputs/
│── train.py
│── detect.py
│── app.py
│── utils.py
│── config.py
│── requirements.txt
│── README.md
```

Dataset path used by this project:

`G:/Nabeel/SRM/Courses/Subjects/Projects/DIP/dataset/`

## 1) Setup

Open PowerShell in:

`G:/Nabeel/SRM/Courses/Subjects/Projects/DIP/pothole_detection`

Run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train Model

This trains YOLOv8n for 50 epochs using `dataset/data.yaml`.

```powershell
python train.py
```

After training:

- Run artifacts: `models/pothole_yolov8n/`
- Best model copy: `models/best.pt`

## 3) Run Detection

### A) Video detection (default sample video)

```powershell
python detect.py --mode video --source "G:/Nabeel/SRM/Courses/Subjects/Projects/DIP/dataset/sample_video.mp4" --output "G:/Nabeel/SRM/Courses/Subjects/Projects/DIP/pothole_detection/outputs/output.mp4"
```

### B) Image detection

```powershell
python detect.py --mode image --source "G:/path/to/image.jpg" --output "G:/Nabeel/SRM/Courses/Subjects/Projects/DIP/pothole_detection/outputs/output_image.jpg"
```

### C) Webcam detection (optional)

```powershell
python detect.py --mode webcam
```

## 4) Streamlit Web Interface

Run:

```powershell
streamlit run app.py
```

Then open the shown local URL in your browser, upload an image/video, and download the annotated result.

## Output

- Bounding boxes: green
- Label format: `pothole (0.85)`
- Pothole count per frame shown on screen
- Video saved to: `outputs/output.mp4` (video mode)

Press `q` to stop video/webcam inference.

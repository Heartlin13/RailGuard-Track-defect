# RailGuard: Railway Track Defect and Obstacle Detection

RailGuard detects track defects (cracks, missing clips, damaged sleepers) and
obstacles (people, cattle, vehicles, debris) from onboard-camera and drone video,
localises them with bounding boxes, and raises severity-based alerts.

## Features
- YOLOv8-based detection of defects and obstacles
- Track ROI extraction using OpenCV (Canny + Hough lines), so only on-track
  obstacles trigger alerts
- ByteTrack tracking, so each defect is logged once
- Severity levels (HIGH / MEDIUM / LOW) based on distance from the camera
- GPS-tagged CSV reports and a Streamlit dashboard with map view

## Tech Stack
Python 3.11, PyTorch, Ultralytics YOLOv8, OpenCV, pandas, Streamlit

## Usage
python main.py --video data/videos/test.mp4 --out outputs --device cpu
streamlit run app.py

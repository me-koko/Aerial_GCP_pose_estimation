# GCP Pose Estimation System

This project is a computer vision-based solution for aerial Ground Control Point (GCP) pose estimation. It uses a multi-task deep learning model built on ResNet18 to perform:

- **Keypoint Localization** — Predicts the exact center coordinates (x, y) of the GCP marker.
- **Shape Classification** — Identifies the marker shape as Cross, Square, or L-Shape.

The model was trained using PyTorch and deployed with a Flask backend and React frontend for real-time image upload and prediction visualization.
Object Detector using OpenCV and Python
This project demonstrates a real-time object detection system using OpenCV's DNN module with a pre-trained SSD MobileNet v3 model.
It detects over 99+ objects from a live webcam feed, draws bounding boxes around detected objects, and displays their names and confidence scores.

Features:
Real-time object detection using a webcam

Identifies a wide variety of objects from the COCO dataset

Displays object name and detection confidence on the video feed

Lightweight and easy to set up

Technologies Used:
Python

OpenCV

SSD MobileNet v3 (Pre-trained on COCO dataset)

How to Run:
Clone the repository.

Make sure you have the following files:

coco.names

ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

frozen_inference_graph.pb

Install OpenCV:

nginx
Copy
Edit
pip install opencv-python
Run the Python script:

csharp
Copy
Edit
python object detection.py
Output:
The webcam will open, and detected objects will be highlighted with their names and accuracy scores in real-time.

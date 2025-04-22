# Object Detector using OpenCV and Python

This project demonstrates a real-time object detection system using **OpenCV's DNN module** with a **pre-trained SSD MobileNet v3 model**.  
It detects **up to 99+ objects** from a live webcam feed, drawing bounding boxes and showing object names with confidence scores.

## Features
- Real-time object detection through webcam
- Detects up to 99+ object categories from the COCO dataset
- Displays object name and detection confidence
- Lightweight and easy to set up

## Technologies Used
- Python
- OpenCV
- SSD MobileNet v3 (trained on COCO dataset)

## How to Run
1. Clone the repository.

2. Make sure the following files are available:
   - `coco.names`
   - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`
   - `frozen_inference_graph.pb`

3. Install OpenCV:
   ```bash
   pip install opencv-python

4. Run the script::
      ```bash
   python object detection.py
      
## Output
The webcam will open and start detecting objects.
Each detected object will be displayed with a bounding box, label, and accuracy score in real-time.


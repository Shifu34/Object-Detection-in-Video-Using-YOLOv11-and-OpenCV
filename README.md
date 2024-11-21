
# Object Detection in Video Using YOLOv11 and OpenCV

## Overview

This project implements object detection on a video file using a pretrained YOLOv11 model. The YOLO model is used to detect objects in each frame of the video alongside their confidence scores. OpenCV is utilized for video processing, drawing bounding boxes around detected objects, and saving the processed video.

---

## Features

1. **YOLOv11 Model**: Utilizes a pretrained YOLOv11 model for real-time object detection.
2. **Confidence Scores**: Displays the confidence score for each detected object.
3. **Bounding Boxes**: Draws bounding boxes around detected objects in each video frame.
4. **Video Processing**: Reads a video file frame by frame, applies object detection, and saves the annotated video.

---

## Requirements

The following libraries are required to execute the project:

- **Python 3.7 or above**
- **OpenCV**: For video processing and visualization.
- **Numpy**: For numerical operations.
- **YOLOv11 Weights and Config Files**:
  - Pretrained weights: `YOLOv11.weights`
  - Configuration file: `YOLOv11.cfg`
  - COCO class labels: `coco.names`

To install the required packages:
```bash
pip install opencv-python numpy
```

---

## Implementation Details

### 1. Load YOLOv11 Model
The pretrained YOLOv11 model is loaded using OpenCV’s DNN module. The weights, configuration file, and class labels are specified during initialization.

```python
# Load YOLO model
net = cv2.dnn.readNet("YOLOv11.weights", "YOLOv11.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
```

### 2. Read Input Video
OpenCV is used to read a video file frame by frame.

```python
cap = cv2.VideoCapture("input_video.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
```

### 3. Perform Object Detection
Each frame is processed using the YOLO model to detect objects. Detected objects are filtered based on a confidence threshold.

```python
# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
```

### 4. Draw Bounding Boxes
Bounding boxes are drawn for detected objects, and labels with confidence scores are displayed.

```python
for detection in detections:
    # Process detection details
    x, y, w, h = detection['box']
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### 5. Save the Resulting Video
The annotated video is saved using OpenCV’s `VideoWriter`.

```python
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
out.write(frame)
```

---

## Results

The resulting video includes:
- Rectangles drawn around detected objects.
- Labels indicating the object class and confidence score.

The video is saved as `output_video.mp4`.

---

## Usage

1. Place the YOLO weights (`YOLOv11.weights`), configuration file (`YOLOv11.cfg`), and class labels (`coco.names`) in the working directory.
2. Replace `"input_video.mp4"` with the path to your input video.
3. Run the notebook or Python script.
4. The annotated video will be saved as `output_video.mp4` in the working directory.

---

## Notes

1. **Confidence Threshold**: Adjust the confidence threshold in the code for optimal detection:
   ```python
   if confidence > 0.5:  # Example threshold
   ```
2. **Performance**: Processing speed may vary depending on hardware. GPU acceleration is recommended for better performance.

---

## Author
- Name: Shafqat Mehmood
- Contact: shafqat129.mehmood@gmail.com

## License
This project is licensed under the MIT License - see the LICENSE file for details.
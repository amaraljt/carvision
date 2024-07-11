# carvision

carvision is a an object detection system that uses OpenCV to detect vehicles in real-time.

## Dataset
Vehicle detection dataset to detect cars, bikes, ambulances, trucks and motorcycles

Data labels are given in a YOLOv8 format as such: 

``<class_id>, <x_center>, <y_center>, <width>, <height>``

https://www.kaggle.com/datasets/alkanerturan/vehicledetection/data

## Installation

Create python virtual environment and download dependencies with the command:

``` pip3 install -r requirements.txt```

## Model Architecture
- Image Classification + Regression Model:
    produces 2 outputs for classifying the object and localizing it

## Training

Training consists of over 600 images with respective classID and box coordinates

## Classification / Detection
Uses box coordinates on images...

## Tests
Tested Model on Normalized Testing Images with a 47% accuracy.

# Bugs
1. Model detects only cars when using openCV video capture
2. Bounding Box coordinates only stay in the middle of the video capture window
# carvision
The purpose of carvision is to be able to detect an object (vehicles) in real-time. This project can be used to help 

## Datasets
Vehicle detection dataset to detect cars, bikes, ambulances, trucks and motorcycles

Data labels are given in a YOLOv8 format as such: 

``<class_id>, <x_center>, <y_center>, <width>, <height>``

https://www.kaggle.com/datasets/alkanerturan/vehicledetection/data

https://www.kaggle.com/datasets/yusufberksardoan/traffic-detection-project/data?select=valid

## Installation

Create python virtual environment and download dependencies with the command:

``` pip3 install -r requirements.txt```

# carvision 1.0
Trained an object detection model with about 400 training images
No data augmentation

# carvision 2.0
Trained a new object detection model with over 5000 images.
Data augmentation and normalization prior to training



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
1. Model detects only one class
2. Bounding Box coordinates have little to no change
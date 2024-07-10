import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from detect_dataset import DetectNet

# Change device type to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectNet().to(device)

# Load pre-trained model
model.load_state_dict(torch.load('../models/detect_net.pth'))
model.eval()

# Class IDs
class_labels = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Video for testing
video_path = '../data/TestVideo/testvideo1.mp4'

# Create CV Video Capture
cap = cv.VideoCapture(video_path)

# Capture Video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize and apply transformations
    frame_resized = cv.resize(frame, (416, 416))
    frame_transformed = transform(frame_resized)
    frame_transformed = frame_transformed.unsqueeze(0).to(device)  # Add batch dimension and send to device
    
    # Make predictions
    with torch.no_grad():
        class_outputs, bbox_outputs = model(frame_transformed)
        class_predictions = torch.argmax(class_outputs, dim=1)
    
    # Assuming only one object per frame for simplicity
    class_id = class_predictions.item()
    bbox = bbox_outputs[0].cpu().numpy()
    
    # Convert normalized coordinates back to the original frame size
    h, w, _ = frame.shape
    x_center, y_center, width, height = bbox
    x_center, y_center, width, height = x_center * w, y_center * h, width * w, height * h
    x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
    x2, y2 = int(x_center + width / 2), int(y_center + height / 2)
    
    # Draw bounding box and label on the frame
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(frame, class_labels[class_id], (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv.imshow('Detected Frame', frame)
    
    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()
    


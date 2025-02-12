import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model for real-time inference

# Define the detection zone (ROI)
ZONE_TOP_LEFT = (200, 300)  # Adjust based on your video
ZONE_BOTTOM_RIGHT = (600, 500)

# Binary occupancy timeline
occupancy_timeline = []
frame_rate = 4  # Each bit represents 1/4 of a second

# Open the video capture (0 for webcam or provide video file)
cap = cv2.VideoCapture("traffic_video.mp4")  # Replace with 0 for webcam

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % (30 // frame_rate) != 0:
        continue  # Process only every 1/4 sec frame

    # Run YOLOv8 inference
    results = model(frame)

    # Draw the detection zone
    cv2.rectangle(frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, (0, 255, 0), 2)

    # Check if any object is inside the zone
    zone_occupied = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            cls = int(box.cls[0])  # Get class ID

            # Class IDs for pedestrians, cars, bikes, and buses (COCO dataset)
            if cls in [0, 2, 3, 5]:  
                if x1 > ZONE_TOP_LEFT[0] and x2 < ZONE_BOTTOM_RIGHT[0] and \
                   y1 > ZONE_TOP_LEFT[1] and y2 < ZONE_BOTTOM_RIGHT[1]:
                    zone_occupied = 1  # Mark as occupied
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Class: {cls}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Update occupancy timeline
    occupancy_timeline.append(zone_occupied)

    # Display the frame
    cv2.imshow("Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert binary list to a string for logging
occupancy_log = ''.join(map(str, occupancy_timeline))
print(f"Occupancy Timeline: {occupancy_log}")

# Save to a text file
with open("occupancy_log.txt", "w") as file:
    file.write(occupancy_log)

# Plot the occupancy timeline
plt.figure(figsize=(10, 3))
plt.plot(occupancy_timeline, drawstyle="steps-pre")
plt.xlabel("Time (1 step = 1/4 sec)")
plt.ylabel("Zone Occupancy (0=empty, 1=occupied)")
plt.title("Traffic Detection Zone Occupancy Over Time")
plt.show()
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Global variable to store bounding boxes and labels
global_bottle_positions = {}

# Function to convert bounding box from (x_min, y_min, x_max, y_max) to (x, y, w, h)
def convert_bbox(x_min, y_min, x_max, y_max):
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    return x, y, w, h

def detect_bottles(image):
    """
    Detect bottles in the provided image using YOLOv8 model.
    
    Args:
        image: The input image on which detection is to be performed (BGR format).
    
    Returns:
        global_bottle_positions: A dictionary containing bounding box positions for each detected bottle.
    """
    # Load the YOLOv8 model
    model = YOLO('yolov8s.pt')  # Ensure you have the correct model path or name

    # Convert the image to RGB (YOLO expects RGB images)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Extract bounding boxes and labels
    boxes = results[0].boxes.xyxy.numpy()  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    confidences = results[0].boxes.conf.numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.numpy().astype(int)  # Class IDs
    class_names = results[0].names  # Class names

    # Filter detections for the class "bottle"
    bottle_detections = []
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        confidence = confidences[i]
        class_id = class_ids[i]
        if class_names[class_id] == 'bottle':  # Filter for "bottle" class
            bottle_detections.append({
                'xmin': x_min,
                'ymin': y_min,
                'xmax': x_max,
                'ymax': y_max,
                'confidence': confidence,
                'name': class_names[class_id]
            })

    # Convert the list of dictionaries to a DataFrame
    bottle_detections_df = pd.DataFrame(bottle_detections)

    # Sort the detections from left to right based on the x_min coordinate
    bottle_detections_df = bottle_detections_df.sort_values('xmin')

    # Update the global bottle positions dictionary
    global global_bottle_positions
    global_bottle_positions.clear()  # Clear the previous entries if any

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_rgb)

    # Define custom colors
    bbox_color = 'blue'  # Color for bounding box edge
    label_color = 'white'  # Color for label text
    bbox_fill_color = 'none'  # Transparent for the bounding box fill

    for index, row in bottle_detections_df.iterrows():
        x_min, y_min, x_max, y_max = row[['xmin', 'ymin', 'xmax', 'ymax']]
        x, y, w, h = convert_bbox(x_min, y_min, x_max, y_max)
        confidence = row['confidence']

        # Draw bounding box on the image
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=bbox_color, facecolor=bbox_fill_color)
        ax.add_patch(rect)

        # Add label with confidence score
        ax.text(x, y, f'bottle_{index + 1} {confidence:.2f}', bbox=dict(facecolor='green', alpha=0.5), fontsize=12, color=label_color)

        # Store the bounding box in the global dictionary
        label = f'bottle_{index + 1}'
        global_bottle_positions[label] = (x, y, w, h)

        # Print the bounding box information
        print(f"{label}: (x={x}, y={y}, w={w}, h={h})")

    # Hide axes and show the image
    plt.axis('off')
    plt.show()

    return global_bottle_positions

import cv2
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
import math

# Helper function to convert (x, y, w, h) to (x_min, y_min, x_max, y_max)
def convert_center_format_to_corners(x, y, w, h):
    # Converts center format bounding boxes to corner format
    x_min = x - (w / 2)
    y_min = y - (h / 2)
    x_max = x + (w / 2)
    y_max = y + (h / 2)
    return x_min, y_min, x_max, y_max

def filter_masks_by_bounding_boxes(converted_boxes, masks):
    # Filters masks that fall within bounding boxes
    global_filtered_masks = {}

    for index, bbox in enumerate(converted_boxes):
        x_center, y_center, width, height = bbox
        x_min_2, y_min_2, x_max_2, y_max_2 = convert_center_format_to_corners(x_center, y_center, width, height)

        filtered_masks = []

        for mask in masks:
            x_min, y_min, mask_width, mask_height = mask['bbox']
            x_max = x_min + mask_width
            y_max = y_min + mask_height

            # Check if the mask's bbox is within the current bbox
            if (x_min_2 <= x_min <= x_max_2) and (y_min_2 <= y_min <= y_max_2) and (x_max_2 >= x_max) and (y_max_2 >= y_max):
                filtered_masks.append(mask)

        global_filtered_masks[f'bbox{index + 1}'] = filtered_masks

    return global_filtered_masks

def analyze_filtered_masks(global_filtered_masks):
    # Analyzes the filtered masks for each bounding box
    all_bbox_data = {}

    for bbox_key, data in global_filtered_masks.items():
        areas = []
        bboxes = []
        predicted_ious = []
        point_coords_list = []
        stability_scores = []
        crop_boxes = []

        for item in data:
            if item['area'] > 1300:
                areas.append(item['area'])
                bboxes.append(item['bbox'])
                predicted_ious.append(item['predicted_iou'])
                point_coords_list.append(item['point_coords'])
                stability_scores.append(item['stability_score'])
                crop_boxes.append(item['crop_box'])

        all_bbox_data[bbox_key] = {
            'Areas': areas,
            'Bounding Boxes': bboxes,
            'Predicted IOUs': predicted_ious,
            'Point Coordinates': point_coords_list,
            'Stability Scores': stability_scores,
            'Crop Boxes': crop_boxes
        }

    return all_bbox_data

# Add the show_anns function to visualize the masks
def show_anns(anns):
    """Visualizes mask annotations."""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def visualize_filtered_masks(image, flattened_masks):
    """
    Visualizes the filtered masks on the image.

    Args:
        image: The original image.
        flattened_masks: A list of filtered masks to visualize.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))

    # Get the segmentations for the masks
    masks = [mask['segmentation'] for mask in flattened_masks]

    # Dynamically set grid size based on the number of masks
    num_masks = len(masks)
    
    # Limit the number of masks to display if necessary
    max_masks_to_display = 10  # You can adjust this limit
    if num_masks > max_masks_to_display:
        masks = masks[:max_masks_to_display]
        num_masks = max_masks_to_display
        print(f"Limiting to the first {max_masks_to_display} masks for display.")
    
    # Set grid size dynamically
    num_columns = 5  # Set a fixed number of columns
    num_rows = math.ceil(num_masks / num_columns)  # Calculate the number of rows based on the number of masks
    
    sv.plot_images_grid(
        images=masks,
        grid_size=(num_rows, num_columns),
        size=(16, 10)  # Adjust size as needed
    )
    
    plt.imshow(image_rgb)
    
    # Use the show_anns function to display masks
    show_anns(flattened_masks)
    
    plt.axis('off')
    plt.show()

def create_named_regions(all_bbox_data):
    # Creates named regions based on bounding boxes and mask data
    new_regions = []

    for bbox_key, bbox_data in all_bbox_data.items():
        bboxes = bbox_data['Bounding Boxes']
        
        for i, bbox in enumerate(bboxes):
            region_name = f"{bbox_key}_region{i+1}"
            x, y, w, h = bbox
            new_regions.append((region_name, x, y, w, h))

    return new_regions

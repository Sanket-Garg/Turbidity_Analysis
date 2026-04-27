import cv2
import numpy as np

# Mouse callback function to handle bounding box drawing
def draw_rectangles(event, x, y, flags, param):
    global ix, iy, drawing, rectangles, current_rectangle, resized_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start a new rectangle
        ix, iy = x, y
        drawing = True
        current_rectangle = (ix, iy, ix, iy)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the current rectangle while dragging
            current_rectangle = (ix, iy, x, y)
            temp_image = resized_image.copy()
            display_instructions(temp_image)
            for rect in rectangles:
                cv2.rectangle(temp_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            if current_rectangle:
                cv2.rectangle(temp_image, (current_rectangle[0], current_rectangle[1]), 
                              (current_rectangle[2], current_rectangle[3]), (0, 255, 0), 2)
            cv2.imshow('image', temp_image)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Finalize the current rectangle
        if current_rectangle:
            rectangles.append(current_rectangle)
            current_rectangle = None
        temp_image = resized_image.copy()
        display_instructions(temp_image)
        for rect in rectangles:
            cv2.rectangle(temp_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.imshow('image', temp_image)
        print("Bounding Boxes:", rectangles)

def display_instructions(image):
    """Overlay instructions on the image."""
    cv2.putText(image, "Press 'r' to stop, 'Space' to continue.", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def handle_key_press(key):
    global mode, rectangles

    if key == ord('r'):
        # Finish drawing and exit
        mode = 'stop'
        print("Final Bounding Boxes:", rectangles)
        return True  # Indicates that the drawing process should end
    elif key == 32:  # Space bar key
        # Continue adding more bounding boxes
        mode = 'add'
        print("Continue adding more bounding boxes.")
        return False  # Continue the process

def convert_bbox_format(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (x, y, w, h).
    """
    converted_boxes = []
    for (x1, y1, x2, y2) in boxes:
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        converted_boxes.append((x_center, y_center, width, height))
    return converted_boxes

def select_bounding_boxes(image):
    """Main function to select bounding boxes on the provided image."""
    global ix, iy, drawing, rectangles, mode, resized_image

    # Initialize variables
    ix, iy, drawing = -1, -1, False
    current_rectangle = None
    rectangles = []
    mode = 'add'  # Initial mode

    # Resize the image for display
    resize_scale = 0.5  # Scale factor (0.5 means 50% of the original size)
    width = int(image.shape[1] * resize_scale)
    height = int(image.shape[0] * resize_scale)
    resized_image = cv2.resize(image, (width, height))

    # Display initial instructions
    display_instructions(resized_image)
    cv2.imshow('image', resized_image)
    cv2.setMouseCallback('image', draw_rectangles)

    while True:
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press
        if handle_key_press(key):
            break  # Exit the loop and end the drawing process

    # Convert bounding boxes back to original image scale if necessary
    rectangles_original_scale = [(int(x1 / resize_scale), int(y1 / resize_scale), 
                                  int(x2 / resize_scale), int(y2 / resize_scale)) for (x1, y1, x2, y2) in rectangles]

    # Convert to (x, y, w, h) format
    converted_boxes = convert_bbox_format(rectangles_original_scale)

    # Sort the bounding boxes from left to right by x-center
    sorted_bounding_boxes = sorted(converted_boxes, key=lambda box: box[0])
    
    # Close the OpenCV window
    cv2.destroyAllWindows()
    
    return sorted_bounding_boxes

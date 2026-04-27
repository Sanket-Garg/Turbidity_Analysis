import os
import cv2
import argparse
from video_processing import get_frames_directory, extract_frames_from_video
from image_segmentation import annotate_and_show_images
from bounding_box_selector import select_bounding_boxes
from yolo_bottle_detection import detect_bottles, global_bottle_positions  # Import the global variable here
from mask_filtering_and_analysis import filter_masks_by_bounding_boxes, analyze_filtered_masks, visualize_filtered_masks, create_named_regions
from turbidity_measurement import TurbidityMeasurement, crop_and_measure, load_images_from_folder, show_last_image_with_annotations
from region_processing import label_and_classify_layers  # Import the region processing logic

def store_images(folder_path: str) -> list:
    images = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
    if not images:
        raise ValueError("No images found in the specified folder.")
    return images

def main(VIDEO_PATH, SAM_CHECKPOINT, MODEL_TYPE, FRAME_INTERVAL, layer_names, DEVICE) -> str:
    # Step 1: Video frame extraction
    frames_directory = get_frames_directory(VIDEO_PATH)
    print(f"Frames will be stored in: {frames_directory}")

    extract_frames_from_video(VIDEO_PATH, frames_directory, FRAME_INTERVAL)
    images = store_images(frames_directory)
    print(f"Total images stored: {len(images)}")

    # Step 2: Image segmentation and annotation, returns the masks
    masks = annotate_and_show_images(images, SAM_CHECKPOINT, MODEL_TYPE, DEVICE)

    # Step 3: Select bounding boxes on the last image
    if images:
        last_image = images[-1]
        bounding_boxes = select_bounding_boxes(last_image)
        print(f"Bounding Boxes: {bounding_boxes}")

        # Step 4: Perform bottle detection using YOLOv8 on the last image
        print("Performing bottle detection using YOLOv8...")
        bottle_positions = detect_bottles(last_image)  # global_bottle_positions will be updated here
        print(f"Bottle Positions: {bottle_positions}")

        # Step 5: Filter and analyze masks by bounding boxes
        print("Filtering masks by bounding boxes...")
        global_filtered_masks = filter_masks_by_bounding_boxes(bounding_boxes, masks)
        all_bbox_data = analyze_filtered_masks(global_filtered_masks)

        # Step 6: Visualize filtered masks
        print("Visualizing filtered masks...")
        flattened_masks = [mask for sublist in global_filtered_masks.values() for mask in sublist]
        visualize_filtered_masks(last_image, flattened_masks)

        # Step 7: Create named regions for each bounding box
        named_regions = create_named_regions(all_bbox_data)
        print("Named regions:", named_regions)

        # --- First Turbidity Measurement (before labeling) ---
        print("Measuring turbidity before labeling...")
        turbidity_instance = TurbidityMeasurement(datetime_format="%Y-%m-%d %H:%M:%S")
        crop_and_measure(images, named_regions, turbidity_instance)

        # Generate and display the first turbidity graph
        turbidity_instance.make_turbidity_over_time_graph_with_stable_visualization()
        show_last_image_with_annotations(last_image, named_regions)

        # Step 8: Region Classification and Layer Labeling
        print("Classifying and labeling layers...")
        final_convert_list, flattened_list = label_and_classify_layers(global_bottle_positions, named_regions, layer_names)
        print("Final Classified Regions:", final_convert_list)

        # --- Second Turbidity Measurement (after labeling) ---
        print("Measuring turbidity after labeling...")
        for i, regions in enumerate(final_convert_list):
            # Create a new instance of TurbidityMeasurement for each set of regions
            turbidity_instance = TurbidityMeasurement(datetime_format="%Y-%m-%d %H:%M:%S")

            # Process each image with timestamps and defined regions
            crop_and_measure(images, regions, turbidity_instance)

            # Generate and display graph for this specific set of regions
            turbidity_instance.make_turbidity_over_time_graph_with_stable_visualization(
                x_axis_units='minutes',
                scatter=False
            )

            # Get stored values and print them
            stored_values = turbidity_instance.get_stored_values()
            print(f"Results for Region Set {i + 1}:")
            print(f"x values: {stored_values['x_values']}")
            print(f"y values: {stored_values['y_values']}")

            # Show the last image with annotations for this specific set of regions
            if images:
                last_image = images[-1]
                show_last_image_with_annotations(last_image, regions)

            # Optionally, save the graph and annotated image with unique filenames
            # turbidity_instance.save_graph(f"turbidity_graph_set_{i + 1}.png")
            # save_last_image_with_annotations(last_image, regions, f"annotated_image_set_{i + 1}.png")

    return frames_directory

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process video for SAM-based image segmentation and turbidity measurement.")

    # Required arguments
    parser.add_argument('--video-path', type=str, required=True, help="Path to the video file")
    parser.add_argument('--sam-checkpoint', type=str, required=True, help="Path to the SAM checkpoint file")
    parser.add_argument('--model-type', type=str, required=True, choices=['vit_h', 'vit_l', 'vit_b'], help="Type of model (e.g., 'vit_h')")

    # Optional arguments with default values
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run the model on (default: 'cpu')")
    parser.add_argument('--frame-interval', type=int, default=8, help="Interval between frames in seconds (default: 8)")
    parser.add_argument('--layer-names', nargs='+', default=['sediment', 'water', 'oil', 'foam'], help="List of layer names (default: ['sediment', 'water', 'oil', 'foam'])")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    frames_directory = main(
        VIDEO_PATH=args.video_path,
        SAM_CHECKPOINT=args.sam_checkpoint,
        MODEL_TYPE=args.model_type,
        FRAME_INTERVAL=args.frame_interval,
        layer_names=args.layer_names,
        DEVICE=args.device
    )

    if frames_directory:
        print(f"Frames are stored in the directory: {frames_directory}")

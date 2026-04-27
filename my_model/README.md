# Colloidal Solution Turbidity Analysis Model

## Overview

This project is designed to analyze the turbidity of colloidal solutions using a combination of computer vision techniques. The system integrates various modules for region selection, segmentation, analysis, and object detection using YOLOv8. The key features include:

- **YOLOv8**: For detecting vials in images or videos.
- **SAM (Segment Anything Model)**: For layer segmentation and processing.
- **Turbidity Measurement**: For calculating and analyzing the turbidity of the solution.

## File Structure

- `main.py`: Runs the entire system, coordinating tasks across modules.
- `bounding_box_selector.py`: Allows for the selection of regions of interest (ROI).
- `image_segmentation.py`: Handles layer segmentation.
- `mask_filtering_and_analysis.py`: Filters and analyzes segmented regions.
- `region_processing.py`: Focuses on layer and phase processing.
- `turbidity_measurement.py`: Calculates turbidity from the segmented regions.
- `video_processing.py`: Processes video input for vial detection and analysis.
- `yolo_bottle_detection.py`: Uses YOLOv8 for vial detection.
- `requirements.txt`: Lists all the dependencies required to run the project.

## Requirements

- **Python 3.8+**
- Required Libraries: OpenCV, YOLOv8, SAM API, and more, as specified in `requirements.txt`.

To install dependencies, use:

```bash
pip install -r requirements.txt
```
## Setup

### Clone the Repository:

First, clone the project repository to your local machine using the following command:

```bash
git clone https://github.com/LARC-Lab/robot-soft-matter-physicist.git
```
###Install Dependencies:
Navigate to the project directory and install all the required dependencies using pip:

```bash
pip install -r requirements.txt
```
## Download SAM checkpoint:
Download the SAM model checkpoint from [this link](https://github.com/facebookresearch/segment-anything).

# Running the Model

To run the model, you must execute main.py with three required inputs:

- **Video Path:** The path to the video file containing vials for analysis.
- **SAM Model Checkpoint Path:** The path to the SAM model checkpoint file you downloaded earlier.
- **SAM Model Type:** Specify the type of SAM model (e.g., vit_b).
Optional parameters allow you to customize how the model runs, such as the device type (e.g., CPU or GPU), frame interval for processing videos, and specific layer names for analysis.

### Example command
```bash
python main.py --video_path path/to/video.mp4 --sam_checkpoint path/to/sam_checkpoint.pth --sam_model_type vit_b 
```
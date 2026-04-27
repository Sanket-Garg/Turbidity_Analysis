import os
import cv2
import numpy as np

FRAME_INTERVAL = 8  # Default interval, can be overridden

def get_frames_directory(video_path: str, frames_folder_name: str = 'Frames') -> str:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_directory = os.path.join(os.path.dirname(video_path), f"{base_name}_{frames_folder_name}")
    os.makedirs(frames_directory, exist_ok=True)
    return frames_directory

def correct_frame_orientation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, flipCode=1)
    elif rotation == 180:
        frame = cv2.flip(frame, flipCode=-1)
    elif rotation == 270:
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, flipCode=0)
    return frame

def extract_frames_from_video(video_path: str, frames_directory: str, frame_interval: int = FRAME_INTERVAL) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * frame_interval)
    frame_count = 0

    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        rotation = 90  # Adjust if needed
        frame = correct_frame_orientation(frame, rotation)

        frame_path = os.path.join(frames_directory, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        print(f"Frame at {frame_count / fps:.2f} seconds saved as {frame_path}")

        frame_count += frame_interval

    cap.release()
    print("Processing completed.")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple

class TurbidityMeasurement:
    def __init__(self, datetime_format: str):
        self._datetime_format = datetime_format
        self.raw_turbidity_data = {}
        self.turbidity_monitoring_start_time = None
        
        # Attributes to store x and y values
        self.x_values = {}
        self.y_values = {}

    def add_measurement(self, image: np.ndarray, region_name: str, time: str = None) -> None:
        if time is None:
            t = datetime.now()
            t_str = t.strftime(self._datetime_format)
        else:
            t_str = time

        if len(self.raw_turbidity_data) == 0:
            t = datetime.strptime(t_str, self._datetime_format)
            self.turbidity_monitoring_start_time = t
        else:
            t = datetime.strptime(t_str, self._datetime_format)

        if not isinstance(image, np.ndarray):
            raise TypeError("Image is not a numpy array.")

        turbidity = self.turbidity_measurement(image)
        if region_name not in self.raw_turbidity_data:
            self.raw_turbidity_data[region_name] = []
        self.raw_turbidity_data[region_name].append((t_str, turbidity))

    def turbidity_measurement(self, image: np.ndarray) -> float:
        blur_kernel = (5, 5)

        def measure(image: np.ndarray) -> float:
            if not isinstance(image, np.ndarray):
                raise TypeError("Image in measure is not a numpy array.")
                
            image = cv2.GaussianBlur(image, blur_kernel, 0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_for_image = np.mean(image, axis=(0, 1))
            v = hsv_for_image[2]
            return v

        return measure(image)

    def make_turbidity_over_time_graph_with_stable_visualization(self, x_axis_units='minutes', scatter=False) -> None:
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.raw_turbidity_data)))

        for (region_name, data), color in zip(self.raw_turbidity_data.items(), colors):
            times, turbidity_values = zip(*data)
            times = [datetime.strptime(t, self._datetime_format) for t in times]
            
            if x_axis_units == 'minutes':
                time_deltas = [(t - times[0]).total_seconds() / 60 for t in times]
                x_label = 'Time (minutes)'
            else:
                time_deltas = [(t - times[0]).total_seconds() for t in times]
                x_label = 'Time (seconds)'

            self.x_values[region_name] = time_deltas
            self.y_values[region_name] = turbidity_values

            if scatter:
                plt.scatter(time_deltas, turbidity_values, color=color, label=f'Region: {region_name}')
            else:
                plt.plot(time_deltas, turbidity_values, marker='o', linestyle='-', color=color, label=f'Region: {region_name}')

        plt.xlabel(x_label)
        plt.ylabel('Turbidity')
        plt.title('Turbidity Measurements Over Time for Multiple Regions')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_stored_values(self) -> dict:
        return {'x_values': self.x_values, 'y_values': self.y_values}

def get_central_bbox(x: int, y: int, w: int, h: int, sub_w: int, sub_h: int) -> Tuple[int, int, int, int]:
    center_x = x + w // 2
    center_y = y + h // 2
    sub_x = max(center_x - sub_w // 2, 0)
    sub_y = max(center_y - sub_h // 2, 0)
    return sub_x, sub_y, sub_w, sub_h

def crop_and_measure(images: List[np.ndarray], regions: List[Tuple[str, int, int, int]], turbidity_instance: TurbidityMeasurement) -> None:
    initial_time = datetime.now()
    for i, img in enumerate(images):
        if img is not None:
            current_time = initial_time + timedelta(seconds=9 * i)
            current_time_str = current_time.strftime(turbidity_instance._datetime_format)
            for region_name, x, y, w, h in regions:
                sub_w = w // 2
                sub_h = h // 2
                sub_x, sub_y, sub_w, sub_h = get_central_bbox(x, y, w, h, sub_w, sub_h)
                sub_x = min(sub_x, img.shape[1] - sub_w)
                sub_y = min(sub_y, img.shape[0] - sub_h)
                cropped_image = img[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
                turbidity_instance.add_measurement(cropped_image, region_name=region_name, time=current_time_str)

def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {img_path}")
    return images

def show_last_image_with_annotations(image: np.ndarray, regions: List[Tuple[str, int, int, int]]) -> None:
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    image_with_boxes = image.copy()
    
    for i, (region_name, x, y, width, height) in enumerate(regions):
        color = colors[i % len(colors)]
        cv2.rectangle(image_with_boxes, (x, y), (x + width, y + height), color, 2)
        cv2.putText(image_with_boxes, region_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Last Image with Region Annotations')
    plt.show()


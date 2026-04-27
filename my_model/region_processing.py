import numpy as np

def calculate_overlap(region1, region2):
    """Calculate the overlap area between two regions."""
    _, x1, y1, w1, h1 = region1
    _, x2, y2, w2, h2 = region2

    overlap_x1 = max(x1, x2)
    overlap_y1 = max(y1, y2)
    overlap_x2 = min(x1 + w1, x2 + w2)
    overlap_y2 = min(y1 + h1, y2 + h2)

    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)

    overlap_area = overlap_width * overlap_height
    area1 = w1 * h1
    area2 = w2 * h2

    return overlap_area, area1, area2

def should_remove(region1, region2):
    """Determine if one region should be removed based on overlap."""
    overlap_area, area1, area2 = calculate_overlap(region1, region2)

    if area1 >= area2:
        smaller_area, larger_area = area2, area1
        smaller_region = region2
        larger_region = region1
    else:
        smaller_area, larger_area = area1, area2
        smaller_region = region1
        larger_region = region2

    overlap_ratio = overlap_area / smaller_area
    if overlap_ratio > 0.99:
        return larger_region

    return None

def remove_overlapping_regions(regions):
    """Remove overlapping regions and keep the smaller ones."""
    regions_to_remove = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            to_remove = should_remove(regions[i], regions[j])
            if to_remove and to_remove not in regions_to_remove:
                regions_to_remove.append(to_remove)

    remaining_regions = [region for region in regions if region not in regions_to_remove]
    
    return remaining_regions, regions_to_remove

def classify_layers(bottle_coords, y1_regions, layer_names):
    """Classify the regions in y1 based on proximity to the bottom of bottle_coords."""
    distances = [(region, calculate_bottom_distance(bottle_coords, region)) for region in y1_regions]
    distances.sort(key=lambda x: x[1])  # Sort by distance from the bottom of the bottle

    classifications = {}

    for i, (region, _) in enumerate(distances):
        if i < len(layer_names):
            classifications[layer_names[i]] = region

    return classifications

def calculate_bottom_distance(x1, y1):
    """Calculate the distance from the bottom of x1 to the bottom of y1."""
    x1_bottom = x1[1] + x1[3]  # y + h of x1
    y1_bottom = y1[1] + y1[3]  # y + h of y1
    return np.abs(x1_bottom - y1_bottom)

def label_and_classify_layers(global_bottle, regions, layer_names):
    """Process the regions to remove overlaps and classify layers based on user-defined layer names."""
    remaining_regions, removed_regions = remove_overlapping_regions(regions)

    print("Remaining Regions:", remaining_regions)
    print("Removed Regions:", removed_regions)

    # Create a dictionary to label the remaining regions
    bbox_dict = {}
    for item in remaining_regions:
        key, x, y, width, height = item
        bbox_key = key.split('_')[0]
        if bbox_key not in bbox_dict:
            bbox_dict[bbox_key] = {}
        bbox_dict[bbox_key][key] = (x, y, width, height)

    # Classify the layers based on proximity to the bottom of the bottle
    final_convert_list = []
    global_bottle_keys = list(global_bottle.keys())
    global_bottle_values = list(global_bottle.values())
    bbox_list = list(bbox_dict.items())

    for i, (bottle_key, bottle_coords) in enumerate(zip(global_bottle_keys, global_bottle_values)):
        bbox_key, bbox_regions = bbox_list[i]
        y1_regions = list(bbox_regions.values())
        
        print(f"Classifying layers for {bottle_key}: {bottle_coords}")
        classified_layers = classify_layers(bottle_coords, y1_regions, layer_names)
        
        for layer_name, bbox in classified_layers.items():
            print(f"  {layer_name.capitalize()}: {bbox}")

        converted_list = [(layer_name, *bbox) for layer_name, bbox in classified_layers.items()]
        final_convert_list.append(converted_list)

    # Flatten the list of classified layers
    flattened_list = [item for sublist in final_convert_list for item in sublist]
    print("Final Flattened List:", flattened_list)
    
    # Return both the final_convert_list and the flattened_list
    return final_convert_list, flattened_list

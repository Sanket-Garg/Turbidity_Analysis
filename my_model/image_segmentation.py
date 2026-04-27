import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import supervision as sv
import cv2  # Make sure to include this for cv2.cvtColor

# Set up SAM model
def setup_sam_model(sam_checkpoint: str, model_type: str, device: str = "cpu") -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)

# Show annotations on image
def show_anns(anns: list) -> None:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Annotate and return images with masks for future use
def annotate_and_show_images(images: list, sam_checkpoint: str, model_type: str, device: str) -> list:
    if not images:
        print("No images to process.")
        return []

    image = images[-1]  # For this example, we just use the last image in the list
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_generator = setup_sam_model(sam_checkpoint, model_type, device)
    masks = mask_generator.generate(image)

    # Display the image and its masks
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

    # Print the masks if required (debugging or viewing)
    print("Generated masks:", masks)

    # Annotate the image with the masks
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=masks)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )
    plt.show()

    # Return the masks for later use
    return masks

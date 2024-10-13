import cv2
import numpy as np

# shoelace formula
def polygon_area(x: np.array, y: np.array) -> float:
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def draw_polygon(image: np.array, polygon: np.array, color: tuple = (0, 255, 0), thickness: int = 2) -> np.array:
    return cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)


def get_size_distribution_lower_and_upper_bounds(sizes: np.array, tail: int, outlier_threshold: float) -> tuple(float):
    sizes = np.array(sizes)

    # Compute IQR and outlier bounds
    Q1 = np.percentile(sizes, tail)
    Q3 = np.percentile(sizes, 100 - tail)
    IQR = Q3 - Q1
    lower_bound = Q1 - outlier_threshold * IQR
    upper_bound = Q3 + outlier_threshold * IQR

    return lower_bound, upper_bound


# mask functions
def find_mask_components(mask: np.array, label: int) -> list(float):
    component_sizes = []
    # filter mask by label
    label_mask = (mask == label).astype(np.uint8)

    # Find connected components (each separate shape)
    num_labels, labels_im = cv2.connectedComponents(label_mask)

    # Collect sizes of connected component, skipping background '0' label
    for i in range(1, num_labels):
        component_sizes = np.sum(labels_im == i)
    return component_sizes


def parse_json_segmentation(json_data: dict) -> list(dict):
    masks = []
    for mask in json_data['masks']:
        mask_id = mask['id']
        mask_data = np.array(mask['data'])
        masks.append({'id': mask_id, 'data': mask_data})
    return masks


# mask functions
def flag_mask_outliers(masks: list(dict), n_classes: int, outlier_threshold: float = 1.5) -> list(str):
    """
    Analyzes the sizes of connected components across a dataset of masks,
    creates a global distribution of label sizes, and flags outliers.

    Args:
    - masks (list of dicts): A list of dicts where each element is a mask_id
      and a 2D array containing a mask with labeled regions.
    - n_classes (int): Number of classes in the dataset.
    - outlier_threshold (float): Threshold multiplier for IQR to flag
      outliers (default is 1.5).

    Returns:
    - outliers (list): A dictionary containing the id of outlier masks.
    """
    mask_label_areas = {}
    outliers = []

    # Loop through each mask in the dataset
    for mask in masks:
        # initialize label areas to handle cases with empty masks
        mask_label_areas[mask['id']] = {label: [] for label in range(n_classes)}
        for label in range(n_classes):
            # Add list of label areas
            mask_label_areas[mask['id']][label] = find_mask_components(mask['data'], label)

    # Compute global distribution of label sizes
    # flatten dictinoary by two levels
    sizes = [size for mask in mask_label_areas.values() for label_sizes in mask.values() for size in label_sizes]
    lower_bound, upper_bound = get_size_distribution_lower_and_upper_bounds(sizes, 15, outlier_threshold)
    outliers = find_outliers(mask_label_areas, upper_bound, lower_bound)

    return outliers


def find_outliers(label_sizes: dict, upper_bound: float, lower_bound: float) -> list(str):
    outliers = []
    for mask in label_sizes:
        flagged = False
        for label in label_sizes[mask]:
            for size in label_sizes[mask][label]:
                if not (lower_bound <= size <= upper_bound):
                    flagged = True
                    break
            if flagged:
                outliers.append(mask)
                break
    return outliers


def overlay_mask(image: np.array, mask: np.array, color_map: dict) -> np.array:
    overlay = image.copy()
    for label, color in color_map.items():
        overlay[mask == label] = color

    # merge the two images
    merged_image = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    return merged_image

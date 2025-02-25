import torch
import torch.nn.functional as F
import numpy as np
import settings

def logit_to_one_hot(logit, n_classes):

    assert logit.shape[1] == n_classes

    pred = F.softmax(logit, dim=1)
    one_hot = F.one_hot(pred.argmax(dim=1), num_classes=n_classes).permute(0, 3, 1, 2)

    return one_hot

def mask_to_one_hot(mask, n_classes):

    one_hot = F.one_hot(mask, num_classes=n_classes).permute(0, 3, 1, 2)

    return one_hot

def one_hot_to_mask(one_hot):

    return one_hot.argmax(dim = 1)

def mask_to_rgb(mask, class_colors):
    """
    Convert a single-channel class ID image to an RGB image using specific RGB codes for each class ID.

    Parameters:
    - mask: 2D numpy array of type long, where each pixel value corresponds to a class ID.
    - class_colors: A dictionary where keys are class IDs and values are tuples of (R, G, B) values.

    Returns:
    - rgb_image: 3D numpy array of type uint8, representing the RGB image.
    """
    # Create a lookup table for class IDs to RGB colors
    max_class_id = max(class_colors.keys(), default=0)
    lookup_table = np.zeros((max_class_id + 1, 3), dtype=np.uint8)
    
    # Fill the lookup table with the corresponding RGB colors
    for class_id, color in class_colors.items():
        lookup_table[class_id] = color
    
    # Use the lookup table to map class IDs to RGB colors
    rgb_image = lookup_table[mask]
    
    return rgb_image / 255.0

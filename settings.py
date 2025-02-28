import torch

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A

# Weights of classes in dataset
dataset_weights = torch.tensor([0.0005, 0.0005, 0.2080, 0.1288, 0.0327, 0.0208, 0.0980, 0.0535, 0.0654, 0.3863, 0.0007, 0.0050])

rgb_map = {
    0: (255, 36, 0),
    1: (0, 0, 0),
    2: (242, 216, 196),
    3: (89, 70, 54),
    4: (166, 166, 166),
    5: (82, 89, 90),
    6: (155, 230, 0),
    7: (0, 138, 53),
    8: (0, 216, 245),
    9: (13, 127, 252),
    10: (255, 249, 0),
    11: (254, 0, 170)
}

transform = A.Compose ([
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.1),
    A.Affine(
        scale=(0.9, 1.1),                   # Equivalent to scale_limit=(-0.1, 0.1)
        translate_percent=(-0.0625, 0.0625),  # Equivalent to shift_limit=(-0.0625, 0.0625)
        rotate=(-10, 10),                   # Equivalent to rotate_limit=(-10, 10)
        shear=0,                            # No shearing
        fill_mask=1,                             # Fill value, similar to fill_mask=1
        p=0.5
    ),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0, 0), p=0.1),
    A.GaussianBlur(blur_limit=5, p=0.2),
    A.ElasticTransform(alpha=50.0, sigma=100.0, p=0.2),
    A.GridDistortion(p=0.2),
    A.OpticalDistortion(distort_limit=(-0.1, 0.1), mode="camera", p=0.2),  
], additional_targets={'image_extra': 'image'})

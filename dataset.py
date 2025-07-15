from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import cv2
import json
import os
import settings


class CaltechDataset(Dataset):

    def __init__(self, images_dir, masks_dir, scale = 1.0, transform=None):
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform = transform

        self.dataset = self._load_dataset()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        
        image_file = self.dataset[index, 0]
        mask_file = self.dataset[index, 1]

        image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)

        image = self._preprocess(image, False)
        mask = self._preprocess(mask, True)
        
        if self.transform is None:
            image = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32)
            mask = torch.from_numpy(mask).to(dtype=torch.int64)
        else:
            aug = self.transform(image=image, mask=mask)
            image = torch.from_numpy(aug["image"]).permute(2, 0, 1).to(dtype=torch.float32)
            mask = torch.from_numpy(aug["mask"]).to(dtype=torch.int64)

        return image, mask
    
    def _preprocess(self, image, is_mask=False):
        
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)

        if is_mask:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            image = image[:, :, 0].astype(np.int64)
        else:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = (image / 255.0).astype(np.float32)

        return image

        

    def _load_dataset(self):
        
        image_files = glob.glob(self.images_dir + "/" + "*.png")
        mask_files = glob.glob(self.masks_dir + "/" + "*.png")

        image_files = sorted(image_files)
        mask_files = sorted(mask_files)

        if len(image_files) != len(mask_files):
            raise ValueError("Mismatch between number of images and masks.")

        return np.column_stack((image_files, mask_files))
    
class CaltechRgbtDataset(Dataset):

    def __init__(self, images_root_dir, masks_dir, scale = 1.0):
        super().__init__()

        self.images_root_dir = images_root_dir
        self.masks_dir = masks_dir
        self.scale = scale

        self.dataset = self._load_dataset()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        
        image_rgb_file = self.dataset[index, 0]
        image_ir_file = self.dataset[index, 1]
        mask_file = self.dataset[index, 2]

        image_rgb = cv2.cvtColor(cv2.imread(image_rgb_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image_ir = cv2.cvtColor(cv2.imread(image_ir_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)

        image_rgb = self._preprocess(image_rgb, False)
        image_ir = self._preprocess(image_ir, False)
        mask = self._preprocess(mask, True)

        
        # Add a new axis to the grayscale image to match dimensions
        image = np.concatenate((image_rgb, image_ir[:, :, 0, np.newaxis]), axis=2)

        image = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32)
        mask = torch.from_numpy(mask).to(dtype=torch.int64)

        return image, mask
    
        
    
    def _preprocess(self, image, is_mask=False):
        
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)

        if is_mask:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            image = image[:, :, 0].astype(np.int64)
        else:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = (image / 255.0).astype(np.float32)

        return image

        

    def _load_dataset(self):
        
        image_rgb_files = glob.glob(self.images_root_dir + "/" + "color" + "/" + "*.png")
        image_ir_files = glob.glob(self.images_root_dir + "/" + "thermal8" + "/" + "*.png")
        mask_files = glob.glob(self.masks_dir + "/" + "*.png")

        image_rgb_files = sorted(image_rgb_files)
        image_ir_files = sorted(image_ir_files)
        mask_files = sorted(mask_files)

        if len(image_rgb_files) != len(mask_files) or len(image_ir_files) != len(mask_files) :
            raise ValueError("Mismatch between number of images and masks.")

        return np.column_stack((image_rgb_files, image_ir_files, mask_files))
    

class CaltechPairDataset(Dataset):

    def __init__(self, images_rgb_dir, images_ir_dir, masks_dir, scale = 1.0, transform=None):
        super().__init__()

        self.images_rgb_dir = images_rgb_dir
        self.images_ir_dir = images_ir_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform = transform
        
        self.dataset = self._load_dataset()

    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        
        image_rgb_file = self.dataset[index, 0]
        image_ir_file = self.dataset[index, 1]
        mask_file = self.dataset[index, 2]

        image_rgb = cv2.cvtColor(cv2.imread(image_rgb_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image_ir = cv2.cvtColor(cv2.imread(image_ir_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)

        image_rgb = self._preprocess(image_rgb, False)
        image_ir = self._preprocess(image_ir, False)
        mask = self._preprocess(mask, True)

        if self.transform is None:
            image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).to(dtype=torch.float32)
            image_ir = torch.from_numpy(image_ir).permute(2, 0, 1).to(dtype=torch.float32)
            mask = torch.from_numpy(mask).to(dtype=torch.int64)
        else:
            aug = self.transform(image=image_rgb, image_extra=image_ir, mask=mask)
            image_rgb = torch.from_numpy(aug["image"]).permute(2, 0, 1).to(dtype=torch.float32)
            image_ir = torch.from_numpy(aug["image_extra"]).permute(2, 0, 1).to(dtype=torch.float32)
            mask = torch.from_numpy(aug["mask"]).to(dtype=torch.int64)

        return image_rgb, image_ir, mask
    
    def _preprocess(self, image, is_mask=False):
        
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)

        if is_mask:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            image = image[:, :, 0].astype(np.int64)
        else:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = (image / 255.0).astype(np.float32)

        return image

    def _load_dataset(self):
            
        image_rgb_files = glob.glob(self.images_rgb_dir + "/" + "*.png")
        image_ir_files = glob.glob(self.images_ir_dir + "/" + "*.png")
        mask_files = glob.glob(self.masks_dir + "/" + "*.png")

        image_rgb_files = sorted(image_rgb_files)
        image_ir_files = sorted(image_ir_files)
        mask_files = sorted(mask_files)

        if not (len(image_rgb_files) == len(image_ir_files) == len(mask_files)):
            raise ValueError("Mismatch between the number of RGB images, IR images, and masks.")

        return np.column_stack((image_rgb_files, image_ir_files, mask_files))


class AntiUAVDetectionDataset(Dataset):
    """
    Dataset class for Anti-UAV-RGBT detection dataset.
    Loads RGB and IR images with bounding box annotations from JSON files.
    """
    
    def __init__(self, dataset_root, split='train', scale=1.0, transform=None, max_objects=10, target_size=(640, 640)):
        super().__init__()
        
        self.dataset_root = dataset_root
        self.split = split
        self.scale = scale
        self.transform = transform
        self.max_objects = max_objects
        self.target_size = target_size
        
        self.dataset = self._load_dataset()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        # Load RGB and IR images
        rgb_image = cv2.cvtColor(cv2.imread(sample['rgb_path'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ir_image = cv2.cvtColor(cv2.imread(sample['ir_path'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        # Preprocess images to fixed size
        rgb_image = self._preprocess_image(rgb_image)
        ir_image = self._preprocess_image(ir_image)
        
        # Get original dimensions (after resize)
        orig_h, orig_w = self.target_size[1], self.target_size[0]
        
        # Process bounding boxes
        bboxes = sample['bboxes']
        scaled_bboxes = []
        
        for bbox in bboxes:
            if len(bbox) == 4:  # [x, y, w, h]
                x, y, w, h = bbox
                # Convert bbox from original image size to target size and normalize
                # Assume original image size is available from cv2.imread
                orig_img = cv2.imread(sample['rgb_path'], cv2.IMREAD_COLOR)
                orig_img_h, orig_img_w = orig_img.shape[:2]
                x_scaled = x * (self.target_size[0] / orig_img_w)
                y_scaled = y * (self.target_size[1] / orig_img_h)
                w_scaled = w * (self.target_size[0] / orig_img_w)
                h_scaled = h * (self.target_size[1] / orig_img_h)
                # Convert to normalized coordinates [0, 1]
                x_norm = x_scaled / self.target_size[0]
                y_norm = y_scaled / self.target_size[1]
                w_norm = w_scaled / self.target_size[0]
                h_norm = h_scaled / self.target_size[1]
                scaled_bboxes.append([x_norm, y_norm, w_norm, h_norm])
        
        # Pad or truncate to max_objects
        while len(scaled_bboxes) < self.max_objects:
            scaled_bboxes.append([0, 0, 0, 0])  # Padding with zeros
        
        if len(scaled_bboxes) > self.max_objects:
            scaled_bboxes = scaled_bboxes[:self.max_objects]
        
        # Create objectness labels (1 if object exists, 0 if padding)
        objectness = [1.0] * min(len(bboxes), self.max_objects) + [0.0] * max(0, self.max_objects - len(bboxes))
        
        # Apply transforms if available
        if self.transform is not None:
            # Note: For detection, we need to be careful with transforms that change bounding boxes
            # For now, we'll apply basic transforms that don't affect bbox coordinates
            aug = self.transform(image=rgb_image, image_extra=ir_image)
            rgb_image = aug["image"]
            ir_image = aug["image_extra"]
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).to(dtype=torch.float32)
        ir_tensor = torch.from_numpy(ir_image).permute(2, 0, 1).to(dtype=torch.float32)
        bbox_tensor = torch.tensor(scaled_bboxes, dtype=torch.float32)
        objectness_tensor = torch.tensor(objectness, dtype=torch.float32)
        
        return rgb_tensor, ir_tensor, bbox_tensor, objectness_tensor
    
    def _preprocess_image(self, image):
        """Preprocess image by resizing to target_size and normalizing"""
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        image = (image / 255.0).astype(np.float32)
        return image
    
    def _load_dataset(self):
        """Load dataset by scanning directories and parsing JSON annotations"""
        dataset = []
        
        # Path to the split directory
        split_dir = os.path.join(self.dataset_root, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Iterate through sequence folders
        for seq_folder in os.listdir(split_dir):
            seq_path = os.path.join(split_dir, seq_folder)
            if not os.path.isdir(seq_path):
                continue
                
            # Check for visible and infrared subdirectories
            visible_dir = os.path.join(seq_path, 'visible')
            infrared_dir = os.path.join(seq_path, 'infrared')
            
            if not (os.path.exists(visible_dir) and os.path.exists(infrared_dir)):
                continue
            
            # Load JSON annotation file (now directly under sequence folder)
            json_path = os.path.join(seq_path, 'infrared.json')
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            # Get image files
            rgb_files = sorted(glob.glob(os.path.join(visible_dir, '*.jpg')))
            ir_files = sorted(glob.glob(os.path.join(infrared_dir, '*.jpg')))
            
            # Ensure we have matching numbers of images
            if len(rgb_files) != len(ir_files):
                print(f"Warning: Mismatch in number of images in {seq_folder}")
                continue
            
            # Get bounding boxes from annotations
            exist_list = annotations.get('exist', [])
            gt_rect_list = annotations.get('gt_rect', [])
            
            # Create samples for each frame
            for i, (rgb_file, ir_file) in enumerate(zip(rgb_files, ir_files)):
                if i < len(exist_list) and exist_list[i] == 1 and i < len(gt_rect_list):
                    # Object exists in this frame
                    bboxes = [gt_rect_list[i]]  # Single object per frame
                else:
                    # No object in this frame
                    bboxes = []
                
                sample = {
                    'rgb_path': rgb_file,
                    'ir_path': ir_file,
                    'bboxes': bboxes
                }
                dataset.append(sample)
        
        print(f"Loaded {len(dataset)} samples for {self.split} split")
        return dataset


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    import random
    np.random.seed(42)
    random.seed(42)

    # Define paths to your data directories
    images_dir = "C:/dev/python/custom_unet/data/split_8/test/color"
    masks_dir = "C:/dev/python/custom_unet/data/split_8/test/annotations"
    images_rgb_dir = "C:/dev/python/custom_unet/data/split_8/test/color"
    images_ir_dir = "C:/dev/python/custom_unet/data/split_8/test/thermal8"

    # Create instances of the datasets
    caltech_dataset = CaltechDataset(images_dir, masks_dir, scale=0.5, transform=settings.transform)
    caltech_pair_dataset = CaltechPairDataset(images_rgb_dir, images_ir_dir, masks_dir, scale=0.5, transform=settings.transform)

    # Test CaltechDataset
    print("Testing CaltechDataset:")
    print(f"Number of samples: {len(caltech_dataset)}")

    # Get a sample from CaltechDataset
    image, mask = caltech_dataset[13]
    print(f"Image shape: {image.shape}")  # Expected: (3, H, W)
    print(f"Mask shape: {mask.shape}")    # Expected: (H, W)

    # Visualize the sample
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C) for visualization
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()

    # Test CaltechPairDataset
    print("\nTesting CaltechPairDataset:")
    print(f"Number of samples: {len(caltech_pair_dataset)}")

    # Get a sample from CaltechPairDataset
    image_rgb, image_ir, mask = caltech_pair_dataset[13]
    print(f"RGB Image shape: {image_rgb.shape}")  # Expected: (3, H, W)
    print(f"IR Image shape: {image_ir.shape}")    # Expected: (3, H, W)
    print(f"Mask shape: {mask.shape}")            # Expected: (H, W)

    # Visualize the sample
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("RGB Image")
    plt.imshow(image_rgb.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C) for visualization
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("IR Image")
    plt.imshow(image_ir.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C) for visualization
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()

    # Test AntiUAVDetectionDataset
    print("\nTesting AntiUAVDetectionDataset:")
    detection_dataset = AntiUAVDetectionDataset("datasets/Anti-UAV-RGBT", split='test', scale=0.5)
    print(f"Number of samples: {len(detection_dataset)}")
    
    if len(detection_dataset) > 0:
        # Get a sample from AntiUAVDetectionDataset
        rgb_tensor, ir_tensor, bbox_tensor, objectness_tensor = detection_dataset[0]
        print(f"RGB Image shape: {rgb_tensor.shape}")
        print(f"IR Image shape: {ir_tensor.shape}")
        print(f"BBox tensor shape: {bbox_tensor.shape}")
        print(f"Objectness tensor shape: {objectness_tensor.shape}")
        print(f"Sample bboxes: {bbox_tensor}")
        print(f"Sample objectness: {objectness_tensor}")

        # Visualize the sample
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("RGB Image")
        plt.imshow(rgb_tensor.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("IR Image")
        plt.imshow(ir_tensor.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Detection Info")
        plt.text(0.1, 0.8, f"BBoxes: {bbox_tensor.shape}", fontsize=12)
        plt.text(0.1, 0.6, f"Objectness: {objectness_tensor.shape}", fontsize=12)
        plt.text(0.1, 0.4, f"Objects: {objectness_tensor.sum().item()}", fontsize=12)
        plt.axis("off")

        plt.show()
        
from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import cv2


class CaltechDataset(Dataset):

    def __init__(self, images_dir, masks_dir, scale = 1.0):
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.scale = scale

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
            image = image / 255.0

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
            image = image / 255.0

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

    def __init__(self, images_rgb_dir, images_ir_dir, masks_dir, scale = 1.0):
        super().__init__()

        self.images_rgb_dir = images_rgb_dir
        self.images_ir_dir = images_ir_dir
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

        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).to(dtype=torch.float32)
        image_ir = torch.from_numpy(image_ir).permute(2, 0, 1).to(dtype=torch.float32)
        mask = torch.from_numpy(mask).to(dtype=torch.int64)

        return image_rgb, image_ir, mask
    
    def _preprocess(self, image, is_mask=False):
        
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)

        if is_mask:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            image = image[:, :, 0].astype(np.int64)
        else:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = image / 255.0

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
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # Define paths to your data directories
    images_dir = "C:/dev/python/custom_unet/data/split_8/test/color"
    masks_dir = "C:/dev/python/custom_unet/data/split_8/test/annotations"
    images_rgb_dir = "C:/dev/python/custom_unet/data/split_8/test/color"
    images_ir_dir = "C:/dev/python/custom_unet/data/split_8/test/thermal8"

    # Create instances of the datasets
    caltech_dataset = CaltechDataset(images_dir, masks_dir, scale=0.5)
    caltech_pair_dataset = CaltechPairDataset(images_rgb_dir, images_ir_dir, masks_dir, scale=0.5)

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

        
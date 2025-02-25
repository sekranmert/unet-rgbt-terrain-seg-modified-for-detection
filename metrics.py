import torch
from torch import Tensor


def mean_iou_score(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    Compute Mean Intersection-over-Union (mIoU) for multi-class segmentation.
    
    Args:
        input (Tensor): Predicted segmentation map (N, C, H, W) with one-hot encoded.
        target (Tensor): One-hot encoded ground truth (N, C, H, W) one-hot encoded..
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: Mean IoU score across all classes.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"
    
    intersection = (input * target).sum(dim=(0, 2, 3))  # Sum over batch, height, width
    union = input.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - intersection  # IoU formula

    iou_per_class = (intersection + epsilon) / (union + epsilon)  # Compute IoU per class

    return iou_per_class.mean()  # Mean IoU across all classes

def iou_score_per_class(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    Compute Intersection-over-Union (IoU) for multi-class segmentation per class.
    
    Args:
        input (Tensor): Predicted segmentation map (N, C, H, W) with one-hot encoding.
        target (Tensor): One-hot encoded ground truth (N, C, H, W).
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: IoU scores per class.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"
    
    intersection = (input * target).sum(dim=(0, 2, 3))  # Sum over batch, height, and width
    union = input.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - intersection  # IoU formula
    
    iou_per_class = (intersection + epsilon) / (union + epsilon)  # Compute IoU per class
    
    return iou_per_class



def weighted_iou_score(input: Tensor, target: Tensor, weight_lut: Tensor, epsilon: float = 1e-6):
    """
    Compute Weighted Intersection-over-Union (wIoU) using per-class weights.

    Args:
        input (Tensor): Predicted segmentation map (N, C, H, W) with one-hot encoded.
        target (Tensor): One-hot encoded ground truth (N, C, H, W) one-hot encoded.
        weight_lut (Tensor): Tensor containing per-class weights.
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: Weighted IoU score.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"
    
    # Ensure weight_lut is on the same device and is a float tensor
    weight_lut = weight_lut.to(input.device).float()

    # Compute intersection and union for each class
    intersection = (input * target).sum(dim=(0, 2, 3))  # Sum over batch, height, width
    union = input.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - intersection  # IoU formula

    # Compute IoU per class
    iou_per_class = (intersection + epsilon) / (union + epsilon)

    # Compute weighted IoU
    weighted_iou = (iou_per_class * weight_lut).sum() / weight_lut.sum()

    return weighted_iou


def dice_coeff(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute Dice Loss for multi-class segmentation.

    Args:
        input (Tensor): Predicted segmentation map (N, C, H, W) with one-hot encoded.
        target (Tensor): One-hot encoded ground truth (N, C, H, W) with one-hot encoded.
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: Dice Loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    # Compute intersection and union
    intersection = (input * target).sum(dim=(2, 3))  # Sum over height and width
    input_sum = input.sum(dim=(2, 3))  # Sum over height and width
    target_sum = target.sum(dim=(2, 3))  # Sum over height and width

    # Compute Dice coefficient per class
    dice_per_class = (2 * intersection + epsilon) / (input_sum + target_sum + epsilon)

    # Average Dice coefficient across classes and batch
    return dice_per_class.mean()

def dice_coeff_per_class(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute Dice Coefficient for multi-class segmentation, averaged over the batch.

    Args:
        input (Tensor): Predicted segmentation map (N, C, H, W) with one-hot encoding.
        target (Tensor): One-hot encoded ground truth (N, C, H, W).
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: Dice Coefficient per class (C,).
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    # Compute intersection and union
    intersection = (input * target).sum(dim=(0, 2, 3))  # Sum over batch, height, and width
    input_sum = input.sum(dim=(0, 2, 3))  # Sum over batch, height, and width
    target_sum = target.sum(dim=(0, 2, 3))  # Sum over batch, height, and width

    # Compute Dice coefficient per class
    dice_per_class = (2 * intersection + epsilon) / (input_sum + target_sum + epsilon)

    return dice_per_class  # Shape: (C,)

def dice_loss(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    return 1 - dice_coeff(input, target)

if __name__ == "__main__":
    
    from utils import *

    # Example inputs (batch size = 1, height = 2, width = 2)
    input_indices = torch.tensor([[[0, 1], [1, 2]]])  # Class indices
    target_indices = torch.tensor([[[2, 0], [0, 1]]])  # Class indices
    num_classes = 3

    # Convert to one-hot encoding
    input_one_hot = mask_to_one_hot(input_indices, num_classes)
    target_one_hot = mask_to_one_hot(target_indices, num_classes)

    # Compute Dice Loss
    loss = dice_loss(input_one_hot, target_one_hot)
    print(f"Dice Loss: {loss.item()}")

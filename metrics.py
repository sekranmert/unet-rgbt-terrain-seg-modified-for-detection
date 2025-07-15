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


# Detection-specific metrics and losses

def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0, reduction: str = 'mean'):
    """
    Smooth L1 Loss for bounding box regression.
    
    Args:
        pred (Tensor): Predicted bounding boxes (N, max_objects, 4)
        target (Tensor): Target bounding boxes (N, max_objects, 4)
        beta (float): Smoothing parameter
        reduction (str): Reduction method ('mean', 'sum', 'none')
    
    Returns:
        Tensor: Smooth L1 loss
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def bbox_iou_loss(pred: Tensor, target: Tensor, reduction: str = 'mean'):
    """
    IoU-based loss for bounding box regression.
    
    Args:
        pred (Tensor): Predicted bounding boxes (N, max_objects, 4) in [x, y, w, h] format
        target (Tensor): Target bounding boxes (N, max_objects, 4) in [x, y, w, h] format
        reduction (str): Reduction method ('mean', 'sum', 'none')
    
    Returns:
        Tensor: IoU loss (1 - IoU)
    """
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    pred_x1 = pred[..., 0]
    pred_y1 = pred[..., 1]
    pred_x2 = pred[..., 0] + pred[..., 2]
    pred_y2 = pred[..., 1] + pred[..., 3]
    
    target_x1 = target[..., 0]
    target_y1 = target[..., 1]
    target_x2 = target[..., 0] + target[..., 2]
    target_y2 = target[..., 1] + target[..., 3]
    
    # Calculate intersection
    x1 = torch.max(pred_x1, target_x1)
    y1 = torch.max(pred_y1, target_y1)
    x2 = torch.min(pred_x2, target_x2)
    y2 = torch.min(pred_y2, target_y2)
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union = pred_area + target_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    loss = 1 - iou
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def objectness_loss(pred: Tensor, target: Tensor, reduction: str = 'mean'):
    """
    Binary cross-entropy loss for objectness classification.
    
    Args:
        pred (Tensor): Predicted objectness scores (N, max_objects)
        target (Tensor): Target objectness labels (N, max_objects)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    
    Returns:
        Tensor: Binary cross-entropy loss
    """
    loss = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def detection_loss(bbox_pred: Tensor, bbox_target: Tensor, 
                  objectness_pred: Tensor, objectness_target: Tensor,
                  bbox_weight: float = 1.0, objectness_weight: float = 1.0,
                  bbox_loss_type: str = 'smooth_l1'):
    """
    Combined detection loss for bounding box regression and objectness classification.
    
    Args:
        bbox_pred (Tensor): Predicted bounding boxes (N, max_objects, 4)
        bbox_target (Tensor): Target bounding boxes (N, max_objects, 4)
        objectness_pred (Tensor): Predicted objectness scores (N, max_objects)
        objectness_target (Tensor): Target objectness labels (N, max_objects)
        bbox_weight (float): Weight for bounding box loss
        objectness_weight (float): Weight for objectness loss
        bbox_loss_type (str): Type of bounding box loss ('smooth_l1' or 'iou')
    
    Returns:
        tuple: (total_loss, bbox_loss, objectness_loss)
    """
    # Only compute bbox loss for objects that exist (objectness_target == 1)
    object_mask = objectness_target > 0.5
    
    if bbox_loss_type == 'smooth_l1':
        bbox_loss = smooth_l1_loss(bbox_pred, bbox_target, reduction='none')
        # Apply mask to only consider existing objects
        bbox_loss = bbox_loss * object_mask.unsqueeze(-1).float()
        bbox_loss = bbox_loss.mean()
    elif bbox_loss_type == 'iou':
        bbox_loss = bbox_iou_loss(bbox_pred, bbox_target, reduction='none')
        bbox_loss = bbox_loss * object_mask.float()
        bbox_loss = bbox_loss.mean()
    else:
        raise ValueError(f"Unknown bbox_loss_type: {bbox_loss_type}")
    
    # Objectness loss
    obj_loss = objectness_loss(objectness_pred, objectness_target)
    
    # Combined loss
    total_loss = bbox_weight * bbox_loss + objectness_weight * obj_loss
    
    return total_loss, bbox_loss, obj_loss


def calculate_map(pred_bboxes: Tensor, pred_objectness: Tensor, 
                 target_bboxes: Tensor, target_objectness: Tensor,
                 iou_threshold: float = 0.5):
    """
    Calculate mean Average Precision (mAP) for detection.
    
    Args:
        pred_bboxes (Tensor): Predicted bounding boxes (N, max_objects, 4)
        pred_objectness (Tensor): Predicted objectness scores (N, max_objects)
        target_bboxes (Tensor): Target bounding boxes (N, max_objects, 4)
        target_objectness (Tensor): Target objectness labels (N, max_objects)
        iou_threshold (float): IoU threshold for true positive
    
    Returns:
        float: mAP score
    """
    # This is a simplified mAP calculation
    # In practice, you might want to use more sophisticated evaluation metrics
    
    batch_size = pred_bboxes.shape[0]
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i in range(batch_size):
        # Get predictions and targets for this sample
        pred_bbox = pred_bboxes[i]  # (max_objects, 4)
        pred_obj = pred_objectness[i]  # (max_objects,)
        target_bbox = target_bboxes[i]  # (max_objects, 4)
        target_obj = target_objectness[i]  # (max_objects,)
        
        # Find predicted objects (objectness > 0.5)
        pred_indices = torch.where(pred_obj > 0.5)[0]
        target_indices = torch.where(target_obj > 0.5)[0]
        
        # Count true positives, false positives, false negatives
        for pred_idx in pred_indices:
            if len(target_indices) > 0:
                # Calculate IoU with all target boxes
                ious = []
                for target_idx in target_indices:
                    iou = calculate_single_iou(pred_bbox[pred_idx], target_bbox[target_idx])
                    ious.append(iou)
                
                max_iou = max(ious) if ious else 0
                if max_iou >= iou_threshold:
                    total_tp += 1
                else:
                    total_fp += 1
            else:
                total_fp += 1
        
        # Count false negatives (targets not detected)
        total_fn += len(target_indices)
    
    # Calculate precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    # F1 score as a proxy for mAP
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


def calculate_single_iou(bbox1: Tensor, bbox2: Tensor):
    """
    Calculate IoU between two bounding boxes in [x, y, w, h] format.
    
    Args:
        bbox1 (Tensor): First bounding box (4,)
        bbox2 (Tensor): Second bounding box (4,)
    
    Returns:
        float: IoU value
    """
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1, w1, h1 = bbox1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = bbox2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


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
    
    # Test detection metrics
    print("\nTesting detection metrics:")
    
    # Sample detection predictions and targets
    batch_size, max_objects = 2, 5
    
    # Predicted bounding boxes (normalized coordinates)
    pred_bboxes = torch.rand(batch_size, max_objects, 4)
    pred_objectness = torch.sigmoid(torch.rand(batch_size, max_objects))
    
    # Target bounding boxes and objectness
    target_bboxes = torch.rand(batch_size, max_objects, 4)
    target_objectness = torch.randint(0, 2, (batch_size, max_objects)).float()
    
    # Test detection loss
    total_loss, bbox_loss, obj_loss = detection_loss(
        pred_bboxes, target_bboxes, pred_objectness, target_objectness
    )
    
    print(f"Detection Loss - Total: {total_loss.item():.4f}, BBox: {bbox_loss.item():.4f}, Objectness: {obj_loss.item():.4f}")
    
    # Test mAP calculation
    map_score = calculate_map(pred_bboxes, pred_objectness, target_bboxes, target_objectness)
    print(f"mAP Score: {map_score:.4f}")

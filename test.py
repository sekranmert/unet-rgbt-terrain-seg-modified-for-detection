import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from metrics import *
from utils import *
from tqdm import tqdm

def test_unet(model, test_dataloader, criterion, device, dataset_weights):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_miou = 0.0
    total_wiou = 0.0
    total_iou_per_class = torch.zeros(1, model.n_classes).to(device=device)
    total_dice_per_class = torch.zeros(1, model.n_classes).to(device=device)

    with torch.no_grad():  # No gradient calculation for validation
        for images, masks in tqdm(test_dataloader, desc="Test"):
            
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            y_pred_logits = model(images)
            y_pred_one_hot = logit_to_one_hot(y_pred_logits, model.n_classes)

            # True mask
            y_true_one_hot = mask_to_one_hot(masks, model.n_classes)

            cls_loss = criterion(y_pred_logits, masks.squeeze(1))
            dice_loss_ = dice_loss(y_pred_one_hot, y_true_one_hot)

            loss = cls_loss + dice_loss_  # Total loss
            
            total_loss += loss.item()
            total_dice += dice_coeff(y_pred_one_hot, y_true_one_hot)
            total_miou += mean_iou_score(y_pred_one_hot, y_true_one_hot)
            total_wiou += weighted_iou_score(y_pred_one_hot, y_true_one_hot, dataset_weights)
            total_iou_per_class += iou_score_per_class(y_pred_one_hot, y_true_one_hot)
            total_dice_per_class += dice_coeff_per_class(y_pred_one_hot, y_true_one_hot)

    model.train()

    # Compute Average Metrics
    avg_loss = total_loss / len(test_dataloader)
    avg_dice = total_dice / len(test_dataloader)
    avg_miou = total_miou / len(test_dataloader)
    avg_wiou = total_wiou / len(test_dataloader)
    total_iou_per_class = total_iou_per_class / len(test_dataloader)
    total_dice_per_class = total_dice_per_class / len(test_dataloader)

    print(f"✔️ Test Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}, mIoU Score: {avg_miou:.4f}, wIoU Score: {avg_wiou:.4f}")

    return avg_loss, avg_dice, avg_miou, avg_wiou, total_iou_per_class, total_dice_per_class  # Return for logging (e.g., WandB)

def test_pairunet(model, test_dataloader, criterion, device, dataset_weights):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_miou = 0.0
    total_wiou = 0.0
    total_iou_per_class = torch.zeros(1, model.n_classes).to(device=device)
    total_dice_per_class = torch.zeros(1, model.n_classes).to(device=device)

    with torch.no_grad():  # No gradient calculation for validation
        for images_rgb, images_ir, masks in tqdm(test_dataloader, desc="Test"):
            
            images_rgb, images_ir, masks = images_rgb.to(device), images_ir.to(device), masks.to(device)

            # Forward pass
            y_pred_logits = model(images_rgb, images_ir)
            y_pred_one_hot = logit_to_one_hot(y_pred_logits, model.n_classes)

            # True mask
            y_true_one_hot = mask_to_one_hot(masks, model.n_classes)

            cls_loss = criterion(y_pred_logits, masks.squeeze(1))
            dice_loss_ = dice_loss(y_pred_one_hot, y_true_one_hot)

            loss = cls_loss + dice_loss_  # Total loss
            
            total_loss += loss.item()
            total_dice += dice_coeff(y_pred_one_hot, y_true_one_hot)
            total_miou += mean_iou_score(y_pred_one_hot, y_true_one_hot)
            total_wiou += weighted_iou_score(y_pred_one_hot, y_true_one_hot, dataset_weights)
            total_iou_per_class += iou_score_per_class(y_pred_one_hot, y_true_one_hot)
            total_dice_per_class += dice_coeff_per_class(y_pred_one_hot, y_true_one_hot)

    model.train()

    # Compute Average Metrics
    avg_loss = total_loss / len(test_dataloader)
    avg_dice = total_dice / len(test_dataloader)
    avg_miou = total_miou / len(test_dataloader)
    avg_wiou = total_wiou / len(test_dataloader)
    total_iou_per_class = total_iou_per_class / len(test_dataloader)
    total_dice_per_class = total_dice_per_class / len(test_dataloader)

    print(f"✔️ Test Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}, mIoU Score: {avg_miou:.4f}, wIoU Score: {avg_wiou:.4f}")

    return avg_loss, avg_dice, avg_miou, avg_wiou, total_iou_per_class, total_dice_per_class  # Return for logging (e.g., WandB)

import argparse
from tqdm import tqdm
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import os

from dataset import *
from metrics import *
from models import *
from utils import *
from val import *
from test import *
import settings

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_model_for_training(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model, optimizer, and scheduler states from a checkpoint file for further training.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load the state into.

    Returns:
        int: The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(checkpoint_path)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state (if provided)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state (if provided)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return the epoch at which the checkpoint was saved
    return checkpoint["epoch"]

def load_dataset(args):

    if args.train_type == "rgb":

        train_set = CaltechDataset("C:/dev/python/custom_unet/data/split_8/train/color",
                                   "C:/dev/python/custom_unet/data/split_8/train/annotations",
                                   args.scale)
        
        val_set = CaltechDataset("C:/dev/python/custom_unet/data/split_8/val/color",
                                 "C:/dev/python/custom_unet/data/split_8/val/annotations",
                                  args.scale)
        
        test_set = CaltechDataset("C:/dev/python/custom_unet/data/split_8/test/color",
                                  "C:/dev/python/custom_unet/data/split_8/test/annotations",
                                  args.scale)

    elif args.train_type == "ir":
        
        train_set = CaltechDataset("C:/dev/python/custom_unet/data/split_8/train/thermal8",
                                   "C:/dev/python/custom_unet/data/split_8/train/annotations",
                                   args.scale)
        
        val_set = CaltechDataset("C:/dev/python/custom_unet/data/split_8/val/thermal8",
                                 "C:/dev/python/custom_unet/data/split_8/val/annotations",
                                 args.scale)
        
        test_set = CaltechDataset("C:/dev/python/custom_unet/data/split_8/test/thermal8",
                                  "C:/dev/python/custom_unet/data/split_8/test/annotations",
                                  args.scale)
    
    elif args.train_type == "rgbcatir":
        
        train_set = CaltechRgbtDataset("C:/dev/python/custom_unet/data/split_8/train",
                                   "C:/dev/python/custom_unet/data/split_8/train/annotations",
                                   args.scale)
        
        val_set = CaltechRgbtDataset("C:/dev/python/custom_unet/data/split_8/val",
                                 "C:/dev/python/custom_unet/data/split_8/val/annotations",
                                 args.scale)
        
        test_set = CaltechRgbtDataset("C:/dev/python/custom_unet/data/split_8/test",
                                  "C:/dev/python/custom_unet/data/split_8/test/annotations",
                                  args.scale)
    
    else:

        train_set = CaltechPairDataset("C:/dev/python/custom_unet/data/split_8/train/color",
                                       "C:/dev/python/custom_unet/data/split_8/train/thermal8",
                                       "C:/dev/python/custom_unet/data/split_8/train/annotations",
                                       args.scale)
        
        val_set = CaltechPairDataset("C:/dev/python/custom_unet/data/split_8/val/color",
                                     "C:/dev/python/custom_unet/data/split_8/val/thermal8",
                                     "C:/dev/python/custom_unet/data/split_8/val/annotations",
                                     args.scale)
        
        test_set = CaltechPairDataset("C:/dev/python/custom_unet/data/split_8/test/color",
                                      "C:/dev/python/custom_unet/data/split_8/test/thermal8",
                                      "C:/dev/python/custom_unet/data/split_8/test/annotations",
                                      args.scale)
        
    
    return train_set, val_set, test_set

def hist_weights_and_gradients(model):

    histograms = {}

    for tag, value in model.named_parameters():

        tag = tag.replace('/', '.')

        if not (torch.isinf(value) | torch.isnan(value)).any():

            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())

        if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():

            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
    
    return histograms

def load_model(args):

    if args.train_type == "rgbir":
        return PairUNet(rgb_channels=args.in_ch_1, ir_channels=args.in_ch_2, n_classes=args.n_classes)     
    else:
        return UNet(n_channels=args.in_ch_1, n_classes=args.n_classes)

def train_unet(model, train_loader, val_loader, optimizer, scheduler, criterion, args):

    device = args.device

    model.to(device)
    model.train()

    best_val_score = -float("inf")  # Initialize with the worst possible value
    best_epoch = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create directory for saving checkpoints

    for epoch in range(1, args.epochs + 1):
        
        # Training phase
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):

            images, masks = images.to(device), masks.to(device)

            # Forward pass
            y_pred_logits = model(images)
            y_pred_one_hot = logit_to_one_hot(y_pred_logits, model.n_classes)

            # True mask
            y_true_one_hot = mask_to_one_hot(masks, model.n_classes)

            cls_loss = criterion(y_pred_logits, masks.squeeze(1))
            dice_loss_ = dice_loss(y_pred_one_hot, y_true_one_hot)

            loss = cls_loss + dice_loss_

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        # Calculate average training loss
        train_loss /= len(train_loader)

        histograms = hist_weights_and_gradients(model)

        # Validation phase
        val_loss, dice_score, miou_score, wiou_score, iou_per_class, dice_per_class = validate_unet(model, val_loader, criterion, device, settings.dataset_weights)
        # Adjust learning rate to maximize 
        scheduler.step(val_loss)
        
        wandb.log(
            {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice_score": dice_score,
                "val_miou_score": miou_score,
                "val_wiou_score": wiou_score,
                'images': wandb.Image(images[0].permute(1, 2, 0).cpu().numpy()),
                'masks': {
                    'true': wandb.Image(mask_to_rgb(masks[0].long().cpu().numpy(), settings.rgb_map)),
                    'pred': wandb.Image(mask_to_rgb(one_hot_to_mask(y_pred_one_hot)[0].long().cpu().numpy(), settings.rgb_map)),
                },
                **histograms
            }
        )

        print("☑️ val_dice_per_class :", list(dice_per_class.squeeze(0).cpu().numpy()))
        print("☑️ val_iou_per_class :", list(iou_per_class.squeeze(0).cpu().numpy()))

        # Save the best model checkpoint
        if dice_score > best_val_score:  # Use the metric you care about (e.g., dice_score, miou_score, etc.)
            best_val_score = dice_score
            best_epoch = epoch

            # Save the model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.exp_name}_best_model_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_dice_score": dice_score,
                    "val_miou_score": miou_score,
                    "val_wiou_score": wiou_score,
                },
                checkpoint_path,
            )
            print(f"Saved best model checkpoint at epoch {epoch} with Dice score: {dice_score:.4f} Name : {args.exp_name}_best_model_epoch_{epoch}.pth")

        # Early stopping: Stop training if no improvement in last `patience` epochs
        if epoch - best_epoch >= args.patience:
            print(f"🛑 Early stopping at Epoch {epoch}: No improvement in the last {args.patience} epochs.")
            break  # Stop training

    print(f"Training complete. Best model saved at epoch {best_epoch} with Dice score: {best_val_score:.4f} Name : {args.exp_name}_best_model_epoch_{best_epoch}.pth")

def train_pairunet(model, train_loader, val_loader, optimizer, scheduler, criterion, args):

    device = args.device

    model.to(device)
    model.train()

    best_val_score = -float("inf")  # Initialize with the worst possible value
    best_epoch = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create directory for saving checkpoints

    for epoch in range(1, args.epochs + 1):
        
        # Training phase
        train_loss = 0
        for images_rgb, images_ir, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):

            images_rgb, images_ir, masks = images_rgb.to(device), images_ir.to(device), masks.to(device)

            # Forward pass
            y_pred_logits = model(images_rgb, images_ir)
            y_pred_one_hot = logit_to_one_hot(y_pred_logits, model.n_classes)

            # True mask
            y_true_one_hot = mask_to_one_hot(masks, model.n_classes)

            cls_loss = criterion(y_pred_logits, masks.squeeze(1))
            dice_loss_ = dice_loss(y_pred_one_hot, y_true_one_hot)

            loss = cls_loss + dice_loss_

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        # Calculate average training loss
        train_loss /= len(train_loader)

        histograms = hist_weights_and_gradients(model)

        # Validation phase
        val_loss, dice_score, miou_score, wiou_score, iou_per_class, dice_per_class = validate_pairunet(model, val_loader, criterion, device, settings.dataset_weights)
        # Adjust learning rate to maximize 
        scheduler.step(val_loss)

        
        wandb.log(
            {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice_score": dice_score,
                "val_miou_score": miou_score,
                "val_wiou_score": wiou_score,
                'image_rgb': wandb.Image(images_rgb[0].permute(1, 2, 0).cpu().numpy()),
                'image_ir': wandb.Image(images_ir[0].permute(1, 2, 0).cpu().numpy()),
                'masks': {
                    'true': wandb.Image(mask_to_rgb(masks[0].long().cpu().numpy(), settings.rgb_map)),
                    'pred': wandb.Image(mask_to_rgb(one_hot_to_mask(y_pred_one_hot)[0].long().cpu().numpy(), settings.rgb_map)),
                },
                **histograms
            }
        )

        print("☑️ val_dice_per_class :", list(dice_per_class.squeeze(0).cpu().numpy()))
        print("☑️ val_iou_per_class :", list(iou_per_class.squeeze(0).cpu().numpy()))

        # Save the best model checkpoint
        if dice_score > best_val_score:  # Use the metric you care about (e.g., dice_score, miou_score, etc.)
            best_val_score = dice_score
            best_epoch = epoch

            # Save the model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.exp_name}_best_model_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_dice_score": dice_score,
                    "val_miou_score": miou_score,
                    "val_wiou_score": wiou_score,
                },
                checkpoint_path,
            )
            print(f"Saved best model checkpoint at epoch {epoch} with Dice score: {dice_score:.4f} Name : {args.exp_name}_best_model_epoch_{epoch}.pth")

        # Early stopping: Stop training if no improvement in last `patience` epochs
        if epoch - best_epoch >= args.patience:
            print(f"🛑 Early stopping at Epoch {epoch}: No improvement in the last {args.patience} epochs.")
            break  # Stop training

    print(f"Training complete. Best model saved at epoch {best_epoch} with Dice score: {best_val_score:.4f} Name : {args.exp_name}_best_model_epoch_{best_epoch}.pth")


def run(args):
    
    wandb.login()
    wandb.init(project=f"{args.project}", config=args, name=f"{args.exp_name}")

    # Set seed for reproducibility
    set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_set, val_set, test_set = load_dataset(args)
    model = load_model(args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, generator=g)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1, generator=g)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1, generator=g)
    
    optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    if args.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Load checkpoint if provided
    # Default starting epoch
    start_epoch = 1 
    if args.from_pth is not None:
        print(f"Loading checkpoint from {args.from_pth}...")
        start_epoch = load_model_for_training(args.from_pth, model, optimizer, scheduler) + 1
        print(f"Resuming training from epoch {start_epoch}")

    if args.train_type == "rgbir":
        train_pairunet(model, train_loader, val_loader, optimizer, scheduler, criterion, args)

        # Test phase
        test_loss, dice_score, miou_score, wiou_score, iou_per_class, dice_per_class = test_pairunet(model, test_loader, criterion, args.device, settings.dataset_weights)
        
    else:
        train_unet(model, train_loader, val_loader, optimizer, scheduler, criterion, args)

        # Test phase
        test_loss, dice_score, miou_score, wiou_score, iou_per_class, dice_per_class = test_unet(model, test_loader, criterion, args.device, settings.dataset_weights)
    

    print("☑️ test_dice_per_class :", list(dice_per_class.squeeze(0).cpu().numpy()))
    print("☑️ test_iou_per_class :", list(iou_per_class.squeeze(0).cpu().numpy()))

    wandb.log(
        {
            "test_loss": test_loss,
            "test_dice_score": dice_score,
            "test_miou_score": miou_score,
            "test_wiou_score": wiou_score
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='UNet trainer')

    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momentum')
    parser.add_argument('--from_pth', type=str, default=None, help='Checkpoint file')
    parser.add_argument('--scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--n_classes', type=int, default=12, help='Number of classes')
    parser.add_argument('--exp_name', type=str, default="Run1", help='Experiment name')
    parser.add_argument('--train_type',  type=str, default="rgb", help='rgb, ir, rgbcatir or rgbir')
    parser.add_argument('--device',  type=str, default="cuda", help='Device')
    parser.add_argument('--project',  type=str, default="Test1", help='Project name')
    parser.add_argument('--seed',  type=int, default=42, help='Seed number')
    parser.add_argument('--patience',  type=int, default=30, help='Early stop patience')
    parser.add_argument('--in_ch_1',  type=int, default=3, help='Input channel size (ex. rgbcatir = 4)')
    parser.add_argument('--in_ch_2',  type=int, default=3, help='Input channel size')

    args = parser.parse_args()
    run(args)

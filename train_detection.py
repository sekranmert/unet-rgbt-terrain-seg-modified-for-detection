import argparse
from tqdm import tqdm
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

import os
import numpy as np

from dataset import AntiUAVDetectionDataset, YoloSplitDetectionDataset, YoloSplitPairDetectionDataset
from metrics import detection_loss, calculate_map
from models import PairUNetWithDetection, UNetWithDetection
from utils import *
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
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_model_for_training(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model, optimizer, and scheduler states from a checkpoint file for further training.
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

def load_detection_dataset(args):
    """
    Load detection datasets for training, validation, and testing.
    """
    # Convert img_size to tuple
    target_size = tuple(args.img_size)
    if getattr(args, 'dataset_type', 'antiuav') == 'yolo_split':
        root = getattr(args, 'yolo_split_root', 'datasets/Anti-UAV-SDO-YOLO-Split')
        if args.modality == 'both':
            train_set = YoloSplitPairDetectionDataset(
                root=root,
                split='train',
                max_objects=args.max_objects,
                target_size=target_size,
                transform=settings.transform if args.aug else None
            )
            val_set = YoloSplitPairDetectionDataset(
                root=root,
                split='val',
                max_objects=args.max_objects,
                target_size=target_size,
                transform=None
            )
            test_set = YoloSplitPairDetectionDataset(
                root=root,
                split='test',
                max_objects=args.max_objects,
                target_size=target_size,
                transform=None
            )
        else:
            train_set = YoloSplitDetectionDataset(
                root=root,
                split='train',
                modality=args.modality,
                max_objects=args.max_objects,
                target_size=target_size,
                transform=settings.transform if args.aug else None
            )
            val_set = YoloSplitDetectionDataset(
                root=root,
                split='val',
                modality=args.modality,
                max_objects=args.max_objects,
                target_size=target_size,
                transform=None
            )
            test_set = YoloSplitDetectionDataset(
                root=root,
                split='test',
                modality=args.modality,
                max_objects=args.max_objects,
                target_size=target_size,
                transform=None
            )
    else:
        # Use original AntiUAVDetectionDataset
        train_set = AntiUAVDetectionDataset(
            dataset_root=args.dataset_root,
            split='train',
            scale=args.scale,
            transform=settings.transform if args.aug else None,
            max_objects=args.max_objects,
            target_size=target_size
        )
        val_set = AntiUAVDetectionDataset(
            dataset_root=args.dataset_root,
            split='val',
            scale=args.scale,
            transform=None,  # No augmentation for validation
            max_objects=args.max_objects,
            target_size=target_size
        )
        test_set = AntiUAVDetectionDataset(
            dataset_root=args.dataset_root,
            split='test',
            scale=args.scale,
            transform=None,  # No augmentation for testing
            max_objects=args.max_objects,
            target_size=target_size
        )
    return train_set, val_set, test_set

def hist_weights_and_gradients(model):
    return {}  # No-op when wandb is removed

def load_detection_model(args):
    """Load the appropriate detection model based on input type."""
    if getattr(args, 'modality', 'visible') == 'both':
        return PairUNetWithDetection(
            rgb_channels=args.in_ch_1,
            ir_channels=args.in_ch_2,
            n_classes=args.n_classes,
            max_objects=args.max_objects
        )
    elif args.train_type == "rgbir":
        return PairUNetWithDetection(
            rgb_channels=args.in_ch_1, 
            ir_channels=args.in_ch_2, 
            n_classes=args.n_classes,
            max_objects=args.max_objects
        )
    else:
        return UNetWithDetection(
            n_channels=args.in_ch_1, 
            n_classes=args.n_classes,
            max_objects=args.max_objects
        )

def train_detection_epoch(model, train_loader, optimizer, criterion, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_bbox_loss = 0.0
    total_obj_loss = 0.0
    total_map = 0.0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        if args.train_type == "rgbir":
            rgb_images, ir_images, bbox_targets, objectness_targets = batch
            rgb_images = rgb_images.to(device)
            ir_images = ir_images.to(device)
            
            # Forward pass
            seg_logits, bbox_pred, objectness_pred = model(rgb_images, ir_images)
        else:
            images, bbox_targets, objectness_targets = batch
            images = images.to(device)
            
            # Forward pass
            seg_logits, bbox_pred, objectness_pred = model(images)
        
        bbox_targets = bbox_targets.to(device)
        objectness_targets = objectness_targets.to(device)
        
        # Calculate detection loss
        det_loss, bbox_loss, obj_loss = detection_loss(
            bbox_pred, bbox_targets, 
            objectness_pred, objectness_targets,
            bbox_weight=args.bbox_weight,
            objectness_weight=args.objectness_weight,
            bbox_loss_type=args.bbox_loss_type
        )
        
        # Calculate mAP for monitoring
        map_score = calculate_map(bbox_pred, objectness_pred, bbox_targets, objectness_targets)
        
        # Backward pass
        optimizer.zero_grad()
        det_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += det_loss.item()
        total_bbox_loss += bbox_loss.item()
        total_obj_loss += obj_loss.item()
        total_map += map_score
    
    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_bbox_loss = total_bbox_loss / len(train_loader)
    avg_obj_loss = total_obj_loss / len(train_loader)
    avg_map = total_map / len(train_loader)
    
    return avg_loss, avg_bbox_loss, avg_obj_loss, avg_map

def validate_detection(model, val_loader, criterion, device, args):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_bbox_loss = 0.0
    total_obj_loss = 0.0
    total_map = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if args.train_type == "rgbir":
                rgb_images, ir_images, bbox_targets, objectness_targets = batch
                rgb_images = rgb_images.to(device)
                ir_images = ir_images.to(device)
                
                # Forward pass
                seg_logits, bbox_pred, objectness_pred = model(rgb_images, ir_images)
            else:
                images, bbox_targets, objectness_targets = batch
                images = images.to(device)
                
                # Forward pass
                seg_logits, bbox_pred, objectness_pred = model(images)
            
            bbox_targets = bbox_targets.to(device)
            objectness_targets = objectness_targets.to(device)
            
            # Calculate detection loss
            det_loss, bbox_loss, obj_loss = detection_loss(
                bbox_pred, bbox_targets, 
                objectness_pred, objectness_targets,
                bbox_weight=args.bbox_weight,
                objectness_weight=args.objectness_weight,
                bbox_loss_type=args.bbox_loss_type
            )
            
            # Calculate mAP
            map_score = calculate_map(bbox_pred, objectness_pred, bbox_targets, objectness_targets)
            
            # Accumulate losses
            total_loss += det_loss.item()
            total_bbox_loss += bbox_loss.item()
            total_obj_loss += obj_loss.item()
            total_map += map_score
    
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_bbox_loss = total_bbox_loss / len(val_loader)
    avg_obj_loss = total_obj_loss / len(val_loader)
    avg_map = total_map / len(val_loader)
    
    return avg_loss, avg_bbox_loss, avg_obj_loss, avg_map

def train_detection_model(model, train_loader, val_loader, optimizer, scheduler, criterion, args):
    """Main training loop for detection."""
    device = args.device
    model.to(device)
    
    best_val_map = -float("inf")
    best_epoch = 0
    checkpoint_dir = "checkpoints_detection"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Reset peak memory stats at the start of each epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Training phase
        train_loss, train_bbox_loss, train_obj_loss, train_map = train_detection_epoch(
            model, train_loader, optimizer, criterion, device, args
        )
        
        # Validation phase
        val_loss, val_bbox_loss, val_obj_loss, val_map = validate_detection(
            model, val_loader, criterion, device, args
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train - Loss: {train_loss:.4f}, BBox: {train_bbox_loss:.4f}, Obj: {train_obj_loss:.4f}, mAP: {train_map:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, BBox: {val_bbox_loss:.4f}, Obj: {val_obj_loss:.4f}, mAP: {val_map:.4f}")
        
        # Print GPU VRAM usage statistics
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / 1024 / 1024
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[GPU] Current VRAM usage: {current_mem:.2f} MB | Peak VRAM usage: {peak_mem:.2f} MB")
        
        # Save best model
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.exp_name}_best_detection_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_map": val_map,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"Saved best model at epoch {epoch} with mAP: {val_map:.4f}")
        
        # Early stopping
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Training complete. Best model saved at epoch {best_epoch} with mAP: {best_val_map:.4f}")
    return best_epoch, best_val_map

def test_detection_model(model, test_loader, criterion, device, args):
    """Test the trained detection model."""
    model.eval()
    total_loss = 0.0
    total_bbox_loss = 0.0
    total_obj_loss = 0.0
    total_map = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if args.train_type == "rgbir":
                rgb_images, ir_images, bbox_targets, objectness_targets = batch
                rgb_images = rgb_images.to(device)
                ir_images = ir_images.to(device)
                
                # Forward pass
                seg_logits, bbox_pred, objectness_pred = model(rgb_images, ir_images)
            else:
                images, bbox_targets, objectness_targets = batch
                images = images.to(device)
                
                # Forward pass
                seg_logits, bbox_pred, objectness_pred = model(images)
            
            bbox_targets = bbox_targets.to(device)
            objectness_targets = objectness_targets.to(device)
            
            # Calculate detection loss
            det_loss, bbox_loss, obj_loss = detection_loss(
                bbox_pred, bbox_targets, 
                objectness_pred, objectness_targets,
                bbox_weight=args.bbox_weight,
                objectness_weight=args.objectness_weight,
                bbox_loss_type=args.bbox_loss_type
            )
            
            # Calculate mAP
            map_score = calculate_map(bbox_pred, objectness_pred, bbox_targets, objectness_targets)
            
            # Accumulate losses
            total_loss += det_loss.item()
            total_bbox_loss += bbox_loss.item()
            total_obj_loss += obj_loss.item()
            total_map += map_score
    
    # Calculate averages
    avg_loss = total_loss / len(test_loader)
    avg_bbox_loss = total_bbox_loss / len(test_loader)
    avg_obj_loss = total_obj_loss / len(test_loader)
    avg_map = total_map / len(test_loader)
    
    print(f"Test Results - Loss: {avg_loss:.4f}, BBox: {avg_bbox_loss:.4f}, Obj: {avg_obj_loss:.4f}, mAP: {avg_map:.4f}")
    
    return avg_loss, avg_bbox_loss, avg_obj_loss, avg_map

def run(args):
    """Main function to run detection training."""
    # No wandb
    # Set seed for reproducibility
    set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Load datasets
    train_set, val_set, test_set = load_detection_dataset(args)
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        generator=g, 
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        generator=g, 
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        generator=g, 
        worker_init_fn=seed_worker
    )
    
    # Load model
    model = load_detection_model(args)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=5
    )
    
    # Load checkpoint if provided
    start_epoch = 1
    if args.from_pth is not None:
        print(f"Loading checkpoint from {args.from_pth}...")
        start_epoch = load_model_for_training(args.from_pth, model, optimizer, scheduler) + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training
    best_epoch, best_val_map = train_detection_model(
        model, train_loader, val_loader, optimizer, scheduler, None, args
    )
    
    # Testing
    test_loss, test_bbox_loss, test_obj_loss, test_map = test_detection_model(
        model, test_loader, None, args.device, args
    )
    
    print(f"Final Test Results - mAP: {test_map:.4f}")
    print(f"Best Validation mAP: {best_val_map:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection trainer for Anti-UAV-RGBT')
    
    # Dataset parameters
    parser.add_argument('--dataset_root', type=str, default='datasets/Anti-UAV-RGBT', 
                       help='Path to Anti-UAV-RGBT dataset (for antiuav dataset_type)')
    parser.add_argument('--yolo_split_root', type=str, default='datasets/Anti-UAV-SDO-YOLO-Split',
                       help='Path to YOLO-format split dataset (for yolo_split dataset_type)')
    parser.add_argument('--max_objects', type=int, default=10, 
                       help='Maximum number of objects per image')
    parser.add_argument('--img_size', type=int, nargs=2, default=[640, 640], 
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Target image size (width height) for resizing')
    parser.add_argument('--dataset_type', type=str, default='antiuav', choices=['antiuav', 'yolo_split'],
                       help='Dataset type: antiuav (default) or yolo_split (for YOLO-format split dataset)')
    parser.add_argument('--modality', type=str, default='visible', choices=['visible', 'infrared', 'both'],
                       help='Modality for yolo_split dataset: visible, infrared, or both (for paired training)')
    
    # Model parameters
    parser.add_argument('--train_type', type=str, default='rgbir', 
                       help='rgb, ir, or rgbir')
    parser.add_argument('--in_ch_1', type=int, default=3, 
                       help='Input channels for first modality')
    parser.add_argument('--in_ch_2', type=int, default=3, 
                       help='Input channels for second modality (for rgbir)')
    parser.add_argument('--n_classes', type=int, default=1, 
                       help='Number of classes (1 for detection)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay')
    parser.add_argument('--scale', type=float, default=0.5, 
                       help='Image scaling factor')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loader workers')
    
    # Loss parameters
    parser.add_argument('--bbox_weight', type=float, default=1.0, 
                       help='Weight for bounding box loss')
    parser.add_argument('--objectness_weight', type=float, default=1.0, 
                       help='Weight for objectness loss')
    parser.add_argument('--bbox_loss_type', type=str, default='smooth_l1', 
                       choices=['smooth_l1', 'iou'], 
                       help='Type of bounding box loss')
    
    # Other parameters
    parser.add_argument('--exp_name', type=str, default="Detection_Run1", 
                       help='Experiment name')
    parser.add_argument('--device', type=str, default="cuda", 
                       help='Device to use')
    parser.add_argument('--project', type=str, default="AntiUAV_Detection", 
                       help='WandB project name')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--patience', type=int, default=20, 
                       help='Early stopping patience')
    parser.add_argument('--from_pth', type=str, default=None, 
                       help='Checkpoint file to resume from')
    parser.add_argument('--aug', action='store_true', 
                       help='Enable data augmentation')
    
    args = parser.parse_args()
    run(args) 
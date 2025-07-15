import torch
import cv2
import numpy as np
import argparse
import os
from models import PairUNetWithDetection, UNetWithDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import AntiUAVDetectionDataset

def load_model(checkpoint_path, model_type='rgbir', device='cuda'):
    """
    Load a trained detection model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        model_type (str): Type of model ('rgbir' or 'rgb')
        device (str): Device to load model on
    
    Returns:
        model: Loaded model
    """
    if model_type == 'rgbir':
        model = PairUNetWithDetection(
            rgb_channels=3, 
            ir_channels=3, 
            n_classes=1, 
            max_objects=10
        )
    else:
        model = UNetWithDetection(
            n_channels=3, 
            n_classes=1, 
            max_objects=10
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path, target_size=(640, 640)):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size (width, height)
    
    Returns:
        tensor: Preprocessed image tensor
    """
    # Load image
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Resize image to target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image = (image / 255.0).astype(np.float32)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_shape

def detect_objects(model, rgb_image, ir_image=None, device='cuda', confidence_threshold=0.5):
    """
    Perform object detection on input images.
    
    Args:
        model: Trained detection model
        rgb_image (tensor): RGB image tensor
        ir_image (tensor): IR image tensor (optional)
        device (str): Device to run inference on
        confidence_threshold (float): Threshold for objectness score
    
    Returns:
        list: List of detected bounding boxes and scores
    """
    model.eval()
    
    with torch.no_grad():
        if ir_image is not None:
            # RGB-IR model
            seg_logits, bbox_pred, objectness_pred = model(rgb_image.to(device), ir_image.to(device))
        else:
            # Single modality model
            seg_logits, bbox_pred, objectness_pred = model(rgb_image.to(device))
        
        # Get predictions
        bbox_pred = bbox_pred.cpu().numpy()[0]  # (max_objects, 4)
        objectness_pred = objectness_pred.cpu().numpy()[0]  # (max_objects,)
        
        # Filter by confidence threshold
        detections = []
        for i in range(len(objectness_pred)):
            if objectness_pred[i] > confidence_threshold:
                bbox = bbox_pred[i]  # [x, y, w, h] in normalized coordinates
                detections.append({
                    'bbox': bbox,
                    'confidence': objectness_pred[i]
                })
        
        return detections

def visualize_detections(image, detections, original_shape, output_path=None):
    """
    Visualize detection results on the image.
    
    Args:
        image (numpy.ndarray): Original image
        detections (list): List of detections
        original_shape (tuple): Original image shape (H, W)
        output_path (str): Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image
    ax.imshow(image)
    
    # Draw bounding boxes
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # Convert normalized coordinates to pixel coordinates
        x, y, w, h = bbox
        x_pixel = x * original_shape[1]
        y_pixel = y * original_shape[0]
        w_pixel = w * original_shape[1]
        h_pixel = h * original_shape[0]
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_pixel, y_pixel), w_pixel, h_pixel,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add confidence score
        ax.text(x_pixel, y_pixel - 5, f'{confidence:.2f}', 
                color='red', fontsize=12, weight='bold')
    
    ax.set_title(f'Detected Objects: {len(detections)}')
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def run_inference_on_dataset(model, dataset_root, split='test', device='cuda', 
                           confidence_threshold=0.5, num_samples=5, img_size=(640, 640)):
    """
    Run inference on a few samples from the dataset.
    
    Args:
        model: Trained detection model
        dataset_root (str): Path to dataset
        split (str): Dataset split to use
        device (str): Device to run inference on
        confidence_threshold (float): Confidence threshold
        num_samples (int): Number of samples to test
        img_size (tuple): Target image size (width, height)
    """
    # Create dataset
    dataset = AntiUAVDetectionDataset(
        dataset_root=dataset_root,
        split=split,
        scale=0.5,
        transform=None,
        max_objects=10,
        target_size=img_size
    )
    
    print(f"Running inference on {num_samples} samples from {split} split...")
    
    for i in range(min(num_samples, len(dataset))):
        # Get sample
        rgb_tensor, ir_tensor, bbox_targets, objectness_targets = dataset[i]
        
        # Run detection
        detections = detect_objects(
            model, 
            rgb_tensor.unsqueeze(0), 
            ir_tensor.unsqueeze(0), 
            device, 
            confidence_threshold
        )
        
        # Get original image for visualization
        rgb_image = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Visualize
        print(f"Sample {i+1}: Found {len(detections)} objects")
        visualize_detections(
            rgb_image, 
            detections, 
            rgb_image.shape[:2],
            output_path=f"detection_result_{i+1}.png"
        )

def main():
    parser = argparse.ArgumentParser(description='Detection inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='rgbir',
                       choices=['rgbir', 'rgb'],
                       help='Type of model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--dataset_root', type=str, default='datasets/Anti-UAV-RGBT',
                       help='Path to dataset for testing')
    parser.add_argument('--test_on_dataset', action='store_true',
                       help='Test on dataset samples')
    parser.add_argument('--rgb_image', type=str, default=None,
                       help='Path to RGB image for single inference')
    parser.add_argument('--ir_image', type=str, default=None,
                       help='Path to IR image for single inference')
    parser.add_argument('--img_size', type=int, nargs=2, default=[640, 640], 
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Target image size (width height) for resizing')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model_type, args.device)
    print("Model loaded successfully!")
    
    if args.test_on_dataset:
        # Test on dataset samples
        run_inference_on_dataset(
            model, 
            args.dataset_root, 
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            img_size=tuple(args.img_size)
        )
    
    elif args.rgb_image:
        # Single image inference
        print(f"Running inference on {args.rgb_image}...")
        
        # Load and preprocess RGB image
        rgb_tensor, rgb_shape = preprocess_image(args.rgb_image, tuple(args.img_size))
        
        if args.ir_image and args.model_type == 'rgbir':
            # Load and preprocess IR image
            ir_tensor, ir_shape = preprocess_image(args.ir_image, tuple(args.img_size))
            
            # Run detection
            detections = detect_objects(
                model, 
                rgb_tensor, 
                ir_tensor, 
                args.device, 
                args.confidence_threshold
            )
        else:
            # Single modality detection
            detections = detect_objects(
                model, 
                rgb_tensor, 
                device=args.device, 
                confidence_threshold=args.confidence_threshold
            )
        
        # Load original image for visualization
        rgb_image = cv2.cvtColor(cv2.imread(args.rgb_image), cv2.COLOR_BGR2RGB)
        
        # Visualize results
        print(f"Found {len(detections)} objects")
        visualize_detections(
            rgb_image, 
            detections, 
            rgb_shape,
            output_path="single_inference_result.png"
        )
    
    else:
        print("Please specify either --test_on_dataset or --rgb_image")

if __name__ == "__main__":
    main() 
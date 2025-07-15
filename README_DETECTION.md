# U-Net RGB-T Detection for Small Object Detection

This repository has been modified to support **small object detection** using the Anti-UAV-RGBT dataset. The original U-Net segmentation backbone has been enhanced with detection heads to predict bounding boxes and objectness scores.

## 🎯 What's New

### Detection Capabilities
- **Bounding Box Prediction**: Predicts [x, y, w, h] coordinates for small objects
- **Objectness Classification**: Determines whether objects exist in the image
- **Multi-Modal Support**: Works with both RGB and RGB-Thermal (RGB-T) data
- **Single-Class Detection**: Optimized for detecting small UAV objects

### New Models
- `UNetWithDetection`: Single modality U-Net with detection head
- `PairUNetWithDetection`: RGB-T dual modality U-Net with detection head
- `DetectionHead`: Modular detection head for bounding box and objectness prediction

### New Dataset Class
- `AntiUAVDetectionDataset`: Handles Anti-UAV-RGBT dataset with JSON annotations

## 📁 Dataset Structure

The code expects the Anti-UAV-RGBT dataset in the following structure:

```
datasets/Anti-UAV-RGBT/
├── train/
│   ├── sequence_1/
│   │   ├── visible/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── infrared/
│   │       ├── 000001.jpg
│   │       ├── 000002.jpg
│   │       ├── infrared.json
│   │       └── ...
│   └── sequence_2/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### JSON Annotation Format
Each sequence folder contains an `infrared.json` file with:
```json
{
  "exist": [1, 1, 0, 1, ...],  // Object existence per frame
  "gt_rect": [
    [x, y, w, h],  // Bounding box for frame 0
    [x, y, w, h],  // Bounding box for frame 1
    ...
  ]
}
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision opencv-python numpy matplotlib wandb tqdm albumentations
```

### 2. Prepare Dataset
Place your Anti-UAV-RGBT dataset in the `datasets/` folder with the structure shown above.

### 3. Train Detection Model

#### RGB-Thermal Detection (Recommended)
```bash
python train_detection.py \
    --dataset_root datasets/Anti-UAV-RGBT \
    --train_type rgbir \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --exp_name "RGBT_Detection" \
    --img_size 640 640 \
    --aug
```

#### RGB-Only Detection
```bash
python train_detection.py \
    --dataset_root datasets/Anti-UAV-RGBT \
    --train_type rgb \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --exp_name "RGB_Detection" \
    --img_size 640 640 \
    --aug
```

### 4. Run Inference

#### Test on Dataset Samples
```bash
python inference_detection.py \
    --checkpoint checkpoints_detection/RGBT_Detection_best_detection_epoch_X.pth \
    --model_type rgbir \
    --test_on_dataset \
    --img_size 640 640 \
    --confidence_threshold 0.5
```

#### Single Image Inference
```bash
python inference_detection.py \
    --checkpoint checkpoints_detection/RGBT_Detection_best_detection_epoch_X.pth \
    --model_type rgbir \
    --rgb_image path/to/rgb_image.jpg \
    --ir_image path/to/ir_image.jpg \
    --img_size 640 640 \
    --confidence_threshold 0.5
```

## 📊 Model Architecture

### Detection Head
The detection head is attached to the bottleneck of the U-Net:

```
Input Image(s) → U-Net Encoder → Bottleneck Features → Detection Head
                                    ↓
                              Segmentation Decoder
```

### Detection Head Components
1. **Global Average Pooling**: Reduces spatial dimensions to feature vector
2. **Feature Extractor**: MLP layers for feature processing
3. **BBox Regression Head**: Predicts [x, y, w, h] coordinates
4. **Objectness Head**: Predicts object existence probability

## 🎛️ Training Parameters

### Key Parameters
- `--img_size`: Target image size for resizing (width height, default: 640 640)
- `--max_objects`: Maximum objects per image (default: 10)
- `--bbox_weight`: Weight for bounding box loss (default: 1.0)
- `--objectness_weight`: Weight for objectness loss (default: 1.0)
- `--bbox_loss_type`: Loss type for bbox regression (`smooth_l1` or `iou`)
- `--confidence_threshold`: Threshold for detection confidence (default: 0.5)

### Loss Function
The detection loss combines:
- **Bounding Box Loss**: Smooth L1 or IoU loss for coordinate regression
- **Objectness Loss**: Binary cross-entropy for object existence

```
Total Loss = bbox_weight × BBox_Loss + objectness_weight × Objectness_Loss
```

## 📈 Evaluation Metrics

### Detection Metrics
- **mAP (mean Average Precision)**: Primary evaluation metric
- **Precision/Recall**: Detection accuracy measures
- **IoU Loss**: Bounding box regression quality
- **Objectness Accuracy**: Object existence prediction accuracy

## 🔧 Advanced Usage

### Custom Dataset
To use with a different dataset, modify the `AntiUAVDetectionDataset` class:

```python
class CustomDetectionDataset(Dataset):
    def __init__(self, dataset_root, split='train', scale=1.0, max_objects=10):
        # Implement your dataset loading logic
        pass
    
    def __getitem__(self, index):
        # Return: rgb_tensor, ir_tensor, bbox_targets, objectness_targets
        pass
```

### Model Modifications
To modify the detection head architecture:

```python
class CustomDetectionHead(nn.Module):
    def __init__(self, in_channels, max_objects=10):
        super().__init__()
        # Custom detection head implementation
        pass
    
    def forward(self, x):
        # Return: bbox_pred, objectness_pred
        pass
```

### Multi-Class Detection
To extend to multiple classes, modify the detection head:

```python
# In DetectionHead class
self.class_head = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, max_objects * num_classes),
    nn.Softmax(dim=-1)
)
```

## 📁 File Structure

```
unet-rgbt-terrain-seg-modified-for-detection/
├── models.py                 # U-Net models with detection heads
├── dataset.py               # Dataset classes including AntiUAVDetectionDataset
├── metrics.py               # Detection loss functions and metrics
├── train_detection.py       # Training script for detection
├── inference_detection.py   # Inference script
├── settings.py              # Configuration and transforms
├── utils.py                 # Utility functions
└── README_DETECTION.md      # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Reduce image scale: `--scale 0.25`

2. **Dataset Loading Errors**
   - Check dataset structure matches expected format
   - Verify JSON annotation files are valid
   - Ensure image files exist and are readable

3. **Poor Detection Performance**
   - Increase training epochs
   - Adjust learning rate
   - Try different loss weights
   - Enable data augmentation

### Performance Tips

1. **For Small Objects**
   - Use higher resolution (larger scale factor)
   - Increase max_objects if needed
   - Use RGB-T model for better performance

2. **Training Stability**
   - Start with lower learning rate
   - Use gradient clipping if needed
   - Monitor loss curves in WandB

## 📚 References

- Original U-Net Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Anti-UAV-RGBT Dataset: [Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking](https://arxiv.org/abs/2101.08466)
- Detection Loss Functions: Based on common object detection practices

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is based on the original U-Net implementation and modified for detection tasks. 
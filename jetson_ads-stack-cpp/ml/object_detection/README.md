# YOLOv1 with U-Net Integration

This project implements a hybrid model combining the YOLOv1 (You Only Look Once) object detection framework with a U-Net architecture for simultaneous object detection and semantic segmentation. The implementation is optimized for efficient training and inference, with considerations for deployment on resource-constrained devices like the Jetson Nano.

## Project Overview

The model leverages a U-Net backbone for feature extraction, followed by a YOLOv1-inspired detection head to predict bounding boxes, object confidence scores, and class probabilities. The architecture is designed to:
- Perform object detection using a grid-based approach (default: 7x7 grid, 2 bounding boxes per cell, 20 classes).
- Support semantic segmentation through the U-Net's upsampling path.
- Utilize pre-trained weights for transfer learning, with the detection head fine-tuned for specific tasks.
- Optimize computational efficiency through reduced channel counts and adaptive pooling.

The codebase includes training scripts, loss functions, and utilities for evaluating performance using metrics like mean Average Precision (mAP).

## Repository Structure

- **`model_object.py`**: Defines the model architecture, including:
  - `DoubleConv`: A module with two convolutional layers, batch normalization, ReLU, and dropout.
  - `UNET`: The main model combining U-Net and YOLOv1 detection head.
  - Utility functions for weight initialization, freezing layers, and loading pre-trained weights.
- **`train.py`**: Implements the training pipeline, including:
  - Data loading with custom transformations for the VOC dataset.
  - Training loop with mixed-precision training (AMP).
  - Evaluation of mAP on the test set.
  - Checkpoint saving when mAP exceeds 0.9.
- **`loss.py`**: Contains the `YoloLoss` class, which computes the YOLOv1 loss, including:
  - Coordinate loss for bounding box predictions.
  - Object and no-object confidence losses.
  - Classification loss for class probabilities.
- **`utils.py`**: Provides utility functions for:
  - Intersection over Union (IoU) calculation.
  - Non-Maximum Suppression (NMS) for post-processing predictions.
  - Mean Average Precision (mAP) computation.
  - Conversion of cell-based predictions to bounding box coordinates.
  - Checkpoint saving.
- **`dataset.py`**: (Assumed) Defines the `ObjDataset` class for loading and preprocessing the dataset (e.g., VOC format).

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**:
   Ensure Python 3.8+ is installed, then install the required packages:
   ```bash
   pip install torch torchvision tqdm
   ```

3. **Prepare the dataset**:
   - The project assumes a dataset in VOC format with images in `data/images` and labels in `data/labels`.
   - Update `IMG_DIR`, `LABEL_DIR`, and CSV paths in `train_detection_only.py` as needed.

4. **Download pre-trained weights**:
   - Place pre-trained weights in the project directory and update `PRETRAINED_PATH` in `train.py`.
   - The model supports loading `.pth` or `.pth.tar` files, ignoring incompatible weights (e.g., detection head).

## Usage

### Training
To train the model, run:
```bash
python train.py
```

**Hyperparameters** (configurable in `train_detection_only.py`):
- `LEARNING_RATE`: 1e-4
- `BATCH_SIZE`: 16
- `EPOCHS`: 50
- `NUM_WORKERS`: 4
- `SPLIT_SIZE`: 7 (grid size)
- `NUM_BOXES`: 2 (bounding boxes per cell)
- `NUM_CLASSES`: 20

The script:
- Loads pre-trained weights and freezes all layers except the detection head and grid adjustment.
- Trains the model using the Adam optimizer and YOLO loss.
- Evaluates mAP on the test set after each epoch.
- Saves a checkpoint (`unet_yolo.pth.tar`) if mAP exceeds 0.9.


## Optimization for Jetson Nano
The detection head is designed for efficiency:
- Reduced channel counts (512 → 128 → 64) to lower memory usage.
- Uses `AdaptiveAvgPool2d` to enforce a fixed S×S grid output.
- Employs `LeakyReLU` and batch normalization for stable training.
- Dropout rates are adjusted based on layer depth (0.1 for early layers, 0.2–0.3 for deeper layers).

## Evaluation
The model is evaluated using mAP with an IoU threshold of 0.5. The `get_bboxes` function processes model outputs and ground truth labels to compute bounding box predictions, followed by NMS. The `mean_average_precision` function calculates mAP across all classes.

## Future Improvements
- Add support for multi-scale training to improve detection robustness.
- Implement data augmentation (e.g., random flips, rotations) in `Compose`.
- Optimize segmentation output for specific tasks (e.g., instance segmentation).
- Integrate mixed-precision training more extensively for faster training on GPUs.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Inspired by the original YOLOv1 paper and U-Net architecture.
- Built with PyTorch and torchvision for efficient deep learning workflows.
import os
from ultralytics import YOLO

# Define paths for project organization
project_dir = "./models"  # Base directory for saving models and results
experiment_name = "yolo_object_lane"  # Experiment name for organization
weights_dir = os.path.join(project_dir, experiment_name, "weights")  # Directory for saving model weights
os.makedirs(weights_dir, exist_ok=True)  # Create weights directory if it does not exist

# Load the pre-trained YOLO11 model
model = YOLO("yolo11n.pt")  # Nano model, optimized for devices like Jetson

if __name__ == "__main__":
    # Train the model with the specified dataset
    results = model.train(
        data="./dataset/data.yaml",  # Path to the dataset configuration file
        epochs=200,  # Total number of training epochs
        imgsz=320,  # Image size (optimized for Jetson)
        hsv_h=0.015,  # Hue adjustment for data augmentation
        hsv_s=0.1,  # Saturation adjustment for data augmentation
        hsv_v=0.1,  # Value (brightness) adjustment for data augmentation
        translate=0.0,  # Translation offset for data augmentation
        scale=0.0,  # Scale for data augmentation
        fliplr=0.0,  # Probability of horizontal flip
        mosaic=0.0,  # Probability of mosaic data augmentation
        erasing=0.0,  # Probability of random erasing
        auto_augment=None,  # Disable automatic augmentation for manual control
        batch=8,  # Batch size for training
        amp=True,  # Enable Mixed Precision to optimize GPU usage
        device=0,  # Use GPU 0 (or CPU if device=-1)
        workers=2,  # Number of workers for data loading
        project=project_dir,  # Base directory for saving results
        name=experiment_name,  # Experiment name for organization
        exist_ok=True,  # Allow overwriting previous results
        freeze=0,  # Do not freeze any model layers during training
        lr0=0.001,  # Initial learning rate
        patience=0,  # Disable early stopping
        weight_decay=0.0005,  # L2 regularization to prevent overfitting
        save_period=20,  # Save model checkpoints every 20 epochs
        save=True  # Save the best model based on mAP metric
    )

    # Validate the model on the validation set
    val_results = model.val(
        data="./dataset/data.yaml",  # Dataset configuration file for validation
        imgsz=320,  # Image size for validation
        batch=16,  # Batch size for validation (larger for powerful GPUs)
        device='0'  # Use GPU 0 for validation
    )

    # Display validation metrics
    print("Validation Results (GPU):")
    print(f"mAP@0.5: {val_results.box.map50:.4f}")  # Mean Average Precision at IoU=0.5
    print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")  # Mean Average Precision at IoU from 0.5 to 0.95
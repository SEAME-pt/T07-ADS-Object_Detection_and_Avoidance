# Requirements Document: Comparing YOLOv8 and YOLOv11 for Traffic Sign Detection on Jetson Nano

This document compares YOLOv8 and YOLOv11 for detecting traffic signs on the NVIDIA Jetson Nano, a small computer designed for AI tasks. It explains why YOLOv11 is the better choice for this project.

## 1. Purpose
The goal is to choose the best model for detecting traffic signs in real-time on the Jetson Nano, ensuring good accuracy, fast performance, and low resource usage.

## 2. Comparison of YOLOv8 and YOLOv11

### 2.1. Accuracy
- **YOLOv8**: Performs well for traffic sign detection but struggles with small signs or complex backgrounds (e.g., trees or bad weather). It achieves around 0.95 mAP (a measure of accuracy) for larger objects but may miss smaller signs.
- **YOLOv11**: More accurate, especially for small traffic signs. It handles complex scenes better, reducing missed detections.

**Why YOLOv11 is Better**: It detects small traffic signs more reliably, which is critical for safe driving systems.

### 2.2. Speed
- **YOLOv8**: Runs fast on the Jetson Nano for traffic sign detection.
- **YOLOv11**: Slightly faster due to improved design. Benchmarks on Jetson devices show it can achieve similar or better speeds.

**Why YOLOv11 is Better**: It maintains or improves speed, ensuring real-time detection on the Jetson Nano’s limited hardware.

### 2.3. Resource Usage
- **YOLOv8**: The Nano version (YOLOv8n) is lightweight but still uses significant memory and power, which can strain the Jetson Nano’s 4GB RAM and 10W power limit.
- **YOLOv11**: The Nano version (YOLOv11n) is designed to be even more efficient, using less memory and power while maintaining high performance. This makes it ideal for the Jetson Nano’s constraints.

**Why YOLOv11 is Better**: It runs more smoothly on the Jetson Nano without overloading its resources.

### 2.4. Ease of Use
- **YOLOv8**: Easy to set up with the Ultralytics library, supports PyTorch, and works well with Jetson’s JetPack software.
- **YOLOv11**: Also uses the Ultralytics library, with similar setup steps. It supports additional features like keypoint detection, which could be useful for future traffic sign projects, but the basic setup is just as straightforward.

**Why YOLOv11 is Better**: It offers the same ease of use but with newer features that make it more flexible for future needs.

## 3. Why Choose YOLOv11 for Traffic Sign Detection on Jetson Nano
YOLOv11 is the better choice for this project because:
- It detects small traffic signs more accurately, which is crucial for safety in real-world driving.
- It runs at a similar or faster speed, ensuring real-time performance on the Jetson Nano.
- It uses fewer resources, making it more reliable on the Jetson Nano’s limited hardware.
- It’s just as easy to use as YOLOv8, with support for the same tools and Jetson’s JetPack.


## 4. Conclusion
YOLOv11 is recommended over YOLOv8 for traffic sign detection on the Jetson Nano because it offers better accuracy, similar or faster speed, and lower resource usage, all while being easy to set up. This makes it ideal for real-time detection in a resource-constrained environment like the Jetson Nano.


Developed by: Team07 - SEA:ME Portugal

[![Team07](https://img.shields.io/badge/SEAME-Team07-blue?style=plastic)](https://github.com/orgs/SEAME-pt/teams/team07)
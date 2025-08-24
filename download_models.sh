#!/bin/bash

# Script to download YOLOv8n model files for vehicle tracking
echo "Downloading YOLOv8n model files for AI service..."

# Create models directory
mkdir -p models
cd models

# Download YOLOv8n ONNX model
echo "Downloading YOLOv8n ONNX model..."
wget -O yolov8n.onnx https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

# Download YOLOv8n weights and config (fallback)
echo "Downloading YOLOv8n weights and config..."
wget -O yolov8n.weights https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.weights
wget -O yolov8n.cfg https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/models/v8/yolov8n.yaml

# Verify downloads
echo "Verifying downloaded files..."
ls -la *.onnx *.weights *.cfg

echo "Model files downloaded successfully!"
echo "Note: You may need to convert the YAML config to proper Darknet format for OpenCV DNN"

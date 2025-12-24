import os
import sys

print("Checking imports...")
try:
    import streamlit
    import tensorflow as tf
    import cv2
    import numpy as np
    from mtcnn.mtcnn import MTCNN
    from PIL import Image
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

print("Checking model files...")
models_dir = os.path.join("Deepfake-Detector", "models")
if not os.path.exists(models_dir): # Adjust path if running from parent
    models_dir = os.path.join("models")
    
model_ssd_path = os.path.join(models_dir, "deepfake_detector_FINAL.weights.h5")
model_v1_path = os.path.join(models_dir, "deepfake_detector_v1.h5")

if os.path.exists(model_ssd_path):
    print(f"Found EfficientNet weights: {model_ssd_path}")
else:
    print(f"MISSING EfficientNet weights: {model_ssd_path}")

if os.path.exists(model_v1_path):
    print(f"Found V1 model: {model_v1_path}")
else:
    print(f"MISSING V1 model: {model_v1_path}")

print("Verification script complete.")

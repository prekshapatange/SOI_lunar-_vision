import os
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# --- Config ---
MODEL_PATH = "../Trained_Model/best.pt"               # Path to trained YOLO model
TEST_DIR = "../Dataset/test/images"                  # Folder with validation/test images
OUTPUT_DIR = "../Predictions"                         # Output directory for predictions
SAVE_TXT = True                                       # Save YOLO-style label txt files

# --- Create output folders ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subfolders inside Predictions/
output_images_dir = os.path.join(OUTPUT_DIR, "images")
output_labels_dir = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(output_images_dir, exist_ok=True)
if SAVE_TXT:
    os.makedirs(output_labels_dir, exist_ok=True)

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)

# --- Run prediction ---
results = model.predict(
    source=TEST_DIR,
    save=True,
    save_txt=SAVE_TXT,
    project=OUTPUT_DIR,
    name="",                  # So outputs are directly saved in OUTPUT_DIR
    exist_ok=True,
    conf=0.3,                 # Confidence threshold
    iou=0.5                   # IoU threshold for NMS
)

print(f"✅ Predictions saved to: {os.path.abspath(OUTPUT_DIR)}")

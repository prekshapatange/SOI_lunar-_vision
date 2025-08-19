import gradio as gr
import cv2
import numpy as np
import json
import time
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load YOLOv8 model
model = YOLO("../Trained_Model/best.pt")  # Update path if needed

# Generate saliency map
def generate_saliency_map(image_np, results):
    saliency = np.zeros(image_np.shape[:2], dtype=np.float32)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        saliency[y1:y2, x1:x2] += float(box.conf[0])
    saliency = np.clip(saliency, 0, 1)
    saliency = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), 0.6, saliency, 0.4, 0)
    return blended

# Main detection function
def detect_craters(image, confidence, show_saliency, iou_thresh):
    start = time.time()
    image_np = np.array(image.convert("RGB"))
    results = model(image_np, conf=confidence, iou=iou_thresh)
    end = time.time()

    # Detection overlay
    result_img = results[0].plot()
    result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

    # Saliency map
    saliency_pil = None
    if show_saliency:
        saliency_img = generate_saliency_map(image_np, results)
        saliency_pil = Image.fromarray(cv2.cvtColor(saliency_img, cv2.COLOR_BGR2RGB))

    # Detection stats
    num_craters = len(results[0].boxes)
    avg_conf = float(np.mean(results[0].boxes.conf.cpu().numpy())) if num_craters > 0 else 0
    stats = f"""
**🕳️ Craters Detected:** {num_craters}  
**🎯 Avg Confidence:** {avg_conf:.2f}  
**🧩 IoU Threshold:** {iou_thresh:.2f}  
**⏱️ Detection Time:** {end - start:.2f} seconds
"""

    # Raw YOLO-style output
    predictions = []
    img_h, img_w = image_np.shape[:2]
    for box in results[0].boxes:
        class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        predictions.append([
            class_id,
            round(x_center, 6),
            round(y_center, 6),
            round(width, 6),
            round(height, 6),
            round(conf, 4)
        ])

    json_result = json.dumps(predictions, indent=2)

    # Save output image
    orig_filename = getattr(image, "name", "output.png")
    base_name = os.path.splitext(os.path.basename(orig_filename))[0]
    ext = os.path.splitext(orig_filename)[1] or ".png"
    output_filename = f"{base_name}_detected{ext}"
    temp_path = os.path.join(tempfile.gettempdir(), output_filename)
    result_pil.save(temp_path)

    return image, result_pil, saliency_pil, stats, json_result, temp_path

# Gradio UI
inputs = [
    gr.Image(type="pil", label="📤 Upload Lunar Image"),
    gr.Slider(0.1, 1.0, value=0.25, label="🎚️ Confidence Threshold"),
    gr.Checkbox(label="🧠 Show Saliency Map"),
    gr.Slider(0.1, 1.0, value=0.5, label="🧩 IoU Threshold")
]

outputs = [
    gr.Image(type="pil", label="🖼️ Original Image"),
    gr.Image(type="pil", label="📦 Crater Detection"),
    gr.Image(type="pil", label="🔥 Saliency Map"),
    gr.Markdown(label="📊 Detection Stats"),
    gr.JSON(label="📋 YOLO-style JSON Output"),
    gr.File(label="📥 Download Annotated Image")
]

app = gr.Interface(
    fn=detect_craters,
    inputs=inputs,
    outputs=outputs,
    title="🌙 Lunar Crater Detection Interface",
    description="Upload a lunar surface image and instantly detect craters using YOLOv8. Includes adjustable confidence and IoU thresholds, saliency maps, JSON output, and download option.",
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    app.launch(share=True)

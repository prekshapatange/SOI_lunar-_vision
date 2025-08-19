
#  Lunar Crater Detection – Gradio Interface

This interactive UI enables users to upload high-resolution lunar surface images and detect craters using a custom-trained YOLOv8 model. It also provides visual explanations (saliency maps), statistical outputs, and downloadable predictions.

---

##  Features

- Real-time crater detection
- Saliency Map generation for explainability
- Stats like number of craters, confidence, and detection time
- JSON output of bounding boxes
- Option to download annotated output

---

## Folder Location

This UI is located in:

```
UI_UX/
├── app.py                  ← Main Gradio interface script
├── requirements.txt        ← All required Python packages
└── README_UI.md            ← (You’re here)
```

---

##  How to Run the Interface

### Step 1: Install Dependencies

If you're running locally, install the required packages:

```bash
pip install -r requirements.txt
```

### Step 2: Launch the App

From within the `UI_UX/` directory, run:

```bash
python app.py
```

This will start a local Gradio interface. If running in Google Colab or remotely, it will also give a public sharing link.
since Gradio doesn't automatically open the webpage,you have to open it manually using the public sharing link.

---

## Model Dependency

Ensure the trained YOLOv8 model (`best.pt`) is available at the following relative path:

```python
model = YOLO("../Trained_Model/best.pt")
```

If you change the directory structure, update this path accordingly in `app.py`.

---

## Example Input & Output

- Input: Lunar surface images (`.jpg`, `.png`)
- Output:
  - Original Image
  - Detected Craters with Bounding Boxes
  - Optional Saliency Map
  - Detection Stats (crater count, confidence)
  - adjustable confidence Threshold
  - adjustable NMS(IoU) Threshold
  - Raw JSON output(in extended YOLO format)
  - Downloadable annotated image

---

## Technologies Used

- [Gradio](https://gradio.app/) – for building the UI
- [YOLOv8](https://github.com/ultralytics/ultralytics) – detection model
- OpenCV, NumPy, Pillow – image processing and I/O
- JSON – structured result output

---

## Contributors

- Team : The MC's 
- IIT Dharwad

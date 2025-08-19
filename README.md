
#  Lunar Crater Detection using YOLOv8

This project leverages the YOLOv8n (nano) object detection model to identify craters on lunar surface images. The pipeline supports training, evaluation, and inference on custom annotated datasets, and produces visual and statistical outputs.

---

## Folder Structure

```
SOI'25-Lunar_Crater_Detection:The MC's/
├── Dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│   |   ├── images/
│   |   └── labels/
|   └── test/
│       ├── images/
│       
|
├── Code_files/
│   ├── train_yolo.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── data.yaml
│   ├── args.yaml
|
|___Documentations/
|   |___README.md ← (you’re here)
|   |___requirements.txt
|   |___Visualisations/
|        |__confusion_matrices,Precision-recall curves,labels,results(also added the        results.csv of the partially trained model) etc..
|
├── Predictions/
│   |── lables ← YOLO-style predicted labels
|   |__images ←  predicted images with bounding boxes
|
├── Trained_Model/
│   └── best.pt
|   └── last.pt
|
├── UI_UX/
│   ├── app.py
│   ├── requirements.txt
│   └── README_UI.md
|
├── 6_Report/
│   └── Final_Report.pdf

```

---

##  Model

We use **YOLOv8n (nano model) (Ultralytics)** for crater detection due to its accuracy and lightweight architecture and efficient computational cost

- Model: YOLOv8 (custom-trained)
- Classes: `['crater']`
- Data Format: YOLO format (`.txt` files with normalized bbox values)

---

##  Training

To train the model:

```bash
python train_yolo.py
```

Make sure the `data.yaml` inside `Code_files/` points correctly to `../Dataset`.
And make sure the 'Dataset' folder is correctly organized w.r.t the given folder structure.
(we have only provided test images and labels inside the Dataset folder)
---

## Evaluation

Evaluate the trained model's performance:

```bash
python evaluate.py
```

This will print precision, recall, and mAP values to the console.

---

##  Inference / Prediction

Run predictions on custom test images using:

```bash
python predict.py
```

The output label files (YOLO format) will be saved in `Predictions/`.

---

##  Visualization (Curves & Metrics)

All visual analytics (e.g., loss curves, PR curves, confusion matrices) can be found in:

```
Documentations/Visualisations/
```

---

##  Requirements

To install all dependencies used in the pipeline:

```bash
pip install -r requirements.txt
```

---

##  Notes

- `args.yaml` contains hyperparameters and training configurations.
- `best.pt` is the trained YOLOv8 model file, saved in `Trained_Model/`.

---


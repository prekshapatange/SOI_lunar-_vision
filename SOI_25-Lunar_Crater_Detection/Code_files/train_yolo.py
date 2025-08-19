from ultralytics import YOLO

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  
    model.train(data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="crater_yolov8",
    patience=10)




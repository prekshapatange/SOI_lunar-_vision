from ultralytics import YOLO

def run_eval():
    model = YOLO("../Trained_Model/best.pt")

    metrics = model.val()
    # Print key metricss
    print("Precision:", metrics.box.p)
    print("Recall:", metrics.box.r)
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_eval()

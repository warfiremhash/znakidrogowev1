from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="traffic_sign.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=2,
        name="traffic_signs",
        device="cpu"
    )

if __name__ == "__main__":
    main()

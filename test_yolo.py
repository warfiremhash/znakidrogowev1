from ultralytics import YOLO
import os


def main():
    # Wczytaj wytrenowany model
    model = YOLO("runs/detect/traffic_signs/weights/best.pt")  # Ścieżka do modelu

    # Ścieżka do folderu z obrazami do przetestowania
    test_images_dir = "datasets/detection/images/val"  # Folder z obrazami testowymi

    # Ścieżka do zapisu wyników
    results_dir = "test/results"
    os.makedirs(results_dir, exist_ok=True)

    # Przetestuj model na każdym obrazie
    for image_name in os.listdir(test_images_dir):
        if image_name.endswith((".jpg", ".png")):  # Filtruj obrazy
            image_path = os.path.join(test_images_dir, image_name)

            # Wykonaj predykcję
            results = model.predict(source=image_path, save=True, save_txt=True, project=results_dir, name="test")

            # Wyświetl wyniki w konsoli
            print(f"Wyniki dla obrazu: {image_name}")
            for result in results:
                print(f"  Klasa: {result.boxes.cls}")
                print(f"  Współrzędne bounding box: {result.boxes.xyxy}")
                print(f"  Pewność predykcji: {result.boxes.conf}")

            print(f"Wynik zapisany w folderze: {results_dir}")


if __name__ == "__main__":
    main()

import tkinter as tk
import cv2
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO

# model YOLOv8
model = YOLO("runs/detect/traffic_signs/weights/best.pt")

def predict_and_display(image_path):
    """Funkcja do przeprowadzenia predykcji i wyświetlenia wyników na zdjęciu."""
    try:
        panel.pack()
        video_panel.pack_forget()

        results = model.predict(source=image_path, conf=0.5, save=False)

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        font_path = "arial.ttf"
        font_size = 70
        font = ImageFont.truetype(font_path, font_size)

        detected_classes = []

        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, xyxy)

            draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)

            label = f"{model.names[cls]}: {conf:.2f}"
            text_bbox = draw.textbbox((x_min, y_min), label, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_background = [(x_min, y_min - text_height - 2), (x_min + text_width + 2, y_min)]
            draw.rectangle(text_background, fill="blue")
            draw.text((x_min, y_min - text_height - 2), label, fill="white", font=font)

            detected_classes.append(f"{model.names[cls]}: {conf:.2%}")

        results_text.delete(1.0, tk.END)
        if detected_classes:
            results_text.insert(tk.END, "Wykryte znaki:\n")
            results_text.insert(tk.END, "\n".join(detected_classes))
        else:
            results_text.insert(tk.END, "Nie wykryto żadnych znaków.")

        display_image(image)

    except Exception as e:
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, f"Błąd: {e}")




def analyze_video(video_path):
    """Funkcja do analizy i odtwarzania nagrania wideo w GUI."""
    try:
        panel.pack_forget()
        video_panel.pack()
        results_text.pack_forget()

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_path = "video/results/output_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        def process_frame():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                out.release()
                results_text.insert(tk.END, f"Wideo zapisano: {output_path}\n")
                return

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model.predict(source=frame, conf=0.5, save=False)
            draw = ImageDraw.Draw(frame_pil)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            for box in results[0].boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = map(int, xyxy)

                draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)
                label = f"{model.names[cls]}: {conf:.2f}"
                draw.text((x_min, y_min - 15), label, fill="red", font=font)

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            out.write(frame)

            frame_image = ImageTk.PhotoImage(frame_pil)
            video_panel.config(image=frame_image)
            video_panel.image = frame_image

            video_panel.after(int(1000 / fps), process_frame)

        process_frame()

    except Exception as e:
        results_text.insert(tk.END, f"Błąd podczas analizy wideo: {e}\n")



def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        predict_and_display(file_path)

def load_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        analyze_video(file_path)

def display_image(image):
    image.thumbnail((800, 600))
    img_tk = ImageTk.PhotoImage(image)
    panel.config(image=img_tk)
    panel.image = img_tk

window = tk.Tk()
window.title("YOLOv8 Traffic Sign Detection")

btn = tk.Button(window, text="Wczytaj obraz", command=load_image, font=("Arial", 14))
btn.pack(pady=10)

btn_video = tk.Button(window, text="Wczytaj wideo", command=load_video, font=("Arial", 14))
btn_video.pack(pady=10)

panel = tk.Label(window)
panel.pack()

video_panel = tk.Label(window)
video_panel.pack()
video_panel.pack_forget()

results_text = tk.Text(window, height=10, width=70, font=("Arial", 12))
results_text.pack(pady=10)

window.mainloop()
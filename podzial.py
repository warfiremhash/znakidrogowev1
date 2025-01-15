import os
import shutil
import random

# Foldery bazowe
IMG_DIR = 'datasets/detection/imgs'
LABEL_DIR = 'datasets/detection/labels'

# Docelowe podfoldery YOLO
TRAIN_IMG_DIR = 'datasets/detection/images/train'
VAL_IMG_DIR = 'datasets/detection/images/val'
TRAIN_LABEL_DIR = 'datasets/detection/labels/train'
VAL_LABEL_DIR = 'datasets/detection/labels/val'

# Tworzymy podfoldery
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

# Lista obrazów od 0 do 150
all_images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg') and f.split('.')[0].isdigit() and 0 <= int(f.split('.')[0]) <= 350]
all_images = sorted(all_images)  # Sortujemy nazwy rosnąco

# Mieszamy wybrane obrazy
random.shuffle(all_images)

# Ustal proporcje train/val (80% train, 20% val)
train_ratio = 0.8
train_count = int(train_ratio * len(all_images))
train_files = all_images[:train_count]
val_files = all_images[train_count:]

# Funkcja do kopiowania powiązanych plików .jpg i .txt
def move_pair(img_name, target_img_dir, target_lbl_dir):
    # Kopiujemy/przenosimy obraz
    shutil.copy(os.path.join(IMG_DIR, img_name),
                os.path.join(target_img_dir, img_name))

    # Odpowiadający plik .txt
    label_name = img_name.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(LABEL_DIR, label_name)):
        shutil.copy(os.path.join(LABEL_DIR, label_name),
                    os.path.join(target_lbl_dir, label_name))
    else:
        # Jeśli brak .txt, YOLO uzna to za "no objects".
        print(f"Brak pliku .txt dla obrazu: {img_name}")

# Rozdział plików
for f in train_files:
    move_pair(f, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

for f in val_files:
    move_pair(f, VAL_IMG_DIR, VAL_LABEL_DIR)

print("Przetwarzanie zakończone!")

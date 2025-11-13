import os
import cv2
import numpy as np

dataset_path = "/Users/bhara-zstch1566/CNN/Eval/dataset"
IMG_SIZE = (128, 128)

X = []
Y = []
class_lab = []
cnt = 0

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def produce(folder_path, label):
    total = 0
    for root, _, files in os.walk(folder_path):
        for img_name in files:
            if not img_name.lower().endswith(VALID_EXTENSIONS):
                continue
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipped unreadable image: {img_path}")
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            X.append(img)
            Y.append(label)
            total += 1
    return total


for folder in os.listdir(dataset_path):
    if folder.startswith('.'):
        continue

    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):
        continue

    count = produce(folder_path, cnt)
    print(f"{cnt}: Loaded class '{folder}' with {count} images.")
    class_lab.append(folder)
    cnt += 1


X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int32)

print("\n✅ Dataset loaded successfully!")
print(f"Total images: {len(X)}")
print(f"Image shape: {X.shape[1:] if len(X) > 0 else '()'} (H,W,C)")
print(f"Classes: {class_lab}")

np.savez_compressed("dataset_preprocessed.npz", X=X, Y=Y)

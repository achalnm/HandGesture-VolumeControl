import argparse
import csv
import os

import cv2
import numpy as np
import pandas as pd

IMG_SIZE = 224
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def generate_csv(folder1, folder2, output_csv):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder', 'Full Path'])
        for folder, label in [(folder1, 'thumbs_up'), (folder2, 'thumbs_down')]:
            for root, _, files in os.walk(folder):
                for fname in sorted(files):
                    if os.path.splitext(fname.lower())[1] in SUPPORTED_EXTS:
                        writer.writerow([label, os.path.abspath(os.path.join(root, fname))])
    print(f"Saved {output_csv}")


def preprocess_data(csv_file, img_size=IMG_SIZE, normalize=True):
    data = pd.read_csv(csv_file)
    images, labels = [], []
    for _, row in data.iterrows():
        try:
            img = cv2.imread(row['Full Path'])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            if normalize:
                img = img / 255.0
            images.append(img)
            labels.append(1 if row['Folder'] == 'thumbs_up' else 0)
        except Exception:
            continue
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', default='folder1')
    parser.add_argument('--folder2', default='folder2')
    parser.add_argument('--output',  default='data/labels.csv')
    args = parser.parse_args()
    generate_csv(args.folder1, args.folder2, args.output)


if __name__ == '__main__':
    main()

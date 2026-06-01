import argparse
import csv
import os

import cv2
import numpy as np
import pandas as pd

# --- Constants ---
IMG_SIZE = 224
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png'}


def generate_csv(folder1_path, folder2_path, output_csv):
    """Scan two image folders and write a two-column labeled CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder', 'Full Path'])
        for folder_path, label in [(folder1_path, 'thumbs_up'), (folder2_path, 'thumbs_down')]:
            for root, _, files in os.walk(folder_path):
                for fname in sorted(files):
                    if os.path.splitext(fname.lower())[1] in SUPPORTED_EXTS:
                        writer.writerow([label, os.path.abspath(os.path.join(root, fname))])
    print(f"CSV saved to {output_csv}")


def preprocess_data(csv_file, img_size=IMG_SIZE, normalize=True):
    """Load images from a CSV and return float32 arrays.

    Args:
        normalize: if True, scale to [0, 1]. Set False when using
                   mobilenet_v2.preprocess_input (which expects [0, 255]).
    """
    data = pd.read_csv(csv_file)
    images, labels = [], []
    for _, row in data.iterrows():
        filename = row['Full Path']
        folder = row['Folder']
        try:
            image = cv2.imread(filename)
            if image is None:
                print(f"Warning: could not load '{filename}', skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MobileNetV2 trained on RGB
            image = cv2.resize(image, (img_size, img_size))
            if normalize:
                image = image / 255.0
            images.append(image)
            labels.append(1 if folder == 'thumbs_up' else 0)
        except Exception as e:
            print(f"Error processing '{filename}': {e}")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description='Generate labeled CSV from image folders')
    parser.add_argument('--folder1', type=str, default='folder1', help='Path to thumbs_up images')
    parser.add_argument('--folder2', type=str, default='folder2', help='Path to thumbs_down images')
    parser.add_argument('--output', type=str, default='data/labels.csv', help='Output CSV path')
    args = parser.parse_args()
    generate_csv(args.folder1, args.folder2, args.output)


if __name__ == '__main__':
    main()

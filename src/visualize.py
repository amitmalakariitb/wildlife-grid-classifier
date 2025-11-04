# src/visualize.py
"""
Create annotated images with rectangles over predicted positive cells.
Reads: /output/predictions.csv
Writes annotated images to /output/annotated/
"""
import os
import cv2
import pandas as pd

import config
import preprocess

INPUT_CSV = config.PREDICTIONS_CSV
IMAGE_FOLDER_FALLBACK = config.PREPROCESSED_DIR
IMAGE_FOLDER_TEST = config.TEST_IMAGES_DIR
OUT_DIR = config.ANNOTATED_DIR
CELL_W, CELL_H = preprocess.CELL_W, preprocess.CELL_H

os.makedirs(OUT_DIR, exist_ok=True)

def annotate_images(image_folder):
    if not os.path.exists(INPUT_CSV):
        print("Predictions CSV not found at", INPUT_CSV)
        return
    df = pd.read_csv(INPUT_CSV)
    for _, row in df.iterrows():
        fname = row['Image']
        img_path = os.path.join(image_folder, fname)
        if not os.path.exists(img_path):
            print("Missing image:", img_path); continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        # ensure target size (preprocessed images should already be correct)
        img = cv2.resize(img, (CELL_W*8, CELL_H*8))
        for i in range(64):
            try:
                val = int(row[f"c{i+1:02d}"])
            except Exception:
                val = 0
            if val == 1:
                r, c = divmod(i, 8)
                x1, y1 = c*CELL_W, r*CELL_H
                x2, y2 = x1 + CELL_W, y1 + CELL_H
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, img)
    print("Annotated images saved to", OUT_DIR)

if __name__ == "__main__":
    image_folder = IMAGE_FOLDER_TEST if os.path.isdir(IMAGE_FOLDER_TEST) else IMAGE_FOLDER_FALLBACK
    annotate_images(image_folder)
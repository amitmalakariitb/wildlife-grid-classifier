# src/predict.py
"""
Run inference on new images and generate /output/predictions.csv
CSV columns: Image,c01..c64
"""
import os
import cv2
import pandas as pd
import joblib
from tqdm import tqdm
import numpy as np

import config
import preprocess
import feature_extraction

MODEL_PATH = config.MODEL_PIPELINE
IMAGE_FOLDER_DEFAULT = config.TEST_IMAGES_DIR if os.path.isdir(config.TEST_IMAGES_DIR) else config.PREPROCESSED_DIR
IMAGE_FOLDER_FALLBACK = config.IMAGES_DIR
OUTPUT_CSV = config.PREDICTIONS_CSV

def predict_folder(image_folder):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    rows = []
    for fn in tqdm(files, desc="Predict images"):
        path = os.path.join(image_folder, fn)
        img = cv2.imread(path)
        if img is None:
            print("Cannot read:", fn); continue
        try:
            cells = preprocess.slice_into_cells(img)
        except Exception as e:
            print("Skipping", fn, "slice failed:", e)
            continue
        feats = [feature_extraction.extract_features_from_cell(c) for c in cells]
        preds = model.predict(np.vstack(feats)).astype(int).tolist()
        rows.append([fn] + preds)
    cols = ["Image"] + [f"c{i+1:02d}" for i in range(64)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Predictions written to:", OUTPUT_CSV)
    return OUTPUT_CSV

if __name__ == "__main__":
    image_folder = IMAGE_FOLDER_DEFAULT if os.path.isdir(IMAGE_FOLDER_DEFAULT) else IMAGE_FOLDER_FALLBACK
    if not os.path.isdir(image_folder):
        print("No images folder found. Create data/preprocessed_images or data/images")
        exit(1)
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_model.py first.")
        exit(1)
    predict_folder(image_folder)
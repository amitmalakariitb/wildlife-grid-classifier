# src/train_model.py
"""
Train a cell-level classifier.
- Reads labels from /data/labels.csv
- Reads images from /data/images
- Preprocess -> slice -> extract features
- Train RandomForest; prints metrics and saves model to /models/model.pkl
"""
import os
import cv2
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import config
import preprocess
import feature_extraction

IMAGE_FOLDER = config.PREPROCESSED_DIR if os.path.isdir(config.PREPROCESSED_DIR) else config.IMAGES_DIR
LABELS_CSV = config.LABELS_CSV
MODEL_PATH = config.MODEL_PIPELINE

def load_dataset():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")
    df = pd.read_csv(LABELS_CSV)
    X_list = []
    y_list = []
    groups = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Images"):
        img_name = row['ImageFileName'] if 'ImageFileName' in row else row['Image']
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        if not os.path.exists(img_path):
            # skip missing files
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        try:
            cells = preprocess.slice_into_cells(img)
        except Exception:
            # skip if image not exactly target size
            continue
        for i, cell in enumerate(cells):
            feat = feature_extraction.extract_features_from_cell(cell)
            X_list.append(feat)
            y_list.append(int(row[f"c{i+1:02d}"]))
            groups.append(img_name)
    if len(X_list) == 0:
        raise RuntimeError("No training data found. Check labels and preprocessed images.")
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, np.array(groups)

def train_random_forest(X, y, save_path=MODEL_PATH):
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced"))
    ])
    print("Training RandomForest pipeline...")
    pipeline.fit(X_train, y_train)
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump(pipeline, save_path)
    print("Model pipeline saved to:", save_path)
    return pipeline

if __name__ == "__main__":
    try:
        X, y, groups = load_dataset()
    except Exception as e:
        print("Error loading dataset:", e)
        exit(1)
    print("Dataset shape:", X.shape, y.shape)
    train_random_forest(X, y)
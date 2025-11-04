
import os
import cv2
import pandas as pd
import numpy as np

import config
import preprocess

IMAGE_DIR = config.PREPROCESSED_DIR
OUT_CSV = config.LABELS_CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

grid_rows, grid_cols = 8, 8
cell_w, cell_h = preprocess.CELL_W, preprocess.CELL_H

images = sorted([f for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
index = 0
labels = {}

# resume previous labels if present
if os.path.exists(OUT_CSV):
    df = pd.read_csv(OUT_CSV)
    for _, row in df.iterrows():
        # accept either "Image" or "ImageFileName" for compatibility
        fname = row.get('Image') or row.get('ImageFileName')
        if not isinstance(fname, str):
            continue
        vals = [int(row.get(f"c{i+1:02d}", 0)) for i in range(grid_rows*grid_cols)]
        labels[fname] = vals

# Start from first unlabeled image (skip files already labeled)
unlabeled = [f for f in images if f not in labels]
if unlabeled:
    index = images.index(unlabeled[0])
    print(f"Starting at first unlabeled image: {unlabeled[0]} (index {index})")
else:
    print("All images already labeled. Starting at first image.")

def save_labels():
    rows = []
    for k, v in labels.items():
        d = {"Image": k}
        d.update({f"c{i+1:02d}": int(v[i]) for i in range(grid_rows*grid_cols)})
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(labels)} labeled images to {OUT_CSV}")

def draw_grid(img, grid_state):
    overlay = img.copy()
    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = r*grid_cols + c
            x1, y1 = c*cell_w, r*cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            color = (0,255,0) if grid_state[idx] == 1 else (0,0,255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    return overlay

# toggle cell on left-click
def mouse_callback(event, x, y, flags, param):
    global grid_state
    if event == cv2.EVENT_LBUTTONDOWN:
        col = min(x // cell_w, grid_cols - 1)
        row = min(y // cell_h, grid_rows - 1)
        idx = int(row*grid_cols + col)
        grid_state[idx] = 1 - grid_state[idx]

if not images:
    print("No images in", IMAGE_DIR)
    exit(1)

print("LabelTool keys: n=next, b=back, s=save, r=reset current, q=quit")
while True:
    if index < 0: index = 0
    if index >= len(images): index = len(images)-1
    fname = images[index]
    img_path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print("Cannot read:", fname)
        index += 1
        continue
    # ensure display at target size (preprocessed images should be correct)
    img = cv2.resize(img, (cell_w*8, cell_h*8))
    grid_state = labels.get(fname, [0]*(grid_rows*grid_cols)).copy()

    cv2.namedWindow("LabelTool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("LabelTool", mouse_callback)

    while True:
        disp = draw_grid(img, grid_state)
        cv2.imshow("LabelTool", disp)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('n'):            # next image
            labels[fname] = grid_state.copy()
            index += 1
            break
        elif key == ord('b'):          # back
            labels[fname] = grid_state.copy()
            index -= 1
            break
        elif key == ord('s'):          # save current labels
            labels[fname] = grid_state.copy()
            save_labels()
        elif key == ord('r'):          # reset current grid
            grid_state = [0] * (grid_rows*grid_cols)
            labels[fname] = grid_state.copy()
            print("Reset grid for", fname)
        elif key == ord('q'):          # quit
            labels[fname] = grid_state.copy()
            save_labels()
            cv2.destroyAllWindows()
            exit(0)
    cv2.destroyAllWindows()

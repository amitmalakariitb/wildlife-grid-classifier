# src/preprocess.py
"""
Preprocessing utilities:
- preprocess_image(path) -> 800x600 BGR image
- slice_into_cells(img) -> list of 64 cell images (BGR)
"""
import cv2
import numpy as np
import os

CELL_W, CELL_H = 100, 75
TARGET_W, TARGET_H = CELL_W * 8, CELL_H * 8  # 800 x 600

def crop_to_4_3(img, center=True):
    """
    Center-crop (or top-left crop) image to 4:3 aspect ratio.
    """
    h, w = img.shape[:2]
    desired_ar = 4.0 / 3.0
    ar = w / h
    if abs(ar - desired_ar) < 1e-6:
        return img
    if ar > desired_ar:
        # too wide -> crop width
        new_w = int(h * desired_ar)
        if center:
            off = (w - new_w) // 2
        else:
            off = 0
        return img[:, off:off + new_w]
    else:
        # too tall -> crop height
        new_h = int(w / desired_ar)
        if center:
            off = (h - new_h) // 2
        else:
            off = 0
        return img[off:off + new_h, :]

def enhance_clahe(img):
    """
    Apply CLAHE to the L channel in LAB color space and return BGR image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_and_save(image_path, out_path, do_enhance=True, center_crop=True):
    """
    Strict preprocessing according to pipeline rules:
    - center-crop to 4:3 (if needed)
    - if crop result >= 800x600 => scale down to 800x600 (no upscaling)
    - if crop result < 800x600 in any dim => skip (return False)
    - optional CLAHE enhancement
    - save result to out_path (overwrites)
    Returns True if saved, False if skipped or unreadable.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    cropped = crop_to_4_3(img, center=center_crop)
    h, w = cropped.shape[:2]
    # Do not upscale: skip images smaller than target
    if h < TARGET_H or w < TARGET_W:
        return False
    # Downscale if larger than target using INTER_AREA
    if (w, h) != (TARGET_W, TARGET_H):
        resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    else:
        resized = cropped
    if do_enhance:
        resized = enhance_clahe(resized)
    # Sanity check
    if resized.shape[0] != TARGET_H or resized.shape[1] != TARGET_W:
        return False
    cv2.imwrite(out_path, resized)
    return True

def slice_into_cells(img):
    """
    Expect exact TARGET_W x TARGET_H image. Return list of 64 cell images (BGR).
    """
    h, w = img.shape[:2]
    if (w, h) != (TARGET_W, TARGET_H):
        raise ValueError(f"slice_into_cells expects {TARGET_W}x{TARGET_H} image, got {w}x{h}")
    cells = []
    for r in range(8):
        for c in range(8):
            y1, y2 = r * CELL_H, (r + 1) * CELL_H
            x1, x2 = c * CELL_W, (c + 1) * CELL_W
            cells.append(img[y1:y2, x1:x2].copy())
    return cells

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <in_image> <out_image>")
        print("Example: python preprocess.py data/images/img1.jpg data/preprocessed_images/img1.jpg")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2]
    ok = preprocess_and_save(src, dst)
    print("Saved" if ok else "Skipped/unreadable")
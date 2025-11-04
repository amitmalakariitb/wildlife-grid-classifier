"""
Enhanced feature extraction for wildlife grid cells (100x75 BGR)
Features:
- LBP histogram (uniform)
- HOG descriptor
- HSV color histograms (16 bins per channel)
- Haralick texture features (GLCM)
- Gabor filter bank responses (5 scales × 8 orientations)
- Color moments (mean, std, skewness per channel)
- Sobel + Canny edge metrics
- Entropy of grayscale image
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog, greycomatrix, greycoprops
from scipy.stats import skew
from skimage.filters import gabor
from skimage.measure import shannon_entropy

# --- Parameters ---
LBP_P = 8
LBP_R = 1
HOG_ORIENT = 9
HOG_PIXELS = (10, 5)
HOG_BLOCK = (2, 2)

def extract_features_from_cell(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    gray_f = gray.astype(np.float32) / 255.0

    # --- 1. LBP (micro texture) ---
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method='uniform').astype(int).ravel()
    n_bins = int(lbp.max() + 1)
    hist_lbp = np.bincount(lbp, minlength=n_bins).astype(float)
    hist_lbp /= (hist_lbp.sum() + 1e-9)

    # --- 2. HOG (shape/orientation) ---
    hog_feat = hog(gray, orientations=HOG_ORIENT, pixels_per_cell=HOG_PIXELS,
                   cells_per_block=HOG_BLOCK, block_norm='L2-Hys', feature_vector=True)

    # --- 3. HSV color histograms (global color) ---
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    color_hist = np.concatenate([h_hist, s_hist, v_hist])
    color_hist /= (color_hist.sum() + 1e-9)

    # --- 4. Haralick (macro texture) ---
    glcm = greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    haralick_feats = np.hstack([
        greycoprops(glcm, 'contrast').mean(),
        greycoprops(glcm, 'correlation').mean(),
        greycoprops(glcm, 'energy').mean(),
        greycoprops(glcm, 'homogeneity').mean()
    ])

    # --- 5. Gabor filter responses ---
    gabor_feats = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        for freq in (0.1, 0.2, 0.3, 0.4, 0.5):
            filt_real, _ = gabor(gray_f, frequency=freq, theta=theta)
            gabor_feats.extend([filt_real.mean(), filt_real.var()])
    gabor_feats = np.array(gabor_feats)

    # --- 6. Color moments ---
    color_moments = []
    for i in range(3):
        ch = cell_bgr[:, :, i].ravel().astype(np.float32)
        color_moments.extend([ch.mean(), ch.std(), skew(ch)])

    # --- 7. Edge and gradient metrics ---
    sx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    sobel_sum = np.sum(mag)
    sobel_mean = np.mean(mag)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.count_nonzero(edges)
    edge_ratio = edge_count / (gray.size + 1e-9)
    edge_feats = np.array([sobel_sum, sobel_mean, edge_count, edge_ratio])

    # --- 8. Entropy (information content) ---
    entropy = np.array([shannon_entropy(gray)])

    # --- Combine all features ---
    feature_vec = np.hstack([
        hist_lbp, hog_feat, color_hist, haralick_feats,
        gabor_feats, color_moments, edge_feats, entropy
    ]).astype(np.float32)

    return feature_vec

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <cell_image>")
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    f = extract_features_from_cell(img)
    print("Feature length:", len(f))

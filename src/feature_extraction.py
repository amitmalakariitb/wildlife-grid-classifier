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
from scipy.stats import skew
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from skimage.measure import shannon_entropy

# robust import for GLCM functions (different skimage versions)
try:
    from skimage.feature import graycomatrix, graycoprops
except Exception:
    try:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
    except Exception:
        graycomatrix = None
        graycoprops = None

# --- Parameters ---
LBP_P = 8
LBP_R = 1
# For 'uniform' LBP the number of bins = P + 2
LBP_N_BINS = LBP_P + 2

HOG_ORIENT = 9
HOG_PIXELS = (10, 5)  # divides 100x75
HOG_BLOCK = (2, 2)

def extract_features_from_cell(cell_bgr):
    """
    Input: 100x75 BGR cell
    Returns: 1D numpy float32 feature vector
    """
    # ensure uint8
    cell = cell_bgr.astype(np.uint8)
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    gray_f = gray.astype(np.float32) / 255.0

    # --- 1. LBP histogram (uniform) ---
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method='uniform').astype(int).ravel()
    # fixed-length histogram
    hist_lbp = np.bincount(lbp, minlength=LBP_N_BINS).astype(np.float32)
    hist_lbp /= (hist_lbp.sum() + 1e-9)

    # --- 2. HOG ---
    hog_feat = hog(gray, orientations=HOG_ORIENT, pixels_per_cell=HOG_PIXELS,
                   cells_per_block=HOG_BLOCK, block_norm='L2-Hys', visualize=False, feature_vector=True)
    hog_feat = np.asarray(hog_feat, dtype=np.float32)

    # --- 3. HSV color histograms (16 bins each) ---
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    color_hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    color_hist /= (color_hist.sum() + 1e-9)

    # --- 4. Haralick (GLCM) features: contrast, correlation, energy, homogeneity (mean over angles) ---
    if graycomatrix is not None:
        try:
            glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, symmetric=True, normed=True)
            c = np.mean(graycoprops(glcm, 'contrast'))
            corr = np.mean(graycoprops(glcm, 'correlation'))
            eng = np.mean(graycoprops(glcm, 'energy'))
            homog = np.mean(graycoprops(glcm, 'homogeneity'))
            haralick_feats = np.array([c, corr, eng, homog], dtype=np.float32)
        except Exception:
            haralick_feats = np.zeros(4, dtype=np.float32)
    else:
        haralick_feats = np.zeros(4, dtype=np.float32)

    # --- 5. Gabor filter bank responses (few frequencies × 8 orientations) ---
    gabor_feats = []
    freqs = [0.1, 0.2, 0.3]
    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    for freq in freqs:
        for theta in thetas:
            try:
                real, imag = gabor(gray_f, frequency=freq, theta=theta)
                mag = np.sqrt(real**2 + imag**2)
                gabor_feats.append(np.mean(mag))
                gabor_feats.append(np.std(mag))
            except Exception:
                gabor_feats.extend([0.0, 0.0])
    gabor_feats = np.array(gabor_feats, dtype=np.float32)

    # --- 6. Color moments: mean, std, skew per channel (B,G,R) ---
    means = cell.mean(axis=(0, 1)).astype(np.float32)  # BGR means
    stds = cell.std(axis=(0, 1)).astype(np.float32)
    # skew per channel
    skws = np.array([skew(cell[:, :, ch].ravel()) if cell[:, :, ch].size > 0 else 0.0 for ch in range(3)], dtype=np.float32)
    color_moments = np.hstack([means, stds, skws]).astype(np.float32)

    # --- 7. Edge and gradient metrics ---
    sx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    sobel_sum = np.sum(mag).astype(np.float32)
    sobel_mean = np.mean(mag).astype(np.float32)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = int(np.count_nonzero(edges))
    edge_ratio = np.float32(edge_count) / (gray.shape[0] * gray.shape[1])

    edge_feats = np.array([sobel_sum, sobel_mean, edge_count, edge_ratio], dtype=np.float32)

    # --- 8. Entropy ---
    try:
        ent = np.float32(shannon_entropy(gray))
    except Exception:
        ent = np.float32(0.0)

    # Concatenate all features into single vector
    feat = np.hstack([
        hist_lbp,
        hog_feat,
        color_hist,
        haralick_feats,
        gabor_feats,
        color_moments,
        edge_feats,
        np.array([ent], dtype=np.float32)
    ]).astype(np.float32)

    return feat

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <cell_image>")
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    fv = extract_features_from_cell(img)
    print("Feature length:", fv.shape[0])
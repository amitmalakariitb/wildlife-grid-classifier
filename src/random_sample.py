import os
import random
import shutil
import sys
import config

SRC = config.PREPROCESSED_DIR
DST = config.TEST_IMAGES_DIR
DEFAULT_N = 36

# optional CLI arg: sample size
if len(sys.argv) > 1:
    try:
        REQUESTED = int(sys.argv[1])
    except Exception:
        REQUESTED = DEFAULT_N
else:
    REQUESTED = DEFAULT_N

os.makedirs(DST, exist_ok=True)

if not os.path.isdir(SRC):
    print("Source folder not found:", SRC)
    print("Run batch_preprocess_images.py first or place images into", SRC)
    sys.exit(1)

all_imgs = [f for f in sorted(os.listdir(SRC)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
n = len(all_imgs)
if n == 0:
    print("No images found in", SRC)
    sys.exit(1)

sample_n = min(REQUESTED, n)
sampled = random.sample(all_imgs, sample_n)
for f in sampled:
    shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))

print(f"✅ Copied {len(sampled)} images to {DST} (requested {REQUESTED}, available {n})")
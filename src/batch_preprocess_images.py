import os
from tqdm import tqdm
import preprocess
import config

INPUT_DIR = config.IMAGES_DIR
OUTPUT_DIR = config.PREPROCESSED_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

def batch_preprocess():
    if not os.path.isdir(INPUT_DIR):
        print(f"Input folder not found: {INPUT_DIR}")
        print("Create the folder and add raw images, or verify config.IMAGES_DIR.")
        return [], []
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    processed = []
    skipped = []
    for fname in tqdm(files, desc="Preprocessing"):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        try:
            ok = preprocess.preprocess_and_save(in_path, out_path)
            if ok:
                processed.append(fname)
            else:
                skipped.append(fname)
        except Exception as e:
            print("Error processing", fname, e)
            skipped.append(fname)
    print(f"Preprocessed images saved to {OUTPUT_DIR} ({len(processed)} saved, {len(skipped)} skipped)")
    if skipped:
        print("First skipped files (max 20):")
        for s in skipped[:20]:
            print(" ", s)
    return processed, skipped

if __name__ == "__main__":
    batch_preprocess()
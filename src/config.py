import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LABELS_CSV = os.path.join(OUTPUT_DIR, "labels.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

MODEL_PIPELINE = os.path.join(MODELS_DIR, "model_pipeline.pkl")

# ensure directories exist
for p in (DATA_DIR, PREPROCESSED_DIR, TEST_IMAGES_DIR, OUTPUT_DIR, MODELS_DIR, ANNOTATED_DIR):
    os.makedirs(p, exist_ok=True)
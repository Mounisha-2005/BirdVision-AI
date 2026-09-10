import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_bird_cnn.keras")
CENTROIDS_PATH = os.path.join(BASE_DIR, "models", "class_centroids.npy")

KNOWN_DIR = os.path.join(BASE_DIR, "test_images")
UNKNOWN_DIR = os.path.join(BASE_DIR, "test_unknown")


def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


print("=" * 70)
print("OPEN-SET THRESHOLD CALIBRATION")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

centroids = np.load(CENTROIDS_PATH)

print("Model loaded.")
print(f"Centroids: {centroids.shape}")


# ------------------------------------------------------------
# FEATURE EXTRACTOR
# ------------------------------------------------------------

feature_layer = model.get_layer("dense")

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------

def preprocess(path):

    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(
        image,
        dtype=np.float32
    )

    return np.expand_dims(image, axis=0)


# ------------------------------------------------------------
# COSINE SIMILARITY
# ------------------------------------------------------------

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ------------------------------------------------------------
# GET SIMILARITY
# ------------------------------------------------------------

def get_similarity(path):

    image = preprocess(path)

    features = feature_extractor.predict(
        image,
        verbose=0
    )[0]

    similarities = [
        cosine_similarity(features, centroid)
        for centroid in centroids
    ]

    return max(similarities)


# ------------------------------------------------------------
# COLLECT IMAGES
# ------------------------------------------------------------

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


known_images = [
    os.path.join(KNOWN_DIR, f)
    for f in os.listdir(KNOWN_DIR)
    if f.lower().endswith(extensions)
]


unknown_images = [
    os.path.join(UNKNOWN_DIR, f)
    for f in os.listdir(UNKNOWN_DIR)
    if f.lower().endswith(extensions)
]


# ------------------------------------------------------------
# CALCULATE SIMILARITIES
# ------------------------------------------------------------

print("\nCalculating known-image similarities...")

known_scores = []

for path in known_images:

    score = get_similarity(path)

    known_scores.append(score)

    print(
        f"KNOWN   {os.path.basename(path):35s}"
        f" {score:.4f}"
    )


print("\nCalculating unknown-image similarities...")

unknown_scores = []

for path in unknown_images:

    score = get_similarity(path)

    unknown_scores.append(score)

    print(
        f"UNKNOWN {os.path.basename(path):35s}"
        f" {score:.4f}"
    )


known_scores = np.array(known_scores)
unknown_scores = np.array(unknown_scores)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("SIMILARITY SUMMARY")
print("=" * 70)

print(
    f"Known minimum : {known_scores.min():.4f}"
)

print(
    f"Known maximum : {known_scores.max():.4f}"
)

print(
    f"Known mean    : {known_scores.mean():.4f}"
)

print(
    f"Unknown minimum : {unknown_scores.min():.4f}"
)

print(
    f"Unknown maximum : {unknown_scores.max():.4f}"
)

print(
    f"Unknown mean    : {unknown_scores.mean():.4f}"
)


# ------------------------------------------------------------
# THRESHOLD TESTING
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

print(
    "\nThreshold | Known Accepted | Unknown Rejected | "
    "Known Reject | Unknown Accept"
)

print("-" * 70)


best_threshold = None
best_score = -1


thresholds = np.arange(
    0.60,
    0.96,
    0.01
)


for threshold in thresholds:

    known_accepted = np.sum(
        known_scores >= threshold
    )

    unknown_rejected = np.sum(
        unknown_scores < threshold
    )

    known_rejected = len(known_scores) - known_accepted

    unknown_accepted = len(unknown_scores) - unknown_rejected

    # Balanced accuracy
    known_rate = known_accepted / len(known_scores)

    unknown_rate = unknown_rejected / len(unknown_scores)

    balanced_score = (
        known_rate + unknown_rate
    ) / 2

    print(
        f"{threshold:.2f}      | "
        f"{known_accepted:14d} | "
        f"{unknown_rejected:16d} | "
        f"{known_rejected:11d} | "
        f"{unknown_accepted:13d}"
    )

    if balanced_score > best_score:

        best_score = balanced_score
        best_threshold = threshold


# ------------------------------------------------------------
# BEST THRESHOLD
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Recommended threshold : {best_threshold:.2f}"
)

print(
    f"Balanced score         : "
    f"{best_score * 100:.2f}%"
)


known_accepted = np.sum(
    known_scores >= best_threshold
)

unknown_rejected = np.sum(
    unknown_scores < best_threshold
)

print(
    f"\nKnown accepted         : "
    f"{known_accepted}/{len(known_scores)}"
)

print(
    f"Unknown rejected       : "
    f"{unknown_rejected}/{len(unknown_scores)}"
)


print("\nCalibration completed.")
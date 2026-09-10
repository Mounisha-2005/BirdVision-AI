import os
import json
import numpy as np
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

VALID_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "valid"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bird_cnn.keras"
)

CENTROIDS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_centroids.npy"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "unknown_threshold.json"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("CALIBRATING UNKNOWN-BIRD DETECTOR")
print("=" * 70)

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

# Repair Lambda layer
try:
    augmentation_model = model.get_layer("data_augmentation")
    clip_layer = augmentation_model.layers[-1]

    if isinstance(clip_layer, tf.keras.layers.Lambda):
        clip_layer.function = clip_to_unit_range

except Exception:
    pass

print("Model loaded successfully.")


# ============================================================
# LOAD CENTROIDS
# ============================================================

centroids = np.load(CENTROIDS_PATH)

print("Centroids loaded:", centroids.shape)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation images:", len(valid_ds.file_paths))


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_layer = model.layers[-3]

feature_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


# ============================================================
# CALCULATE VALIDATION SIMILARITIES
# ============================================================

similarities = []

print("\nCalculating validation similarities...")

for images, labels in valid_ds:

    features = feature_model(
        images,
        training=False
    ).numpy()

    # Normalize feature vectors
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    features = features / np.maximum(norms, 1e-8)

    # Cosine similarity against every class centroid
    similarity_matrix = np.matmul(
        features,
        centroids.T
    )

    # Similarity to the TRUE class
    batch_indices = np.arange(len(labels))

    true_class_similarity = similarity_matrix[
        batch_indices,
        labels.numpy()
    ]

    similarities.extend(
        true_class_similarity.tolist()
    )


similarities = np.array(similarities)


# ============================================================
# STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SIMILARITY RESULTS")
print("=" * 70)

print(f"Minimum similarity : {similarities.min():.4f}")
print(f"Maximum similarity : {similarities.max():.4f}")
print(f"Mean similarity    : {similarities.mean():.4f}")
print(f"Median similarity  : {np.median(similarities):.4f}")
print(f"5th percentile     : {np.percentile(similarities, 5):.4f}")
print(f"10th percentile    : {np.percentile(similarities, 10):.4f}")


# ============================================================
# INITIAL THRESHOLD
# ============================================================

# We use the 5th percentile as an initial threshold.
# This allows approximately 95% of validation images
# to remain accepted.

threshold = float(
    np.percentile(similarities, 5)
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_data = {
    "threshold": threshold,
    "method": "5th_percentile_validation_similarity",
    "validation_images": int(len(similarities)),
    "minimum_similarity": float(similarities.min()),
    "maximum_similarity": float(similarities.max()),
    "mean_similarity": float(similarities.mean()),
    "median_similarity": float(np.median(similarities))
}

with open(THRESHOLD_PATH, "w") as f:
    json.dump(
        threshold_data,
        f,
        indent=4
    )


print("\nInitial threshold:", f"{threshold:.4f}")

print("\nSaved:")
print(THRESHOLD_PATH)

print("\n" + "=" * 70)
print("THRESHOLD CALIBRATION COMPLETED")
print("=" * 70)
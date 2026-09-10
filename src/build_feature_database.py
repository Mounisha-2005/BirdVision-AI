import os
import json
import numpy as np
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "train"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bird_cnn.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

CENTROIDS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_centroids.npy"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("BUILDING BIRD FEATURE DATABASE")
print("=" * 70)

print("\nLoading trained model...")

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

# Repair Lambda layer after loading
try:
    augmentation_model = model.get_layer("data_augmentation")
    clip_layer = augmentation_model.layers[-1]

    if isinstance(clip_layer, tf.keras.layers.Lambda):
        clip_layer.function = clip_to_unit_range

except Exception:
    pass

print("Model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Number of classes: {len(class_names)}")


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training images...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training images: {len(train_ds.file_paths)}")


# ============================================================
# CREATE FEATURE EXTRACTOR
# ============================================================

# Architecture:
# ...
# GlobalAveragePooling
# Dense(128)
# Dropout
# Dense(20)

# We want the 128-dimensional feature representation
feature_layer = model.layers[-3]

print("\nFeature layer:")
print(feature_layer.name)
print(feature_layer.output_shape if hasattr(feature_layer, "output_shape") else "")

feature_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

print("\nExtracting features...")
print("This may take a few minutes on CPU.")

features = []
labels = []

for batch_images, batch_labels in train_ds:

    batch_features = feature_model(
        batch_images,
        training=False
    ).numpy()

    features.append(batch_features)
    labels.append(batch_labels.numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

print("\nFeature extraction completed.")

print("Feature shape:", features.shape)
print("Labels shape:", labels.shape)


# ============================================================
# NORMALIZE FEATURES
# ============================================================

norms = np.linalg.norm(
    features,
    axis=1,
    keepdims=True
)

features_normalized = features / np.maximum(norms, 1e-8)


# ============================================================
# CALCULATE CLASS CENTROIDS
# ============================================================

print("\nCalculating class centroids...")

num_classes = len(class_names)

centroids = []

for class_index in range(num_classes):

    class_features = features_normalized[
        labels == class_index
    ]

    centroid = np.mean(
        class_features,
        axis=0
    )

    centroid_norm = np.linalg.norm(centroid)

    centroid = centroid / max(
        centroid_norm,
        1e-8
    )

    centroids.append(centroid)

centroids = np.array(centroids)


# ============================================================
# SAVE CENTROIDS
# ============================================================

np.save(
    CENTROIDS_PATH,
    centroids
)

print("\nFeature database saved successfully!")

print("\nSaved file:")
print(CENTROIDS_PATH)

print("\nCentroid shape:")
print(centroids.shape)

print("\n" + "=" * 70)
print("FEATURE DATABASE CREATION COMPLETED")
print("=" * 70)
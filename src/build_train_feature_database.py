import os
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "train_features.npy"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "train_labels.npy"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# CUSTOM FUNCTION USED BY MODEL
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("BUILDING TRAINING FEATURE DATABASE")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

print("Model loaded successfully.")


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("dense").output
)

print("Feature layer : dense")
print("Feature size  : 128")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = sorted([
    name
    for name in os.listdir(TRAIN_DIR)
    if os.path.isdir(
        os.path.join(TRAIN_DIR, name)
    )
])

print("\nNumber of classes:", len(class_names))


# ============================================================
# COLLECT TRAINING IMAGES
# ============================================================

image_paths = []
labels = []

for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    for filename in sorted(os.listdir(class_dir)):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            continue

        image_path = os.path.join(
            class_dir,
            filename
        )

        image_paths.append(image_path)
        labels.append(class_index)


print("Training images:", len(image_paths))


# ============================================================
# FEATURE EXTRACTION
# ============================================================

all_features = []

batch_images = []

for i, image_path in enumerate(image_paths):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            IMAGE_SIZE
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        batch_images.append(
            image_array
        )

        # Process batch
        if (
            len(batch_images) == BATCH_SIZE
            or i == len(image_paths) - 1
        ):

            batch = np.array(
                batch_images,
                dtype=np.float32
            )

            features = feature_extractor.predict(
                batch,
                verbose=0
            )

            all_features.append(
                features
            )

            batch_images = []

        if (i + 1) % 320 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(image_paths)} images"
            )

    except Exception as e:

        print(
            f"\nERROR processing:"
            f"\n{image_path}"
            f"\nReason: {e}"
        )


# ============================================================
# COMBINE FEATURES
# ============================================================

features = np.concatenate(
    all_features,
    axis=0
)

labels = np.array(
    labels,
    dtype=np.int32
)


# ============================================================
# L2 NORMALIZATION
# ============================================================

norms = np.linalg.norm(
    features,
    axis=1,
    keepdims=True
)

features = features / np.maximum(
    norms,
    1e-8
)


# ============================================================
# SAVE
# ============================================================

np.save(
    FEATURES_PATH,
    features
)

np.save(
    LABELS_PATH,
    labels
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DATABASE CREATED")
print("=" * 70)

print(
    "Feature shape :",
    features.shape
)

print(
    "Label shape   :",
    labels.shape
)

print("\nSaved files:")

print(
    FEATURES_PATH
)

print(
    LABELS_PATH
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
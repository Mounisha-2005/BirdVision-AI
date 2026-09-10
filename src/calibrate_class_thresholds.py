import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_bird_cnn.keras"
)

CENTROIDS_PATH = os.path.join(
    BASE_DIR, "models", "class_centroids.npy"
)

VALID_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "valid"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
)


def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


print("=" * 70)
print("CLASS-SPECIFIC OPEN-SET CALIBRATION")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

centroids = np.load(CENTROIDS_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

feature_layer = model.get_layer("dense")

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


def preprocess_image(path):

    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(
        image,
        dtype=np.float32
    )

    return np.expand_dims(image, axis=0)


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


thresholds = {}

print("\nCalculating thresholds...\n")

for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        VALID_DIR,
        class_name
    )

    scores = []

    for filename in os.listdir(class_dir):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            continue

        path = os.path.join(
            class_dir,
            filename
        )

        image = preprocess_image(path)

        features = feature_extractor.predict(
            image,
            verbose=0
        )[0]

        similarity = cosine_similarity(
            features,
            centroids[class_index]
        )

        scores.append(similarity)

    scores = np.array(scores)

    # Conservative threshold:
    # retain all validation examples while
    # leaving a small safety margin.
    minimum = float(scores.min())

    threshold = max(
        0.60,
        minimum - 0.02
    )

    thresholds[class_name] = threshold

    print(
        f"{class_name:35s} "
        f"min={minimum:.4f} "
        f"mean={scores.mean():.4f} "
        f"threshold={threshold:.4f}"
    )


output_path = os.path.join(
    BASE_DIR,
    "models",
    "class_specific_thresholds.json"
)

with open(output_path, "w") as f:

    json.dump(
        thresholds,
        f,
        indent=4
    )


print("\n" + "=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(output_path)
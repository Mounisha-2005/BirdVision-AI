import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

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
    "class_specific_thresholds.json"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

KNOWN_DIR = os.path.join(
    BASE_DIR,
    "test_images"
)

UNKNOWN_DIR = os.path.join(
    BASE_DIR,
    "test_unknown"
)


# ============================================================
# CUSTOM FUNCTION USED BY SAVED MODEL
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("CLASS-SPECIFIC UNKNOWN IMAGE DETECTOR")
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
# LOAD CENTROIDS
# ============================================================

centroids = np.load(CENTROIDS_PATH)

print(
    f"Centroids shape: {centroids.shape}"
)


# ============================================================
# LOAD CLASS-SPECIFIC THRESHOLDS
# ============================================================

with open(THRESHOLD_PATH, "r") as f:
    class_thresholds = json.load(f)

print(
    f"Class-specific thresholds loaded: "
    f"{len(class_thresholds)}"
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(
    f"Number of classes: {len(class_names)}"
)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_layer = model.get_layer("dense")

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# TEST ONE IMAGE
# ============================================================

def test_image(
    image_path,
    expected_type
):

    image = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )


    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    features = feature_extractor.predict(
        image,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # COMPARE WITH ALL CLASS CENTROIDS
    # --------------------------------------------------------

    similarities = []

    for centroid in centroids:

        similarity = cosine_similarity(
            features,
            centroid
        )

        similarities.append(
            similarity
        )

    similarities = np.array(
        similarities
    )


    # --------------------------------------------------------
    # CLOSEST CENTROID
    # --------------------------------------------------------

    closest_class_index = int(
        np.argmax(similarities)
    )

    closest_class = class_names[
        closest_class_index
    ]

    max_similarity = float(
        similarities[
            closest_class_index
        ]
    )


    # --------------------------------------------------------
    # GET THRESHOLD FOR CLOSEST CLASS
    # --------------------------------------------------------

    class_threshold = float(
        class_thresholds[
            closest_class
        ]
    )


    # --------------------------------------------------------
    # CHECK CNN / CENTROID AGREEMENT
    # --------------------------------------------------------

    agreement = (
        predicted_index
        ==
        closest_class_index
    )


    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------
    #
    # Supported only when:
    #
    # 1. Feature similarity reaches the
    #    threshold of the closest class
    #
    # 2. CNN prediction agrees with the
    #    closest centroid
    #
    # --------------------------------------------------------

    if (
        max_similarity >= class_threshold
        and
        agreement
    ):

        decision = "SUPPORTED"

    else:

        decision = "UNKNOWN"


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"Image              : "
        f"{os.path.basename(image_path)}"
    )

    print(
        f"Expected type      : "
        f"{expected_type}"
    )

    print(
        f"CNN prediction     : "
        f"{predicted_class}"
    )

    print(
        f"CNN confidence     : "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Closest class      : "
        f"{closest_class}"
    )

    print(
        f"Feature similarity : "
        f"{max_similarity:.4f}"
    )

    print(
        f"Class threshold    : "
        f"{class_threshold:.4f}"
    )

    print(
        f"CNN/centroid agree : "
        f"{agreement}"
    )

    print(
        f"Decision            : "
        f"{decision}"
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    if expected_type == "KNOWN":

        if decision == "SUPPORTED":

            print(
                "RESULT             : "
                "CORRECT"
            )

        else:

            print(
                "RESULT             : "
                "FALSE REJECTION"
            )

    elif expected_type == "UNKNOWN":

        if decision == "UNKNOWN":

            print(
                "RESULT             : "
                "CORRECTLY REJECTED"
            )

        else:

            print(
                "RESULT             : "
                "FALSE ACCEPTANCE"
            )


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


known_images = []

if os.path.exists(KNOWN_DIR):

    for filename in sorted(
        os.listdir(KNOWN_DIR)
    ):

        if filename.lower().endswith(
            image_extensions
        ):

            known_images.append(
                os.path.join(
                    KNOWN_DIR,
                    filename
                )
            )


unknown_images = []

if os.path.exists(UNKNOWN_DIR):

    for filename in sorted(
        os.listdir(UNKNOWN_DIR)
    ):

        if filename.lower().endswith(
            image_extensions
        ):

            unknown_images.append(
                os.path.join(
                    UNKNOWN_DIR,
                    filename
                )
            )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    f"Known images found   : "
    f"{len(known_images)}"
)

print(
    f"Unknown images found : "
    f"{len(unknown_images)}"
)

print("=" * 70)


# ============================================================
# TEST KNOWN IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("TESTING KNOWN IMAGES")
print("=" * 70)

for image_path in known_images:

    test_image(
        image_path,
        "KNOWN"
    )


# ============================================================
# TEST UNKNOWN IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("TESTING UNKNOWN IMAGES")
print("=" * 70)

for image_path in unknown_images:

    test_image(
        image_path,
        "UNKNOWN"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("CLASS-SPECIFIC UNKNOWN IMAGE TESTING COMPLETED")
print("=" * 70)
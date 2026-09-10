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

VALID_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "valid"
)

UNKNOWN_DIR = os.path.join(
    BASE_DIR,
    "test_unknown"
)


# ============================================================
# MODEL LOADING
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


print("=" * 75)
print("OPEN-SET DETECTOR - FULL CALIBRATION")
print("=" * 75)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

centroids = np.load(CENTROIDS_PATH)

print("Model loaded successfully.")
print(f"Centroids shape: {centroids.shape}")


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_layer = model.get_layer("dense")

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def preprocess_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    return np.expand_dims(
        image,
        axis=0
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

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


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path):

    image = preprocess_image(path)

    # CNN prediction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    top1_index = sorted_indices[0]
    top2_index = sorted_indices[1]

    confidence = float(
        probabilities[top1_index]
    )

    top2_confidence = float(
        probabilities[top2_index]
    )

    margin = (
        confidence
        - top2_confidence
    )

    # Feature extraction
    features = feature_extractor.predict(
        image,
        verbose=0
    )[0]

    # Similarity to all class centroids
    similarities = np.array([
        cosine_similarity(
            features,
            centroid
        )
        for centroid in centroids
    ])

    nearest_class_index = int(
        np.argmax(similarities)
    )

    max_similarity = float(
        similarities[nearest_class_index]
    )

    prediction_matches_nearest = (
        top1_index == nearest_class_index
    )

    return {
        "confidence": confidence,
        "margin": margin,
        "similarity": max_similarity,
        "prediction_class": int(top1_index),
        "nearest_class": nearest_class_index,
        "agreement": prediction_matches_nearest
    }


# ============================================================
# FIND VALIDATION IMAGES
# ============================================================

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


validation_images = []

for root, dirs, files in os.walk(VALID_DIR):

    for file in files:

        if file.lower().endswith(
            extensions
        ):

            validation_images.append(
                os.path.join(
                    root,
                    file
                )
            )


unknown_images = []

for file in os.listdir(UNKNOWN_DIR):

    if file.lower().endswith(
        extensions
    ):

        unknown_images.append(
            os.path.join(
                UNKNOWN_DIR,
                file
            )
        )


print(
    f"\nValidation images: "
    f"{len(validation_images)}"
)

print(
    f"Unknown images: "
    f"{len(unknown_images)}"
)


# ============================================================
# ANALYZE VALIDATION DATA
# ============================================================

print("\nAnalyzing validation images...")

known_results = []

for i, path in enumerate(
    validation_images,
    start=1
):

    result = analyze_image(path)

    known_results.append(result)

    if i % 20 == 0:

        print(
            f"Processed {i}/"
            f"{len(validation_images)}"
        )


# ============================================================
# ANALYZE UNKNOWN DATA
# ============================================================

print("\nAnalyzing unknown images...")

unknown_results = []

for i, path in enumerate(
    unknown_images,
    start=1
):

    result = analyze_image(path)

    unknown_results.append(result)

    print(
        f"{os.path.basename(path):35s} "
        f"confidence={result['confidence']:.4f} "
        f"similarity={result['similarity']:.4f} "
        f"margin={result['margin']:.4f} "
        f"agreement={result['agreement']}"
    )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

known_confidence = np.array([
    x["confidence"]
    for x in known_results
])

known_similarity = np.array([
    x["similarity"]
    for x in known_results
])

known_margin = np.array([
    x["margin"]
    for x in known_results
])

known_agreement = np.array([
    x["agreement"]
    for x in known_results
])


unknown_confidence = np.array([
    x["confidence"]
    for x in unknown_results
])

unknown_similarity = np.array([
    x["similarity"]
    for x in unknown_results
])

unknown_margin = np.array([
    x["margin"]
    for x in unknown_results
])

unknown_agreement = np.array([
    x["agreement"]
    for x in unknown_results
])


# ============================================================
# DISTRIBUTION SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("KNOWN VALIDATION DISTRIBUTION")
print("=" * 75)

print(
    f"Confidence minimum : "
    f"{known_confidence.min():.4f}"
)

print(
    f"Confidence 5%      : "
    f"{np.percentile(known_confidence, 5):.4f}"
)

print(
    f"Confidence mean    : "
    f"{known_confidence.mean():.4f}"
)

print(
    f"Similarity minimum : "
    f"{known_similarity.min():.4f}"
)

print(
    f"Similarity 5%      : "
    f"{np.percentile(known_similarity, 5):.4f}"
)

print(
    f"Similarity mean    : "
    f"{known_similarity.mean():.4f}"
)

print(
    f"Margin minimum     : "
    f"{known_margin.min():.4f}"
)

print(
    f"Margin 5%          : "
    f"{np.percentile(known_margin, 5):.4f}"
)

print(
    f"Agreement          : "
    f"{known_agreement.sum()}/"
    f"{len(known_agreement)}"
)


print("\n")
print("=" * 75)
print("UNKNOWN DISTRIBUTION")
print("=" * 75)

print(
    f"Confidence minimum : "
    f"{unknown_confidence.min():.4f}"
)

print(
    f"Confidence maximum : "
    f"{unknown_confidence.max():.4f}"
)

print(
    f"Confidence mean    : "
    f"{unknown_confidence.mean():.4f}"
)

print(
    f"Similarity minimum : "
    f"{unknown_similarity.min():.4f}"
)

print(
    f"Similarity maximum : "
    f"{unknown_similarity.max():.4f}"
)

print(
    f"Similarity mean    : "
    f"{unknown_similarity.mean():.4f}"
)

print(
    f"Margin minimum     : "
    f"{unknown_margin.min():.4f}"
)

print(
    f"Margin maximum     : "
    f"{unknown_margin.max():.4f}"
)

print(
    f"Agreement          : "
    f"{unknown_agreement.sum()}/"
    f"{len(unknown_agreement)}"
)


# ============================================================
# COMBINED RULE SEARCH
# ============================================================

print("\n")
print("=" * 75)
print("COMBINED OPEN-SET RULE SEARCH")
print("=" * 75)

print(
    "\nRule:"
)

print(
    "SUPPORTED if:"
)

print(
    "confidence >= threshold"
)

print(
    "AND similarity >= threshold"
)

print(
    "AND CNN prediction agrees with nearest centroid"
)


best_rule = None
best_balanced = -1


confidence_thresholds = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

similarity_thresholds = np.arange(
    0.65,
    0.91,
    0.01
)


for conf_threshold in confidence_thresholds:

    for sim_threshold in similarity_thresholds:

        known_supported = (
            (known_confidence >= conf_threshold)
            &
            (known_similarity >= sim_threshold)
            &
            known_agreement
        )

        unknown_supported = (
            (unknown_confidence >= conf_threshold)
            &
            (unknown_similarity >= sim_threshold)
            &
            unknown_agreement
        )

        known_accept_rate = (
            np.mean(known_supported)
        )

        unknown_reject_rate = (
            1.0
            -
            np.mean(unknown_supported)
        )

        balanced_accuracy = (
            known_accept_rate
            +
            unknown_reject_rate
        ) / 2


        if balanced_accuracy > best_balanced:

            best_balanced = balanced_accuracy

            best_rule = {
                "confidence": conf_threshold,
                "similarity": float(sim_threshold),
                "known_accept": float(
                    known_accept_rate
                ),
                "unknown_reject": float(
                    unknown_reject_rate
                )
            }


# ============================================================
# BEST RULE
# ============================================================

print("\n")
print("=" * 75)
print("BEST COMBINED RULE")
print("=" * 75)

print(
    f"Confidence threshold : "
    f"{best_rule['confidence']:.2f}"
)

print(
    f"Similarity threshold : "
    f"{best_rule['similarity']:.2f}"
)

print(
    f"Known acceptance     : "
    f"{best_rule['known_accept'] * 100:.2f}%"
)

print(
    f"Unknown rejection    : "
    f"{best_rule['unknown_reject'] * 100:.2f}%"
)

print(
    f"Balanced accuracy    : "
    f"{best_balanced * 100:.2f}%"
)


# ============================================================
# SAVE RESULT
# ============================================================

output_path = os.path.join(
    BASE_DIR,
    "models",
    "open_set_calibration.json"
)

with open(
    output_path,
    "w"
) as f:

    json.dump(
        {
            "confidence_threshold":
                best_rule["confidence"],

            "similarity_threshold":
                best_rule["similarity"],

            "known_acceptance":
                best_rule["known_accept"],

            "unknown_rejection":
                best_rule["unknown_reject"],

            "balanced_accuracy":
                best_balanced
        },
        f,
        indent=4
    )


print(
    f"\nSaved calibration:"
)

print(output_path)

print("\nCalibration completed.")
import os
import json
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_bird_cnn.keras"
)

CENTROID_PATH = os.path.join(
    BASE_DIR, "models", "class_centroids.npy"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
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
    BASE_DIR, "test_unknown"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "optimized_open_set.json"
)

IMG_SIZE = (224, 224)


# ============================================================
# CUSTOM FUNCTION FOR SAVED MODEL
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("OPEN-SET DETECTOR OPTIMIZATION")
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

centroids = np.load(CENTROID_PATH)

print("Centroids shape:", centroids.shape)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Number of classes:", len(class_names))


# ============================================================
# FIND FEATURE LAYER
# ============================================================

feature_layer = model.get_layer("dense")

feature_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)

print("Feature layer:", feature_layer.name)
print("Feature size:", feature_layer.output.shape[-1])


# ============================================================
# IMAGE PROCESSING
# ============================================================

def load_image(path):

    image = load_img(
        path,
        target_size=IMG_SIZE
    )

    image = img_to_array(image).astype(
        np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# EXTRACT INFORMATION FROM ONE IMAGE
# ============================================================

def analyze_image(path):

    image = load_image(path)

    # CNN prediction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    sorted_probs = np.sort(
        probabilities
    )[::-1]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        sorted_probs[0]
    )

    second_confidence = float(
        sorted_probs[1]
    )

    margin = (
        confidence -
        second_confidence
    )

    # Feature extraction
    feature = feature_model.predict(
        image,
        verbose=0
    )[0]

    # Normalize feature
    feature_norm = (
        feature /
        (np.linalg.norm(feature) + 1e-8)
    )

    # Normalize centroids
    centroid_norms = (
        centroids /
        (
            np.linalg.norm(
                centroids,
                axis=1,
                keepdims=True
            ) + 1e-8
        )
    )

    # Cosine similarity
    similarities = np.dot(
        centroid_norms,
        feature_norm
    )

    closest_index = int(
        np.argmax(similarities)
    )

    max_similarity = float(
        similarities[closest_index]
    )

    agreement = (
        predicted_index ==
        closest_index
    )

    return {
        "predicted_index": predicted_index,
        "confidence": confidence,
        "margin": float(margin),
        "closest_index": closest_index,
        "similarity": max_similarity,
        "agreement": agreement
    }


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nCollecting validation images...")

known_samples = []

for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        VALID_DIR,
        class_name
    )

    if not os.path.isdir(class_dir):
        continue

    for filename in os.listdir(class_dir):

        path = os.path.join(
            class_dir,
            filename
        )

        if not os.path.isfile(path):
            continue

        try:

            result = analyze_image(
                path
            )

            result["true_index"] = (
                class_index
            )

            result["label"] = 1

            known_samples.append(
                result
            )

        except Exception as e:

            print(
                "Skipped:",
                path,
                e
            )

print(
    "Validation samples:",
    len(known_samples)
)


# ============================================================
# LOAD UNKNOWN DATA
# ============================================================

print("\nCollecting unknown images...")

unknown_samples = []

for filename in sorted(
    os.listdir(UNKNOWN_DIR)
):

    path = os.path.join(
        UNKNOWN_DIR,
        filename
    )

    if not os.path.isfile(path):
        continue

    try:

        result = analyze_image(
            path
        )

        result["true_index"] = -1
        result["label"] = 0

        unknown_samples.append(
            result
        )

    except Exception as e:

        print(
            "Skipped:",
            path,
            e
        )

print(
    "Unknown samples:",
    len(unknown_samples)
)


# ============================================================
# CREATE ARRAYS
# ============================================================

all_samples = (
    known_samples +
    unknown_samples
)

y_true = np.array(
    [
        sample["label"]
        for sample in all_samples
    ]
)


# ============================================================
# PRINT BASIC STATISTICS
# ============================================================

known_conf = np.array(
    [
        x["confidence"]
        for x in known_samples
    ]
)

known_sim = np.array(
    [
        x["similarity"]
        for x in known_samples
    ]
)

known_margin = np.array(
    [
        x["margin"]
        for x in known_samples
    ]
)

unknown_conf = np.array(
    [
        x["confidence"]
        for x in unknown_samples
    ]
)

unknown_sim = np.array(
    [
        x["similarity"]
        for x in unknown_samples
    ]
)

unknown_margin = np.array(
    [
        x["margin"]
        for x in unknown_samples
    ]
)


print("\n" + "=" * 70)
print("DATA DISTRIBUTION")
print("=" * 70)

print("\nKNOWN VALIDATION")

print(
    "Confidence:",
    f"{known_conf.min():.4f}",
    "to",
    f"{known_conf.max():.4f}"
)

print(
    "Similarity:",
    f"{known_sim.min():.4f}",
    "to",
    f"{known_sim.max():.4f}"
)

print(
    "Margin:",
    f"{known_margin.min():.4f}",
    "to",
    f"{known_margin.max():.4f}"
)


print("\nUNKNOWN")

print(
    "Confidence:",
    f"{unknown_conf.min():.4f}",
    "to",
    f"{unknown_conf.max():.4f}"
)

print(
    "Similarity:",
    f"{unknown_sim.min():.4f}",
    "to",
    f"{unknown_sim.max():.4f}"
)

print(
    "Margin:",
    f"{unknown_margin.min():.4f}",
    "to",
    f"{unknown_margin.max():.4f}"
)


# ============================================================
# GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR BEST OPEN-SET RULE")
print("=" * 70)


confidence_thresholds = np.arange(
    0.30,
    0.96,
    0.05
)

similarity_thresholds = np.arange(
    0.55,
    0.91,
    0.02
)

margin_thresholds = np.arange(
    0.00,
    0.91,
    0.05
)


best_result = None


for conf_threshold in confidence_thresholds:

    for sim_threshold in similarity_thresholds:

        for margin_threshold in margin_thresholds:

            predictions = []

            for sample in all_samples:

                accepted = (

                    sample["confidence"]
                    >= conf_threshold

                    and

                    sample["similarity"]
                    >= sim_threshold

                    and

                    sample["margin"]
                    >= margin_threshold

                    and

                    sample["agreement"]
                )

                predictions.append(
                    1 if accepted else 0
                )

            predictions = np.array(
                predictions
            )

            known_mask = (
                y_true == 1
            )

            unknown_mask = (
                y_true == 0
            )

            known_acceptance = (
                predictions[
                    known_mask
                ].mean()
                if known_mask.any()
                else 0
            )

            unknown_rejection = (
                1 -
                predictions[
                    unknown_mask
                ].mean()
                if unknown_mask.any()
                else 0
            )

            balanced_accuracy = (
                known_acceptance +
                unknown_rejection
            ) / 2

            # Penalize very low known acceptance
            if known_acceptance < 0.80:
                continue

            # Penalize very poor unknown rejection
            if unknown_rejection < 0.50:
                continue

            score = balanced_accuracy

            if (
                best_result is None
                or score >
                best_result["balanced_accuracy"]
            ):

                best_result = {

                    "confidence_threshold":
                        float(conf_threshold),

                    "similarity_threshold":
                        float(sim_threshold),

                    "margin_threshold":
                        float(margin_threshold),

                    "known_acceptance":
                        float(known_acceptance),

                    "unknown_rejection":
                        float(unknown_rejection),

                    "false_rejection_rate":
                        float(
                            1 -
                            known_acceptance
                        ),

                    "false_acceptance_rate":
                        float(
                            1 -
                            unknown_rejection
                        ),

                    "balanced_accuracy":
                        float(
                            balanced_accuracy
                        )
                }


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST RULE FOUND")
print("=" * 70)

if best_result is None:

    print(
        "\nNo rule satisfied the minimum constraints."
    )

else:

    print(
        "\nConfidence threshold :",
        best_result[
            "confidence_threshold"
        ]
    )

    print(
        "Similarity threshold :",
        best_result[
            "similarity_threshold"
        ]
    )

    print(
        "Margin threshold     :",
        best_result[
            "margin_threshold"
        ]
    )

    print(
        "\nKnown acceptance      :",
        f"{best_result['known_acceptance'] * 100:.2f}%"
    )

    print(
        "Unknown rejection     :",
        f"{best_result['unknown_rejection'] * 100:.2f}%"
    )

    print(
        "False rejection rate  :",
        f"{best_result['false_rejection_rate'] * 100:.2f}%"
    )

    print(
        "False acceptance rate  :",
        f"{best_result['false_acceptance_rate'] * 100:.2f}%"
    )

    print(
        "Balanced accuracy     :",
        f"{best_result['balanced_accuracy'] * 100:.2f}%"
    )


# ============================================================
# SAVE RESULT
# ============================================================

output = {

    "method":
        "Confidence + Similarity + Margin + CNN/Centroid Agreement",

    "validation_images":
        len(known_samples),

    "unknown_images":
        len(unknown_samples),

    "best_rule":
        best_result
}


with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


print(
    "\nSaved optimization result:"
)

print(
    OUTPUT_PATH
)

print("\n" + "=" * 70)
print("OPEN-SET OPTIMIZATION COMPLETED")
print("=" * 70)
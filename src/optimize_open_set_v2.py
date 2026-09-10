import os
import json
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "optimized_open_set_v2.json"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# CUSTOM FUNCTION
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("OPEN-SET DETECTOR V2 OPTIMIZATION")
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

centroids = np.load(
    CENTROIDS_PATH
)

print(
    "Centroids shape:",
    centroids.shape
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = sorted([
    name
    for name in os.listdir(VALID_DIR)
    if os.path.isdir(
        os.path.join(VALID_DIR, name)
    )
])

print(
    "Number of classes:",
    len(class_names)
)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("dense").output
)

print(
    "Feature layer: dense"
)

print(
    "Feature size:",
    feature_extractor.output_shape[-1]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(image_path):

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # CNN prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    # Feature extraction
    feature = feature_extractor.predict(
        image_array,
        verbose=0
    )[0]

    # Normalize feature
    feature_norm = np.linalg.norm(
        feature
    )

    feature = feature / max(
        feature_norm,
        1e-8
    )

    # Normalize centroids
    centroid_norms = np.linalg.norm(
        centroids,
        axis=1,
        keepdims=True
    )

    normalized_centroids = (
        centroids
        /
        np.maximum(
            centroid_norms,
            1e-8
        )
    )

    # Cosine similarity
    similarities = np.dot(
        normalized_centroids,
        feature
    )

    nearest_class = int(
        np.argmax(similarities)
    )

    similarity = float(
        similarities[nearest_class]
    )

    # Top-2 CNN predictions
    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    top1 = int(
        sorted_indices[0]
    )

    top2 = int(
        sorted_indices[1]
    )

    confidence = float(
        probabilities[top1]
    )

    margin = float(
        probabilities[top1]
        -
        probabilities[top2]
    )

    # CNN / centroid agreement
    agreement = (
        top1 == nearest_class
    )

    return {
        "cnn_class": top1,
        "confidence": confidence,
        "margin": margin,
        "similarity": similarity,
        "nearest_class": nearest_class,
        "agreement": agreement
    }


# ============================================================
# COLLECT VALIDATION IMAGES
# ============================================================

def collect_validation_images():

    images = []

    for class_index, class_name in enumerate(
        class_names
    ):

        class_dir = os.path.join(
            VALID_DIR,
            class_name
        )

        for filename in sorted(
            os.listdir(class_dir)
        ):

            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            ):

                images.append(
                    (
                        os.path.join(
                            class_dir,
                            filename
                        ),
                        class_index
                    )
                )

    return images


# ============================================================
# COLLECT UNKNOWN IMAGES
# ============================================================

def collect_unknown_images():

    images = []

    if not os.path.isdir(
        UNKNOWN_DIR
    ):
        return images

    for filename in sorted(
        os.listdir(UNKNOWN_DIR)
    ):

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ):

            images.append(
                os.path.join(
                    UNKNOWN_DIR,
                    filename
                )
            )

    return images


# ============================================================
# VALIDATION DATA
# ============================================================

print("\nCollecting validation images...")

validation_images = (
    collect_validation_images()
)

print(
    "Validation samples:",
    len(validation_images)
)


# ============================================================
# UNKNOWN DATA
# ============================================================

print("\nCollecting unknown images...")

unknown_images = (
    collect_unknown_images()
)

print(
    "Unknown samples:",
    len(unknown_images)
)


# ============================================================
# PROCESS KNOWN DATA
# ============================================================

print(
    "\nProcessing validation images..."
)

known_data = []

for i, (image_path, true_label) in enumerate(
    validation_images
):

    result = process_image(
        image_path
    )

    result["true_label"] = true_label

    known_data.append(
        result
    )

    if (i + 1) % 20 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(validation_images)}"
        )


# ============================================================
# PROCESS UNKNOWN DATA
# ============================================================

print(
    "\nProcessing unknown images..."
)

unknown_data = []

for i, image_path in enumerate(
    unknown_images
):

    result = process_image(
        image_path
    )

    result["filename"] = os.path.basename(
        image_path
    )

    unknown_data.append(
        result
    )

    if (i + 1) % 5 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(unknown_images)}"
        )


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

known_conf = np.array([
    x["confidence"]
    for x in known_data
])

known_sim = np.array([
    x["similarity"]
    for x in known_data
])

known_margin = np.array([
    x["margin"]
    for x in known_data
])

unknown_conf = np.array([
    x["confidence"]
    for x in unknown_data
])

unknown_sim = np.array([
    x["similarity"]
    for x in unknown_data
])

unknown_margin = np.array([
    x["margin"]
    for x in unknown_data
])


print("\n" + "=" * 70)
print("DATA DISTRIBUTION")
print("=" * 70)

print("\nKNOWN VALIDATION")

print(
    f"Confidence: "
    f"{known_conf.min():.4f} "
    f"to "
    f"{known_conf.max():.4f}"
)

print(
    f"Similarity: "
    f"{known_sim.min():.4f} "
    f"to "
    f"{known_sim.max():.4f}"
)

print(
    f"Margin: "
    f"{known_margin.min():.4f} "
    f"to "
    f"{known_margin.max():.4f}"
)


print("\nUNKNOWN")

print(
    f"Confidence: "
    f"{unknown_conf.min():.4f} "
    f"to "
    f"{unknown_conf.max():.4f}"
)

print(
    f"Similarity: "
    f"{unknown_sim.min():.4f} "
    f"to "
    f"{unknown_sim.max():.4f}"
)

print(
    f"Margin: "
    f"{unknown_margin.min():.4f} "
    f"to "
    f"{unknown_margin.max():.4f}"
)


# ============================================================
# V2 OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR BEST V2 RULE")
print("=" * 70)


best = None


# Confidence thresholds
confidence_thresholds = np.arange(
    0.30,
    0.96,
    0.02
)

# Similarity thresholds
similarity_thresholds = np.arange(
    0.60,
    0.96,
    0.01
)

# Margin thresholds
margin_thresholds = np.arange(
    0.00,
    0.96,
    0.02
)


for confidence_threshold in (
    confidence_thresholds
):

    for similarity_threshold in (
        similarity_thresholds
    ):

        for margin_threshold in (
            margin_thresholds
        ):

            # ==================================================
            # KNOWN ACCEPTANCE
            # ==================================================

            known_accepted = 0

            for item in known_data:

                accepted = (
                    item["confidence"]
                    >= confidence_threshold
                    and
                    item["similarity"]
                    >= similarity_threshold
                    and
                    item["margin"]
                    >= margin_threshold
                    and
                    item["agreement"]
                )

                if accepted:

                    known_accepted += 1


            known_rate = (
                known_accepted
                /
                len(known_data)
            )


            # IMPORTANT:
            # V2 requires at least 90% known acceptance

            if known_rate < 0.90:

                continue


            # ==================================================
            # UNKNOWN REJECTION
            # ==================================================

            unknown_rejected = 0

            for item in unknown_data:

                accepted = (
                    item["confidence"]
                    >= confidence_threshold
                    and
                    item["similarity"]
                    >= similarity_threshold
                    and
                    item["margin"]
                    >= margin_threshold
                    and
                    item["agreement"]
                )

                if not accepted:

                    unknown_rejected += 1


            unknown_rejection_rate = (
                unknown_rejected
                /
                len(unknown_data)
            )


            # ==================================================
            # METRICS
            # ==================================================

            false_rejection_rate = (
                1.0
                -
                known_rate
            )

            false_acceptance_rate = (
                1.0
                -
                unknown_rejection_rate
            )

            balanced_accuracy = (
                known_rate
                +
                unknown_rejection_rate
            ) / 2


            # ==================================================
            # BEST RULE
            # ==================================================

            if (
                best is None
                or
                balanced_accuracy
                >
                best["balanced_accuracy"]
            ):

                best = {

                    "confidence_threshold":
                        float(
                            confidence_threshold
                        ),

                    "similarity_threshold":
                        float(
                            similarity_threshold
                        ),

                    "margin_threshold":
                        float(
                            margin_threshold
                        ),

                    "known_acceptance":
                        float(
                            known_rate
                        ),

                    "unknown_rejection":
                        float(
                            unknown_rejection_rate
                        ),

                    "false_rejection_rate":
                        float(
                            false_rejection_rate
                        ),

                    "false_acceptance_rate":
                        float(
                            false_acceptance_rate
                        ),

                    "balanced_accuracy":
                        float(
                            balanced_accuracy
                        )
                }


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST V2 RULE FOUND")
print("=" * 70)

if best is None:

    print(
        "\nNo rule achieved "
        "at least 90% known acceptance."
    )

else:

    print(
        "\nConfidence threshold :",
        best["confidence_threshold"]
    )

    print(
        "Similarity threshold :",
        best["similarity_threshold"]
    )

    print(
        "Margin threshold     :",
        best["margin_threshold"]
    )

    print(
        "\nKnown acceptance      :",
        f"{best['known_acceptance'] * 100:.2f}%"
    )

    print(
        "Unknown rejection     :",
        f"{best['unknown_rejection'] * 100:.2f}%"
    )

    print(
        "False rejection rate  :",
        f"{best['false_rejection_rate'] * 100:.2f}%"
    )

    print(
        "False acceptance rate :",
        f"{best['false_acceptance_rate'] * 100:.2f}%"
    )

    print(
        "Balanced accuracy     :",
        f"{best['balanced_accuracy'] * 100:.2f}%"
    )


# ============================================================
# SAVE RESULT
# ============================================================

output = {

    "method":
        "Confidence + Similarity + Margin + CNN/Centroid Agreement",

    "minimum_known_acceptance":
        0.90,

    "validation_images":
        len(validation_images),

    "unknown_images":
        len(unknown_images),

    "best_rule":
        best
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
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
print("V2 OPTIMIZATION COMPLETED")
print("=" * 70)
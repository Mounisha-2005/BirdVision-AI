import os
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
print("OPEN-SET DETECTOR DIAGNOSTIC")
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


# ============================================================
# NORMALIZE CENTROIDS
# ============================================================

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


# ============================================================
# PROCESS IMAGE
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

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    feature = feature_extractor.predict(
        image_array,
        verbose=0
    )[0]

    feature = feature / max(
        np.linalg.norm(feature),
        1e-8
    )

    similarities = np.dot(
        normalized_centroids,
        feature
    )

    cnn_class = int(
        np.argmax(probabilities)
    )

    centroid_class = int(
        np.argmax(similarities)
    )

    confidence = float(
        probabilities[cnn_class]
    )

    sorted_probs = np.sort(
        probabilities
    )[::-1]

    margin = float(
        sorted_probs[0]
        -
        sorted_probs[1]
    )

    similarity = float(
        similarities[centroid_class]
    )

    agreement = (
        cnn_class == centroid_class
    )

    return {
        "cnn_class": cnn_class,
        "centroid_class": centroid_class,
        "confidence": confidence,
        "margin": margin,
        "similarity": similarity,
        "agreement": agreement
    }


# ============================================================
# COLLECT VALIDATION IMAGES
# ============================================================

validation_images = []

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

            validation_images.append(
                (
                    os.path.join(
                        class_dir,
                        filename
                    ),
                    class_index
                )
            )


print(
    "\nValidation images:",
    len(validation_images)
)


# ============================================================
# DIAGNOSTIC
# ============================================================

results = []

for i, (image_path, true_class) in enumerate(
    validation_images
):

    result = process_image(
        image_path
    )

    result["true_class"] = true_class
    result["path"] = image_path

    results.append(
        result
    )

    if (i + 1) % 20 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(validation_images)}"
        )


# ============================================================
# BASIC COUNTS
# ============================================================

total = len(results)

cnn_correct = sum(
    r["cnn_class"] == r["true_class"]
    for r in results
)

centroid_correct = sum(
    r["centroid_class"] == r["true_class"]
    for r in results
)

agreement_count = sum(
    r["agreement"]
    for r in results
)

agreement_and_cnn_correct = sum(
    r["agreement"]
    and
    r["cnn_class"] == r["true_class"]
    for r in results
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC RESULTS")
print("=" * 70)

print(
    f"\nTotal validation images : {total}"
)

print(
    f"CNN correct             : "
    f"{cnn_correct}/{total} "
    f"({cnn_correct / total * 100:.2f}%)"
)

print(
    f"Centroid correct        : "
    f"{centroid_correct}/{total} "
    f"({centroid_correct / total * 100:.2f}%)"
)

print(
    f"CNN/Centroid agreement  : "
    f"{agreement_count}/{total} "
    f"({agreement_count / total * 100:.2f}%)"
)

print(
    f"Agreement + CNN correct : "
    f"{agreement_and_cnn_correct}/{total} "
    f"({agreement_and_cnn_correct / total * 100:.2f}%)"
)


# ============================================================
# CONFIDENCE / SIMILARITY / MARGIN
# ============================================================

conf = np.array([
    r["confidence"]
    for r in results
])

sim = np.array([
    r["similarity"]
    for r in results
])

margin = np.array([
    r["margin"]
    for r in results
])


print("\n" + "=" * 70)
print("KNOWN VALIDATION SCORE STATISTICS")
print("=" * 70)

print(
    f"\nConfidence mean   : {conf.mean():.4f}"
)

print(
    f"Confidence median : {np.median(conf):.4f}"
)

print(
    f"Similarity mean   : {sim.mean():.4f}"
)

print(
    f"Similarity median : {np.median(sim):.4f}"
)

print(
    f"Margin mean       : {margin.mean():.4f}"
)

print(
    f"Margin median     : {np.median(margin):.4f}"
)


# ============================================================
# DISAGREEMENT DETAILS
# ============================================================

print("\n" + "=" * 70)
print("CNN / CENTROID DISAGREEMENTS")
print("=" * 70)

disagreements = [
    r for r in results
    if not r["agreement"]
]

print(
    f"\nNumber of disagreements: "
    f"{len(disagreements)}"
)

for r in disagreements:

    print(
        "\nImage:",
        os.path.basename(r["path"])
    )

    print(
        "True class:",
        class_names[r["true_class"]]
    )

    print(
        "CNN:",
        class_names[r["cnn_class"]]
    )

    print(
        "Centroid:",
        class_names[r["centroid_class"]]
    )

    print(
        f"Confidence: {r['confidence']:.4f}"
    )

    print(
        f"Similarity: {r['similarity']:.4f}"
    )

    print(
        f"Margin: {r['margin']:.4f}"
    )


# ============================================================
# MAXIMUM POSSIBLE ACCEPTANCE
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT CONCLUSION")
print("=" * 70)

print(
    "\nWith the current CNN + centroid agreement rule:"
)

print(
    f"Maximum possible acceptance based only on "
    f"agreement = "
    f"{agreement_count / total * 100:.2f}%"
)

if agreement_count < 0.90 * total:

    print(
        "\n90% known acceptance is IMPOSSIBLE "
        "while requiring CNN/Centroid agreement."
    )

else:

    print(
        "\n90% known acceptance is possible "
        "from the agreement condition."
    )

print(
    "\nDiagnostic completed."
)

print("=" * 70)
# ============================================================
# FINAL OPEN-SET BIRD CLASSIFICATION DETECTOR - V3
# CNN + Feature Similarity + Confidence + Margin
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bird_cnn.keras"
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "train"
)

KNOWN_DIR = os.path.join(
    BASE_DIR,
    "test_images"
)

UNKNOWN_DIR = os.path.join(
    BASE_DIR,
    "test_unknown"
)

CENTROID_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_centroids.npy"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR,
    "models",
    "open_set_v3_results.json"
)

# ============================================================
# 2. PARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)

# These are the V3 thresholds obtained from the previous
# calibration experiment.
CONFIDENCE_THRESHOLD = 0.40
MARGIN_THRESHOLD = 0.00
SIMILARITY_THRESHOLD = 0.89
TOP_K_SIMILARITY_THRESHOLD = 0.87
SIMILARITY_GAP_THRESHOLD = -0.05

TOP_K = 5

# ============================================================
# 3. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "ABBOTTS BABBLER",
    "ABBOTTS BOOBY",
    "ABYSSINIAN GROUND HORNBILL",
    "AFRICAN CROWNED CRANE",
    "AFRICAN EMERALD CUCKOO",
    "AFRICAN FIREFINCH",
    "AFRICAN OYSTER CATCHER",
    "AFRICAN PIED HORNBILL",
    "AFRICAN PYGMY GOOSE",
    "ALBATROSS",
    "ALBERTS TOWHEE",
    "ALEXANDRINE PARAKEET",
    "ALPINE CHOUGH",
    "ALTAMIRA YELLOWTHROAT",
    "AMERICAN AVOCET",
    "AMERICAN BITTERN",
    "AMERICAN COOT",
    "AMERICAN FLAMINGO",
    "AMERICAN GOLDFINCH",
    "AMERICAN KESTREL"
]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# 4. FIX FOR SAVED KERAS LAMBDA FUNCTION
# ============================================================

def clip_to_unit_range(x):
    """
    Clips augmented image pixel values to [0, 1].

    This function must exist before loading the .keras model
    because the saved model contains a Lambda layer referring
    to this function.
    """
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("=" * 70)
print("OPEN-SET BIRD CLASSIFICATION DETECTOR V3")
print("=" * 70)

print("\n[1/7] Loading CNN model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
        "Please check that best_bird_cnn.keras exists."
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    compile=False
)

print("Model loaded successfully.")

# ============================================================
# 6. DISPLAY MODEL INFORMATION
# ============================================================

print("\nModel input shape:")
print(model.input_shape)

print("\nModel output shape:")
print(model.output_shape)

# ============================================================
# 7. FIND FEATURE LAYER
# ============================================================

print("\n[2/7] Searching for feature layer...")

feature_layer = None

# Your CNN architecture contains:
# GlobalAveragePooling2D -> Dense(128) -> Dropout -> Dense(20)

# Prefer the 128-unit Dense layer.
for layer in model.layers:

    if isinstance(layer, tf.keras.layers.Dense):

        if layer.units == 128:
            feature_layer = layer
            break

if feature_layer is None:
    raise RuntimeError(
        "Could not find the 128-unit Dense feature layer."
    )

print(
    "Feature layer:",
    feature_layer.name
)

print(
    "Feature size:",
    feature_layer.units
)

# ============================================================
# 8. CREATE FEATURE EXTRACTOR
# ============================================================

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)
print("Feature extractor ready.")

# ============================================================
# 9. LOAD CENTROIDS
# ============================================================

print("\n[3/7] Loading class centroids...")

if not os.path.exists(CENTROID_PATH):

    print(
        "\nWARNING: class_centroids.npy was not found."
    )

    print(
        "The script will calculate centroids from the training data."
    )

    # --------------------------------------------------------
    # Calculate centroids automatically
    # --------------------------------------------------------

    train_paths = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = os.path.join(
            TRAIN_DIR,
            class_name
        )

        if not os.path.isdir(class_dir):
            print(
                f"WARNING: Missing class directory: {class_dir}"
            )
            continue

        for filename in os.listdir(class_dir):

            filepath = os.path.join(
                class_dir,
                filename
            )

            if os.path.isfile(filepath):

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                ]:

                    train_paths.append(
                        (filepath, class_index)
                    )

    print(
        "Training images found:",
        len(train_paths)
    )

    all_features = []
    all_labels = []

    batch_images = []
    batch_labels = []

    BATCH_SIZE = 32

    for counter, (filepath, label) in enumerate(
        train_paths,
        start=1
    ):

        try:

            image = Image.open(
                filepath
            ).convert("RGB")

            image = image.resize(
                IMAGE_SIZE
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

            # IMPORTANT:
            # Model already contains Rescaling(1/255)
            # Therefore DO NOT divide by 255 here.

            batch_images.append(
                image_array
            )

            batch_labels.append(
                label
            )

            if (
                len(batch_images) == BATCH_SIZE
                or counter == len(train_paths)
            ):

                batch_array = np.asarray(
                    batch_images,
                    dtype=np.float32
                )

                features = feature_extractor.predict(
                    batch_array,
                    verbose=0
                )

                all_features.append(
                    features
                )

                all_labels.extend(
                    batch_labels
                )

                batch_images = []
                batch_labels = []

        except Exception as error:

            print(
                "Skipping:",
                filepath,
                "|",
                error
            )

    all_features = np.concatenate(
        all_features,
        axis=0
    )

    all_labels = np.asarray(
        all_labels
    )

    centroids = np.zeros(
        (
            NUM_CLASSES,
            all_features.shape[1]
        ),
        dtype=np.float32
    )

    for class_index in range(NUM_CLASSES):

        class_features = all_features[
            all_labels == class_index
        ]

        if len(class_features) == 0:

            raise RuntimeError(
                f"No training features found for class "
                f"{CLASS_NAMES[class_index]}"
            )

        centroids[class_index] = np.mean(
            class_features,
            axis=0
        )

    # Normalize centroids
    centroid_norms = np.linalg.norm(
        centroids,
        axis=1,
        keepdims=True
    )

    centroid_norms = np.maximum(
        centroid_norms,
        1e-8
    )

    centroids = centroids / centroid_norms

    np.save(
        CENTROID_PATH,
        centroids
    )

    print(
        "Centroids calculated and saved."
    )

else:

    centroids = np.load(
        CENTROID_PATH
    )

    print(
        "Centroids loaded successfully."
    )

# ============================================================
# 10. CHECK CENTROID SHAPE
# ============================================================

print(
    "Centroid shape:",
    centroids.shape
)

if centroids.shape != (
    NUM_CLASSES,
    feature_layer.units
):

    raise ValueError(
        "\nCentroid shape does not match model feature size.\n"
        f"Expected: {(NUM_CLASSES, feature_layer.units)}\n"
        f"Found: {centroids.shape}"
    )

# Normalize centroids again for safety.
centroid_norms = np.linalg.norm(
    centroids,
    axis=1,
    keepdims=True
)

centroid_norms = np.maximum(
    centroid_norms,
    1e-8
)

centroids = (
    centroids /
    centroid_norms
)

# ============================================================
# 11. IMAGE LOADING FUNCTION
# ============================================================

def load_image(filepath):

    image = Image.open(
        filepath
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # DO NOT divide by 255.
    # The CNN has Rescaling(1/255).

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# 12. COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    feature_vector,
    centroid_matrix
):

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32
    )

    feature_vector_norm = np.linalg.norm(
        feature_vector
    )

    if feature_vector_norm < 1e-8:

        return np.zeros(
            len(centroid_matrix)
        )

    feature_vector = (
        feature_vector /
        feature_vector_norm
    )

    centroid_norms = np.linalg.norm(
        centroid_matrix,
        axis=1,
        keepdims=True
    )

    centroid_norms = np.maximum(
        centroid_norms,
        1e-8
    )

    normalized_centroids = (
        centroid_matrix /
        centroid_norms
    )

    similarities = np.dot(
        normalized_centroids,
        feature_vector
    )

    return similarities


# ============================================================
# 13. PREDICTION FUNCTION
# ============================================================

def predict_image(filepath):

    image_array = load_image(
        filepath
    )

    # CNN prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = (
        CLASS_NAMES[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # Confidence margin
    # --------------------------------------------------------

    sorted_probabilities = np.sort(
        probabilities
    )[::-1]

    if len(sorted_probabilities) >= 2:

        margin = float(
            sorted_probabilities[0]
            -
            sorted_probabilities[1]
        )

    else:

        margin = confidence

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    feature = feature_extractor.predict(
        image_array,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Similarity to every class centroid
    # --------------------------------------------------------

    similarities = cosine_similarity(
        feature,
        centroids
    )

    predicted_class_similarity = float(
        similarities[predicted_index]
    )

    # --------------------------------------------------------
    # Top-K class similarity
    # --------------------------------------------------------

    top_k_count = min(
        TOP_K,
        len(similarities)
    )

    top_k_indices = np.argsort(
        similarities
    )[::-1][:top_k_count]

    top_k_similarities = similarities[
        top_k_indices
    ]

    top_k_similarity = float(
        np.mean(
            top_k_similarities
        )
    )

    # --------------------------------------------------------
    # Similarity gap
    # --------------------------------------------------------

    sorted_similarities = np.sort(
        similarities
    )[::-1]

    if len(sorted_similarities) >= 2:

        similarity_gap = float(
            sorted_similarities[0]
            -
            sorted_similarities[1]
        )

    else:

        similarity_gap = float(
            sorted_similarities[0]
        )

    # --------------------------------------------------------
    # V3 open-set decision
    # --------------------------------------------------------

    confidence_ok = (
        confidence >=
        CONFIDENCE_THRESHOLD
    )

    margin_ok = (
        margin >=
        MARGIN_THRESHOLD
    )

    similarity_ok = (
        predicted_class_similarity >=
        SIMILARITY_THRESHOLD
    )

    top_k_ok = (
        top_k_similarity >=
        TOP_K_SIMILARITY_THRESHOLD
    )

    similarity_gap_ok = (
        similarity_gap >=
        SIMILARITY_GAP_THRESHOLD
    )

    supported = (
        confidence_ok
        and
        margin_ok
        and
        similarity_ok
        and
        top_k_ok
        and
        similarity_gap_ok
    )

    if supported:

        final_label = predicted_class
        status = "SUPPORTED"

    else:

        final_label = "UNKNOWN / NOT SUPPORTED"
        status = "UNKNOWN"

    return {

        "file": os.path.basename(filepath),

        "path": filepath,

        "prediction": predicted_class,

        "final_label": final_label,

        "status": status,

        "confidence": confidence,

        "confidence_percent": (
            confidence * 100
        ),

        "margin": margin,

        "predicted_class_similarity": (
            predicted_class_similarity
        ),

        "top_k_similarity": (
            top_k_similarity
        ),

        "similarity_gap": (
            similarity_gap
        ),

        "confidence_ok": confidence_ok,

        "margin_ok": margin_ok,

        "similarity_ok": similarity_ok,

        "top_k_ok": top_k_ok,

        "similarity_gap_ok": (
            similarity_gap_ok
        )
    }


# ============================================================
# 14. FIND IMAGE FILES
# ============================================================

def get_image_files(directory):

    if not os.path.isdir(directory):

        return []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    files = []

    for filename in sorted(
        os.listdir(directory)
    ):

        filepath = os.path.join(
            directory,
            filename
        )

        if not os.path.isfile(
            filepath
        ):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in valid_extensions:

            files.append(
                filepath
            )

    return files


# ============================================================
# 15. TEST KNOWN IMAGES
# ============================================================

print("\n[4/7] Testing known images...")

known_files = get_image_files(
    KNOWN_DIR
)

if len(known_files) == 0:

    print(
        "No known test images found."
    )

else:

    print(
        f"Known images found: {len(known_files)}"
    )

known_results = []

for filepath in known_files:

    try:

        result = predict_image(
            filepath
        )

        known_results.append(
            result
        )

        print("\n" + "-" * 70)

        print(
            "Image:",
            result["file"]
        )

        print(
            "CNN prediction:",
            result["prediction"]
        )

        print(
            "Confidence:",
            f"{result['confidence_percent']:.2f}%"
        )

        print(
            "Margin:",
            f"{result['margin']:.4f}"
        )

        print(
            "Predicted-class similarity:",
            f"{result['predicted_class_similarity']:.4f}"
        )

        print(
            "Top-K similarity:",
            f"{result['top_k_similarity']:.4f}"
        )

        print(
            "Similarity gap:",
            f"{result['similarity_gap']:.4f}"
        )

        print(
            "Final decision:",
            result["final_label"]
        )

    except Exception as error:

        print(
            "\nERROR:",
            filepath
        )

        print(
            error
        )


# ============================================================
# 16. TEST UNKNOWN IMAGES
# ============================================================

print("\n[5/7] Testing unknown images...")

unknown_files = get_image_files(
    UNKNOWN_DIR
)

if len(unknown_files) == 0:

    print(
        "No unknown test images found."
    )

else:

    print(
        f"Unknown images found: "
        f"{len(unknown_files)}"
    )

unknown_results = []

for filepath in unknown_files:

    try:

        result = predict_image(
            filepath
        )

        unknown_results.append(
            result
        )

        print("\n" + "-" * 70)

        print(
            "Image:",
            result["file"]
        )

        print(
            "CNN prediction:",
            result["prediction"]
        )

        print(
            "Confidence:",
            f"{result['confidence_percent']:.2f}%"
        )

        print(
            "Margin:",
            f"{result['margin']:.4f}"
        )

        print(
            "Predicted-class similarity:",
            f"{result['predicted_class_similarity']:.4f}"
        )

        print(
            "Top-K similarity:",
            f"{result['top_k_similarity']:.4f}"
        )

        print(
            "Similarity gap:",
            f"{result['similarity_gap']:.4f}"
        )

        print(
            "Final decision:",
            result["final_label"]
        )

    except Exception as error:

        print(
            "\nERROR:",
            filepath
        )

        print(
            error
        )


# ============================================================
# 17. CALCULATE SUMMARY
# ============================================================

print("\n[6/7] Calculating summary...")

known_accepted = sum(
    1
    for result in known_results
    if result["status"] == "SUPPORTED"
)

known_rejected = (
    len(known_results)
    -
    known_accepted
)

unknown_rejected = sum(
    1
    for result in unknown_results
    if result["status"] == "UNKNOWN"
)

unknown_accepted = (
    len(unknown_results)
    -
    unknown_rejected
)

if len(known_results) > 0:

    known_acceptance_rate = (
        known_accepted /
        len(known_results)
    )

else:

    known_acceptance_rate = 0.0


if len(unknown_results) > 0:

    unknown_rejection_rate = (
        unknown_rejected /
        len(unknown_results)
    )

else:

    unknown_rejection_rate = 0.0


print("\n" + "=" * 70)
print("OPEN-SET DETECTION SUMMARY")
print("=" * 70)

print(
    "\nKnown images:"
)

print(
    "Total:",
    len(known_results)
)

print(
    "Accepted:",
    known_accepted
)

print(
    "Rejected:",
    known_rejected
)

print(
    "Known acceptance rate:",
    f"{known_acceptance_rate * 100:.2f}%"
)

print(
    "\nUnknown images:"
)

print(
    "Total:",
    len(unknown_results)
)

print(
    "Rejected:",
    unknown_rejected
)

print(
    "Accepted:",
    unknown_accepted
)

print(
    "Unknown rejection rate:",
    f"{unknown_rejection_rate * 100:.2f}%"
)

# ============================================================
# 18. SAVE RESULTS
# ============================================================

print("\n[7/7] Saving results...")

results = {

    "method":
        "CNN + predicted-class feature similarity",

    "feature_layer":
        feature_layer.name,

    "feature_size":
        int(feature_layer.units),

    "num_classes":
        NUM_CLASSES,

    "top_k":
        TOP_K,

    "thresholds": {

        "confidence":
            CONFIDENCE_THRESHOLD,

        "margin":
            MARGIN_THRESHOLD,

        "similarity":
            SIMILARITY_THRESHOLD,

        "top_k_similarity":
            TOP_K_SIMILARITY_THRESHOLD,

        "similarity_gap":
            SIMILARITY_GAP_THRESHOLD
    },

    "summary": {

        "known_total":
            len(known_results),

        "known_accepted":
            known_accepted,

        "known_rejected":
            known_rejected,

        "known_acceptance_rate":
            known_acceptance_rate,

        "unknown_total":
            len(unknown_results),

        "unknown_rejected":
            unknown_rejected,

        "unknown_accepted":
            unknown_accepted,

        "unknown_rejection_rate":
            unknown_rejection_rate
    },

    "known_results":
        known_results,

    "unknown_results":
        unknown_results
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

print(
    "\nResults saved to:"
)

print(
    OUTPUT_JSON
)

print("\n" + "=" * 70)
print("V3 EXECUTION COMPLETED")
print("=" * 70)
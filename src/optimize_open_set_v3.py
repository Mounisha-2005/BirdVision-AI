import os
import json
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity


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

VALID_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "valid"
)

UNKNOWN_DIR = os.path.join(BASE_DIR, "test_unknown")

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_bird_cnn.keras"
)

FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "train_features.npy"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "models", "train_labels.npy"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "models", "optimized_open_set_v3.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

# We will compare the uploaded image against
# the closest training examples of its predicted class.
TOP_K = 5


# ============================================================
# CUSTOM FUNCTION USED BY MODEL
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("OPEN-SET DETECTOR V3 OPTIMIZATION")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

print("Model loaded successfully.")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = sorted(
    [
        name
        for name in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, name))
    ]
)

NUM_CLASSES = len(class_names)

print("Number of classes:", NUM_CLASSES)

for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("dense").output
)

print("Feature layer: dense")
print("Feature size:", feature_extractor.output_shape[-1])


# ============================================================
# LOAD TRAINING FEATURE DATABASE
# ============================================================

train_features = np.load(FEATURES_PATH)
train_labels = np.load(LABELS_PATH)

print()
print("Training feature database:")
print("Features shape:", train_features.shape)
print("Labels shape :", train_labels.shape)


# ============================================================
# GROUP TRAINING FEATURES BY CLASS
# ============================================================

class_features = {}

for class_id in range(NUM_CLASSES):
    class_features[class_id] = train_features[
        train_labels == class_id
    ]

print()
print("Training feature groups created.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image = image.convert("RGB")

    image_array = img_to_array(image).astype(
        np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_feature(feature):

    norm = np.linalg.norm(feature)

    if norm == 0:
        return feature

    return feature / norm


# ============================================================
# GET IMAGE SCORES
# ============================================================

def analyze_image(image_path):

    image = preprocess_image(image_path)

    # --------------------------------------------------------
    # CNN prediction
    # --------------------------------------------------------

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    sorted_indices = np.argsort(
        prediction
    )[::-1]

    top1 = int(sorted_indices[0])
    top2 = int(sorted_indices[1])

    confidence = float(
        prediction[top1]
    )

    second_confidence = float(
        prediction[top2]
    )

    margin = (
        confidence -
        second_confidence
    )

    # --------------------------------------------------------
    # Extract feature
    # --------------------------------------------------------

    feature = feature_extractor.predict(
        image,
        verbose=0
    )[0]

    feature = normalize_feature(
        feature
    )

    # --------------------------------------------------------
    # Compare ONLY against predicted class
    # --------------------------------------------------------

    predicted_class_features = (
        class_features[top1]
    )

    similarities = cosine_similarity(
        feature.reshape(1, -1),
        predicted_class_features
    )[0]

    similarities = np.sort(
        similarities
    )[::-1]

    max_similarity = float(
        similarities[0]
    )

    k = min(
        TOP_K,
        len(similarities)
    )

    top_k_similarity = float(
        np.mean(
            similarities[:k]
        )
    )

    # --------------------------------------------------------
    # Compare against every OTHER class
    # --------------------------------------------------------

    other_class_scores = []

    for class_id in range(NUM_CLASSES):

        if class_id == top1:
            continue

        features = class_features[
            class_id
        ]

        sims = cosine_similarity(
            feature.reshape(1, -1),
            features
        )[0]

        other_class_scores.append(
            float(np.max(sims))
        )

    best_other_similarity = max(
        other_class_scores
    )

    similarity_gap = (
        max_similarity -
        best_other_similarity
    )

    return {
        "predicted_class": top1,
        "confidence": confidence,
        "margin": margin,
        "max_similarity": max_similarity,
        "top_k_similarity": top_k_similarity,
        "best_other_similarity": best_other_similarity,
        "similarity_gap": similarity_gap
    }


# ============================================================
# COLLECT VALIDATION DATA
# ============================================================

print()
print("=" * 70)
print("ANALYZING VALIDATION DATA")
print("=" * 70)

validation_results = []

for class_id, class_name in enumerate(class_names):

    class_dir = os.path.join(
        VALID_DIR,
        class_name
    )

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    for filename in files:

        image_path = os.path.join(
            class_dir,
            filename
        )

        result = analyze_image(
            image_path
        )

        result["true_class"] = class_id
        result["filename"] = filename
        result["type"] = "known"

        validation_results.append(
            result
        )

print(
    "Validation samples:",
    len(validation_results)
)


# ============================================================
# COLLECT UNKNOWN DATA
# ============================================================

print()
print("=" * 70)
print("ANALYZING UNKNOWN DATA")
print("=" * 70)

unknown_results = []

unknown_files = [
    f for f in os.listdir(UNKNOWN_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]

for filename in unknown_files:

    image_path = os.path.join(
        UNKNOWN_DIR,
        filename
    )

    result = analyze_image(
        image_path
    )

    result["true_class"] = -1
    result["filename"] = filename
    result["type"] = "unknown"

    unknown_results.append(
        result
    )

print(
    "Unknown samples:",
    len(unknown_results)
)


# ============================================================
# DISPLAY SCORE RANGES
# ============================================================

def show_range(results, name):

    confidence = [
        x["confidence"]
        for x in results
    ]

    margin = [
        x["margin"]
        for x in results
    ]

    similarity = [
        x["max_similarity"]
        for x in results
    ]

    topk = [
        x["top_k_similarity"]
        for x in results
    ]

    gap = [
        x["similarity_gap"]
        for x in results
    ]

    print()
    print(name)

    print(
        "Confidence:",
        f"{min(confidence):.4f}",
        "to",
        f"{max(confidence):.4f}"
    )

    print(
        "Margin:",
        f"{min(margin):.4f}",
        "to",
        f"{max(margin):.4f}"
    )

    print(
        "Max similarity:",
        f"{min(similarity):.4f}",
        "to",
        f"{max(similarity):.4f}"
    )

    print(
        "Top-K similarity:",
        f"{min(topk):.4f}",
        "to",
        f"{max(topk):.4f}"
    )

    print(
        "Similarity gap:",
        f"{min(gap):.4f}",
        "to",
        f"{max(gap):.4f}"
    )


show_range(
    validation_results,
    "KNOWN VALIDATION"
)

show_range(
    unknown_results,
    "UNKNOWN"
)


# ============================================================
# FAST V3 OPTIMIZATION
# ============================================================

print()
print("=" * 70)
print("FAST V3 OPTIMIZATION")
print("=" * 70)

# ------------------------------------------------------------
# Convert results to NumPy arrays
# ------------------------------------------------------------

known_conf = np.array(
    [x["confidence"] for x in validation_results],
    dtype=np.float32
)

known_margin = np.array(
    [x["margin"] for x in validation_results],
    dtype=np.float32
)

known_sim = np.array(
    [x["max_similarity"] for x in validation_results],
    dtype=np.float32
)

known_topk = np.array(
    [x["top_k_similarity"] for x in validation_results],
    dtype=np.float32
)

known_gap = np.array(
    [x["similarity_gap"] for x in validation_results],
    dtype=np.float32
)

unknown_conf = np.array(
    [x["confidence"] for x in unknown_results],
    dtype=np.float32
)

unknown_margin = np.array(
    [x["margin"] for x in unknown_results],
    dtype=np.float32
)

unknown_sim = np.array(
    [x["max_similarity"] for x in unknown_results],
    dtype=np.float32
)

unknown_topk = np.array(
    [x["top_k_similarity"] for x in unknown_results],
    dtype=np.float32
)

unknown_gap = np.array(
    [x["similarity_gap"] for x in unknown_results],
    dtype=np.float32
)

# ------------------------------------------------------------
# Threshold candidates
#
# Much smaller than the previous 10+ million combinations.
# ------------------------------------------------------------

confidence_thresholds = np.arange(
    0.40, 0.96, 0.02
)

margin_thresholds = np.arange(
    0.00, 0.91, 0.04
)

similarity_thresholds = np.arange(
    0.65, 0.96, 0.02
)

topk_thresholds = np.arange(
    0.65, 0.96, 0.02
)

gap_thresholds = np.arange(
    -0.05, 0.31, 0.04
)

print("Candidate thresholds:")
print("Confidence :", len(confidence_thresholds))
print("Margin     :", len(margin_thresholds))
print("Similarity :", len(similarity_thresholds))
print("Top-K      :", len(topk_thresholds))
print("Gap        :", len(gap_thresholds))

total_combinations = (
    len(confidence_thresholds)
    * len(margin_thresholds)
    * len(similarity_thresholds)
    * len(topk_thresholds)
    * len(gap_thresholds)
)

print()
print("Total combinations:", total_combinations)
print("This is a FAST vectorized search.")
print()

# ------------------------------------------------------------
# FAST SEARCH
# ------------------------------------------------------------

best_result = None

for conf_t in confidence_thresholds:

    known_conf_ok = known_conf >= conf_t
    unknown_conf_ok = unknown_conf >= conf_t

    for margin_t in margin_thresholds:

        known_margin_ok = known_margin >= margin_t
        unknown_margin_ok = unknown_margin >= margin_t

        for sim_t in similarity_thresholds:

            known_sim_ok = known_sim >= sim_t
            unknown_sim_ok = unknown_sim >= sim_t

            for topk_t in topk_thresholds:

                known_topk_ok = known_topk >= topk_t
                unknown_topk_ok = unknown_topk >= topk_t

                for gap_t in gap_thresholds:

                    # ----------------------------------------
                    # Vectorized acceptance
                    # ----------------------------------------

                    known_accept_mask = (
                        known_conf_ok
                        & known_margin_ok
                        & known_sim_ok
                        & known_topk_ok
                        & (known_gap >= gap_t)
                    )

                    unknown_accept_mask = (
                        unknown_conf_ok
                        & unknown_margin_ok
                        & unknown_sim_ok
                        & unknown_topk_ok
                        & (unknown_gap >= gap_t)
                    )

                    known_rate = (
                        np.mean(known_accept_mask)
                    )

                    unknown_rejection_rate = (
                        1.0
                        - np.mean(unknown_accept_mask)
                    )

                    # ----------------------------------------
                    # We need reasonable known acceptance.
                    #
                    # CNN validation accuracy is 82%.
                    # Therefore 90% known acceptance is not
                    # realistic if we want reliable predictions.
                    # ----------------------------------------

                    if known_rate < 0.70:
                        continue

                    balanced_accuracy = (
                        known_rate
                        + unknown_rejection_rate
                    ) / 2.0

                    # ----------------------------------------
                    # Save best rule
                    # ----------------------------------------

                    if (
                        best_result is None
                        or balanced_accuracy
                        > best_result["balanced_accuracy"]
                    ):

                        best_result = {

                            "confidence_threshold":
                                float(conf_t),

                            "margin_threshold":
                                float(margin_t),

                            "similarity_threshold":
                                float(sim_t),

                            "top_k_similarity_threshold":
                                float(topk_t),

                            "similarity_gap_threshold":
                                float(gap_t),

                            "known_acceptance":
                                float(known_rate),

                            "unknown_rejection":
                                float(
                                    unknown_rejection_rate
                                ),

                            "false_rejection_rate":
                                float(
                                    1.0 - known_rate
                                ),

                            "false_acceptance_rate":
                                float(
                                    1.0
                                    - unknown_rejection_rate
                                ),

                            "balanced_accuracy":
                                float(
                                    balanced_accuracy
                                )
                        }

# ============================================================
# PRINT BEST RESULT
# ============================================================

print()
print("=" * 70)
print("BEST V3 RULE FOUND")
print("=" * 70)

if best_result is None:

    print("No suitable V3 rule found.")

else:

    print(
        f"Confidence threshold       : "
        f"{best_result['confidence_threshold']:.4f}"
    )

    print(
        f"Margin threshold            : "
        f"{best_result['margin_threshold']:.4f}"
    )

    print(
        f"Similarity threshold       : "
        f"{best_result['similarity_threshold']:.4f}"
    )

    print(
        f"Top-K similarity threshold : "
        f"{best_result['top_k_similarity_threshold']:.4f}"
    )

    print(
        f"Similarity gap threshold   : "
        f"{best_result['similarity_gap_threshold']:.4f}"
    )

    print()

    print(
        f"Known acceptance            : "
        f"{best_result['known_acceptance'] * 100:.2f}%"
    )

    print(
        f"Unknown rejection           : "
        f"{best_result['unknown_rejection'] * 100:.2f}%"
    )

    print(
        f"False rejection rate        : "
        f"{best_result['false_rejection_rate'] * 100:.2f}%"
    )

    print(
        f"False acceptance rate       : "
        f"{best_result['false_acceptance_rate'] * 100:.2f}%"
    )

    print(
        f"Balanced accuracy           : "
        f"{best_result['balanced_accuracy'] * 100:.2f}%"
    )

# ============================================================
# SAVE CONFIGURATION
# ============================================================

if best_result is not None:

    configuration = {

        "method":
            "CNN + predicted-class training-feature similarity",

        "feature_layer":
            "dense",

        "feature_size":
            128,

        "top_k":
            TOP_K,

        "validation_samples":
            len(validation_results),

        "unknown_samples":
            len(unknown_results),

        "best_rule":
            best_result
    }

    with open(
        OUTPUT_PATH,
        "w"
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH
    )

print()
print("=" * 70)
print("V3 OPTIMIZATION COMPLETE")
print("=" * 70)
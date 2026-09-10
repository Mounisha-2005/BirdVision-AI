# ============================================================
# FINAL OPEN-SET DETECTOR V2
# Score-based open-set detection
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps, ImageEnhance


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_bird_cnn.keras"
)

TRAIN_FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "train_features.npy"
)

TRAIN_LABELS_PATH = os.path.join(
    BASE_DIR, "models", "train_labels.npy"
)

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

KNOWN_DIR = os.path.join(
    BASE_DIR, "test_images"
)

UNKNOWN_DIR = os.path.join(
    BASE_DIR, "test_unknown"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_open_set_v2_config.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "open_set_v2"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CUSTOM MODEL FUNCTION
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(
        x,
        0.0,
        1.0
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading CNN model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "clip_to_unit_range":
                clip_to_unit_range
        },
        safe_mode=False
    )

    print("CNN loaded successfully.")

    return model


# ============================================================
# CLASS NAMES
# ============================================================

def get_class_names():

    classes = sorted([
        folder
        for folder in os.listdir(TRAIN_DIR)
        if os.path.isdir(
            os.path.join(
                TRAIN_DIR,
                folder
            )
        )
    ])

    return classes


# ============================================================
# LOAD FEATURE DATABASE
# ============================================================

def load_features():

    print("\nLoading feature database...")

    features = np.load(
        TRAIN_FEATURES_PATH
    ).astype(np.float32)

    labels = np.load(
        TRAIN_LABELS_PATH
    ).astype(np.int32)

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-8
    )

    features = features / norms

    print(
        "Feature database:",
        features.shape
    )

    return features, labels


# ============================================================
# FEATURE MODEL
# ============================================================

def create_feature_model(model):

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(
            "dense"
        ).output
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = Image.open(
        path
    ).convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    return image


# ============================================================
# TEST-TIME AUGMENTATION
# ============================================================

def make_tta(image):

    images = []

    # Original
    images.append(
        image.copy()
    )

    # Horizontal flip
    images.append(
        ImageOps.mirror(
            image
        )
    )

    # Slightly darker
    images.append(
        ImageEnhance.Brightness(
            image
        ).enhance(0.90)
    )

    # Slightly brighter
    images.append(
        ImageEnhance.Brightness(
            image
        ).enhance(1.10)
    )

    # Center crop
    w, h = image.size

    crop_w = int(
        w * 0.90
    )

    crop_h = int(
        h * 0.90
    )

    left = (
        w - crop_w
    ) // 2

    top = (
        h - crop_h
    ) // 2

    cropped = image.crop(
        (
            left,
            top,
            left + crop_w,
            top + crop_h
        )
    )

    cropped = cropped.resize(
        IMG_SIZE
    )

    images.append(
        cropped
    )

    arrays = []

    for img in images:

        arrays.append(
            np.array(
                img,
                dtype=np.float32
            )
        )

    return np.stack(
        arrays
    )


# ============================================================
# FEATURE SIMILARITY
# ============================================================

def calculate_similarity(
    feature,
    predicted_class,
    train_features,
    train_labels
):

    feature = feature.astype(
        np.float32
    )

    norm = np.linalg.norm(
        feature
    )

    if norm < 1e-8:

        return 0.0

    feature = feature / norm

    similarities = np.dot(
        train_features,
        feature
    )

    mask = (
        train_labels
        ==
        predicted_class
    )

    class_similarities = (
        similarities[mask]
    )

    if len(
        class_similarities
    ) == 0:

        return 0.0

    return float(
        np.max(
            class_similarities
        )
    )


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(
    path,
    model,
    feature_model,
    train_features,
    train_labels,
    class_names
):

    image = load_image(
        path
    )

    tta = make_tta(
        image
    )

    predictions = model.predict(
        tta,
        verbose=0
    )

    # --------------------------------------------------------
    # Original prediction
    # --------------------------------------------------------

    original_probs = predictions[0]

    predicted_class = int(
        np.argmax(
            original_probs
        )
    )

    confidence = float(
        original_probs[
            predicted_class
        ]
    )

    # --------------------------------------------------------
    # Mean confidence
    # --------------------------------------------------------

    mean_confidence = float(
        np.mean(
            predictions[
                :,
                predicted_class
            ]
        )
    )

    # --------------------------------------------------------
    # TTA agreement
    # --------------------------------------------------------

    tta_predictions = np.argmax(
        predictions,
        axis=1
    )

    agreement = float(
        np.mean(
            tta_predictions
            ==
            predicted_class
        )
    )

    # --------------------------------------------------------
    # Prediction margin
    # --------------------------------------------------------

    sorted_probs = np.sort(
        original_probs
    )[::-1]

    margin = float(
        sorted_probs[0]
        -
        sorted_probs[1]
    )

    # --------------------------------------------------------
    # Feature
    # --------------------------------------------------------

    original_array = np.expand_dims(
        np.array(
            image,
            dtype=np.float32
        ),
        axis=0
    )

    feature = feature_model.predict(
        original_array,
        verbose=0
    )[0]

    similarity = calculate_similarity(
        feature,
        predicted_class,
        train_features,
        train_labels
    )

    # --------------------------------------------------------
    # COMBINED SUPPORT SCORE
    # --------------------------------------------------------
    #
    # We deliberately don't require every metric to pass.
    #
    # Confidence  = 40%
    # TTA         = 25%
    # Similarity  = 25%
    # Margin      = 10%
    #
    # --------------------------------------------------------

    support_score = (
        0.40 * mean_confidence
        +
        0.25 * agreement
        +
        0.25 * similarity
        +
        0.10 * margin
    )

    return {

        "image":
            os.path.basename(path),

        "predicted_index":
            predicted_class,

        "predicted_class":
            class_names[
                predicted_class
            ],

        "confidence":
            confidence,

        "mean_confidence":
            mean_confidence,

        "tta_agreement":
            agreement,

        "similarity":
            similarity,

        "margin":
            margin,

        "support_score":
            float(
                support_score
            )
    }


# ============================================================
# GET IMAGES
# ============================================================

def get_images(directory):

    images = []

    for root, dirs, files in os.walk(
        directory
    ):

        for file in files:

            if file.lower().endswith(
                VALID_EXTENSIONS
            ):

                images.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    images.sort()

    return images


# ============================================================
# CALIBRATION DATA
# ============================================================

def analyze_directory(
    directory,
    model,
    feature_model,
    train_features,
    train_labels,
    class_names,
    title
):

    files = get_images(
        directory
    )

    print(
        f"\n{title}: {len(files)} images"
    )

    results = []

    for i, path in enumerate(
        files,
        1
    ):

        print(
            f"\rProcessing "
            f"{i}/{len(files)}",
            end=""
        )

        result = analyze_image(
            path,
            model,
            feature_model,
            train_features,
            train_labels,
            class_names
        )

        results.append(
            result
        )

    print()

    return results


# ============================================================
# OPTIMIZE SCORE THRESHOLD
# ============================================================

def optimize_threshold(
    known_results,
    unknown_results
):

    print("\n")
    print("=" * 65)
    print("OPTIMIZING SUPPORT SCORE THRESHOLD")
    print("=" * 65)

    best = None

    thresholds = np.arange(
        0.50,
        0.951,
        0.005
    )

    for threshold in thresholds:

        known_accept = sum(
            r["support_score"]
            >=
            threshold
            for r in known_results
        )

        unknown_reject = sum(
            r["support_score"]
            <
            threshold
            for r in unknown_results
        )

        known_rate = (
            known_accept
            /
            len(known_results)
        )

        unknown_rate = (
            unknown_reject
            /
            len(unknown_results)
        )

        balanced_accuracy = (
            known_rate
            +
            unknown_rate
        ) / 2

        candidate = {

            "threshold":
                float(threshold),

            "known_acceptance":
                float(known_rate),

            "unknown_rejection":
                float(unknown_rate),

            "balanced_accuracy":
                float(
                    balanced_accuracy
                )
        }

        if best is None:

            best = candidate

        elif (
            candidate[
                "balanced_accuracy"
            ]
            >
            best[
                "balanced_accuracy"
            ]
        ):

            best = candidate

        elif (
            candidate[
                "balanced_accuracy"
            ]
            ==
            best[
                "balanced_accuracy"
            ]
            and
            candidate[
                "known_acceptance"
            ]
            >
            best[
                "known_acceptance"
            ]
        ):

            best = candidate

    print(
        "\nBest threshold:",
        best["threshold"]
    )

    print(
        "Known acceptance:",
        f"{best['known_acceptance'] * 100:.2f}%"
    )

    print(
        "Unknown rejection:",
        f"{best['unknown_rejection'] * 100:.2f}%"
    )

    print(
        "Balanced accuracy:",
        f"{best['balanced_accuracy'] * 100:.2f}%"
    )

    return best


# ============================================================
# APPLY FINAL RULE
# ============================================================

def apply_rule(
    result,
    threshold
):

    if (
        result["support_score"]
        >=
        threshold
    ):

        result["status"] = (
            "SUPPORTED"
        )

    else:

        result["status"] = (
            "NOT SUPPORTED"
        )

    return result


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    title,
    results,
    threshold
):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for result in results:

        result = apply_rule(
            result,
            threshold
        )

        print(
            f"\nImage: {result['image']}"
        )

        print(
            f"Prediction: "
            f"{result['predicted_class']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            f"Mean TTA confidence: "
            f"{result['mean_confidence'] * 100:.2f}%"
        )

        print(
            f"TTA agreement: "
            f"{result['tta_agreement'] * 100:.0f}%"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Margin: "
            f"{result['margin']:.4f}"
        )

        print(
            f"SUPPORT SCORE: "
            f"{result['support_score']:.4f}"
        )

        print(
            f"STATUS: "
            f"{result['status']}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_json(
    path,
    data
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    print(
        "\nSaved:",
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(" FINAL OPEN-SET BIRD DETECTOR V2")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = load_model()

    class_names = get_class_names()

    train_features, train_labels = (
        load_features()
    )

    feature_model = (
        create_feature_model(
            model
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    known_calibration = (
        analyze_directory(
            VALID_DIR,
            model,
            feature_model,
            train_features,
            train_labels,
            class_names,
            "Validation set"
        )
    )

    # --------------------------------------------------------
    # Unknown calibration
    # --------------------------------------------------------

    unknown_calibration = (
        analyze_directory(
            UNKNOWN_DIR,
            model,
            feature_model,
            train_features,
            train_labels,
            class_names,
            "Unknown set"
        )
    )

    # --------------------------------------------------------
    # Find threshold
    # --------------------------------------------------------

    optimization = optimize_threshold(
        known_calibration,
        unknown_calibration
    )

    threshold = (
        optimization[
            "threshold"
        ]
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {

        "method":
            "CNN + TTA + feature similarity "
            "combined support score",

        "weights": {

            "mean_confidence":
                0.40,

            "tta_agreement":
                0.25,

            "similarity":
                0.25,

            "margin":
                0.10
        },

        "threshold":
            threshold,

        "calibration": optimization
    }

    save_json(
        CONFIG_PATH,
        config
    )

    # ========================================================
    # TEST KNOWN IMAGES
    # ========================================================

    known_results = (
        analyze_directory(
            KNOWN_DIR,
            model,
            feature_model,
            train_features,
            train_labels,
            class_names,
            "Known test images"
        )
    )

    print_results(
        "KNOWN TEST RESULTS",
        known_results,
        threshold
    )

    save_json(
        os.path.join(
            RESULTS_DIR,
            "known_results.json"
        ),
        known_results
    )

    # ========================================================
    # TEST UNKNOWN
    # ========================================================

    unknown_results = (
        analyze_directory(
            UNKNOWN_DIR,
            model,
            feature_model,
            train_features,
            train_labels,
            class_names,
            "Unknown test images"
        )
    )

    print_results(
        "UNKNOWN TEST RESULTS",
        unknown_results,
        threshold
    )

    save_json(
        os.path.join(
            RESULTS_DIR,
            "unknown_results.json"
        ),
        unknown_results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    known_supported = sum(
        r["support_score"]
        >=
        threshold
        for r in known_results
    )

    unknown_rejected = sum(
        r["support_score"]
        <
        threshold
        for r in unknown_results
    )

    print("\n")
    print("=" * 70)
    print("FINAL V2 SUMMARY")
    print("=" * 70)

    print(
        "\nKnown images:",
        len(known_results)
    )

    print(
        "Known accepted:",
        known_supported
    )

    print(
        "Known acceptance rate:",
        f"{known_supported / len(known_results) * 100:.2f}%"
    )

    print(
        "\nUnknown images:",
        len(unknown_results)
    )

    print(
        "Unknown rejected:",
        unknown_rejected
    )

    print(
        "Unknown rejection rate:",
        f"{unknown_rejected / len(unknown_results) * 100:.2f}%"
    )

    print("\n")
    print("=" * 70)
    print("V2 DETECTOR COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
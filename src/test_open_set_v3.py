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

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_bird_cnn.keras"
)

FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "train_features.npy"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "models", "train_labels.npy"
)

CONFIG_PATH = os.path.join(
    BASE_DIR, "models", "optimized_open_set_v3.json"
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "train"
)

KNOWN_DIR = os.path.join(BASE_DIR, "test_images")
UNKNOWN_DIR = os.path.join(BASE_DIR, "test_unknown")

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

class_names = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

print("=" * 70)
print("V3 OPEN-SET FINAL TEST")
print("=" * 70)

print("Model loaded.")
print("Classes:", len(class_names))


# ============================================================
# LOAD FEATURE DATABASE
# ============================================================

train_features = np.load(FEATURES_PATH)
train_labels = np.load(LABELS_PATH)

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("dense").output
)

print("Feature database:", train_features.shape)


# ============================================================
# GROUP TRAINING FEATURES BY CLASS
# ============================================================

class_features = {}

for class_id in range(len(class_names)):
    class_features[class_id] = train_features[
        train_labels == class_id
    ]


# ============================================================
# LOAD V3 RULE
# ============================================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

rule = config["best_rule"]

CONFIDENCE_THRESHOLD = float(rule["confidence_threshold"])
MARGIN_THRESHOLD = float(rule["margin_threshold"])
SIMILARITY_THRESHOLD = float(rule["similarity_threshold"])
TOPK_THRESHOLD = float(rule["top_k_similarity_threshold"])
GAP_THRESHOLD = float(rule["similarity_gap_threshold"])

print("\nV3 thresholds:")
print("Confidence :", CONFIDENCE_THRESHOLD)
print("Margin     :", MARGIN_THRESHOLD)
print("Similarity :", SIMILARITY_THRESHOLD)
print("Top-K      :", TOPK_THRESHOLD)
print("Gap        :", GAP_THRESHOLD)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image_path):

    image = load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    image_array = img_to_array(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # CNN prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(probabilities)[::-1]

    predicted_class = int(top_indices[0])
    second_class = int(top_indices[1])

    confidence = float(probabilities[predicted_class])

    margin = float(
        confidence -
        probabilities[second_class]
    )

    # Feature extraction
    feature = feature_extractor.predict(
        image_array,
        verbose=0
    )[0]

    feature_norm = np.linalg.norm(feature)

    if feature_norm > 0:
        feature = feature / feature_norm

    # Similarities to predicted class
    similarities = cosine_similarity(
        feature.reshape(1, -1),
        class_features[predicted_class]
    )[0]

    similarities = np.sort(similarities)[::-1]

    max_similarity = float(similarities[0])

    k = min(5, len(similarities))

    topk_similarity = float(
        np.mean(similarities[:k])
    )

    # Best competing class
    other_class_similarities = []

    for class_id in range(len(class_names)):

        if class_id == predicted_class:
            continue

        sims = cosine_similarity(
            feature.reshape(1, -1),
            class_features[class_id]
        )[0]

        other_class_similarities.append(
            float(np.max(sims))
        )

    best_other_similarity = max(
        other_class_similarities
    )

    similarity_gap = (
        max_similarity -
        best_other_similarity
    )

    accepted = (
    confidence >= CONFIDENCE_THRESHOLD
    and margin >= MARGIN_THRESHOLD
    and max_similarity >= SIMILARITY_THRESHOLD
    and topk_similarity >= TOPK_THRESHOLD
    and similarity_gap >= GAP_THRESHOLD
    )
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "margin": margin,
        "max_similarity": max_similarity,
        "topk_similarity": topk_similarity,
        "similarity_gap": similarity_gap,
        "accepted": accepted
    }


# ============================================================
# TEST KNOWN IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("KNOWN TEST IMAGES")
print("=" * 70)

known_files = sorted([
    f for f in os.listdir(KNOWN_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
])

known_correct = 0

for filename in known_files:

    path = os.path.join(KNOWN_DIR, filename)

    result = analyze_image(path)

    predicted = class_names[
        result["predicted_class"]
    ]

    print("\nFile:", filename)
    print("Prediction:", predicted)
    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )
    print(
        "Similarity:",
        f"{result['max_similarity']:.4f}"
    )
    print(
        "Top-K similarity:",
        f"{result['topk_similarity']:.4f}"
    )
    print(
        "Similarity gap:",
        f"{result['similarity_gap']:.4f}"
    )

    if result["accepted"]:

        print("STATUS: SUPPORTED")

        # Expected known classes
        expected = {
            "1.jpg": "ABYSSINIAN GROUND HORNBILL",
            "2.jpg": "AMERICAN KESTREL",
            "4.jpg": "ALEXANDRINE PARAKEET"
        }

        if filename in expected:
            if predicted == expected[filename]:
                known_correct += 1
                print("CLASSIFICATION: CORRECT")
            else:
                print(
                    "CLASSIFICATION: WRONG "
                    "(accepted but incorrect)"
                )

    else:
        print("STATUS: UNKNOWN / NOT SUPPORTED")


# ============================================================
# TEST UNKNOWN IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("UNKNOWN TEST IMAGES")
print("=" * 70)

unknown_files = sorted([
    f for f in os.listdir(UNKNOWN_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
])

unknown_rejected = 0

for filename in unknown_files:

    path = os.path.join(
        UNKNOWN_DIR,
        filename
    )

    result = analyze_image(path)

    predicted = class_names[
        result["predicted_class"]
    ]

    print("\nFile:", filename)
    print("CNN would predict:", predicted)
    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )
    print(
        "Similarity:",
        f"{result['max_similarity']:.4f}"
    )
    print(
        "Top-K similarity:",
        f"{result['topk_similarity']:.4f}"
    )
    print(
        "Similarity gap:",
        f"{result['similarity_gap']:.4f}"
    )

    if result["accepted"]:
        print("STATUS: FALSE ACCEPTANCE")
    else:
        print("STATUS: UNKNOWN / CORRECTLY REJECTED")
        unknown_rejected += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL V3 TEST SUMMARY")
print("=" * 70)

print(
    f"Known images correctly accepted/classified : "
    f"{known_correct}/{len(known_files)}"
)

print(
    f"Unknown images rejected : "
    f"{unknown_rejected}/{len(unknown_files)}"
)

if len(unknown_files) > 0:
    print(
        "Unknown rejection rate : "
        f"{unknown_rejected / len(unknown_files) * 100:.2f}%"
    )

print("=" * 70)
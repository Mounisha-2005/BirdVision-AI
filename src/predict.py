import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bird_cnn.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

TEST_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "test_images"
)

IMG_SIZE = (224, 224)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BIRD CLASSIFICATION CNN - NEW IMAGE PREDICTION")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found:\n{CLASS_NAMES_PATH}"
    )

if not os.path.exists(TEST_IMAGE_DIR):
    raise FileNotFoundError(
        f"Test image folder not found:\n{TEST_IMAGE_DIR}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nNumber of classes:", len(class_names))

for i, name in enumerate(class_names):
    print(f"{i:2d} : {name}")


# ============================================================
# CUSTOM FUNCTION USED BY MODEL
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST TRAINED MODEL")
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
# REPAIR SAVED LAMBDA FUNCTION
# ============================================================

try:

    augmentation_model = model.get_layer(
        "data_augmentation"
    )

    clip_layer = augmentation_model.layers[-1]

    if isinstance(
        clip_layer,
        tf.keras.layers.Lambda
    ):
        clip_layer.function = clip_to_unit_range

    print("Saved Lambda layer repaired successfully.")

except Exception as e:

    print("Lambda repair skipped:", e)


# ============================================================
# FIND IMAGE FILES
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_files = []

for filename in os.listdir(TEST_IMAGE_DIR):

    if filename.lower().endswith(valid_extensions):

        image_files.append(
            os.path.join(
                TEST_IMAGE_DIR,
                filename
            )
        )


# ============================================================
# CHECK IMAGES
# ============================================================

if len(image_files) == 0:

    print("\nNo images found in:")
    print(TEST_IMAGE_DIR)

    print("\nPut one or more bird images inside:")
    print(TEST_IMAGE_DIR)

    print("\nSupported formats:")
    print("JPG, JPEG, PNG, BMP, WEBP")

    exit()


print("\n" + "=" * 70)
print("IMAGES FOUND")
print("=" * 70)

print("Number of images:", len(image_files))

for image_path in image_files:

    print(
        " -",
        os.path.basename(image_path)
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize(IMG_SIZE)

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.array(image)

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    image_array = image_array.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = predictions[0]

    # --------------------------------------------------------
    # Best class
    # --------------------------------------------------------

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ] * 100

    # --------------------------------------------------------
    # Top 3 predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    return (
        predicted_class,
        confidence,
        probabilities,
        top_indices
    )


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)


for image_path in image_files:

    print("\n" + "-" * 70)

    print(
        "Image:",
        os.path.basename(image_path)
    )

    try:

        (
            predicted_class,
            confidence,
            probabilities,
            top_indices
        ) = predict_image(
            image_path
        )

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        print(
            "\nPredicted Bird :",
            predicted_class
        )

        print(
            "Confidence     :",
            f"{confidence:.2f}%"
        )

        # ----------------------------------------------------
        # Top 3
        # ----------------------------------------------------

        print("\nTop 3 Predictions:")

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            print(
                f"{rank}. "
                f"{class_names[index]} "
                f"({probabilities[index] * 100:.2f}%)"
            )

    except Exception as e:

        print(
            "\nPrediction failed:"
        )

        print(e)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)
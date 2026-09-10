import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Dell\Desktop\Bird_Classification_CNN"

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Desktop",
    "Bird_Classification_CNN",
    "dataset",
    "test"
)

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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

EVALUATION_DIR = os.path.join(
    RESULTS_DIR,
    "evaluation"
)

os.makedirs(EVALUATION_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BIRD CLASSIFICATION CNN - FINAL TEST EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found:\n{CLASS_NAMES_PATH}"
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
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST TRAINED MODEL")
print("=" * 70)


# ---------------------------------------------------------
# Custom function used by the Lambda layer in the saved model
# ---------------------------------------------------------
def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ---------------------------------------------------------
# Load the saved best model
# ---------------------------------------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "clip_to_unit_range": clip_to_unit_range
    },
    safe_mode=False
)

print("Best trained model loaded successfully.")

# Repair the Lambda layer from the saved model


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

        print(
            "Saved Lambda layer repaired successfully."
        )

except Exception as e:
    print(
        "Lambda repair skipped:",
        e
    )


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\nFINAL TEST RESULTS")
print("-" * 70)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

report_path = os.path.join(
    EVALUATION_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BIRD CLASSIFICATION CNN\n"
    )

    f.write(
        "FINAL TEST CLASSIFICATION REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CREATING CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(16, 16)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    colorbar=False
)

plt.title(
    "Bird Classification CNN - Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    EVALUATION_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_data = pd.DataFrame(
    {
        "actual_class": [
            class_names[i]
            for i in y_true
        ],

        "predicted_class": [
            class_names[i]
            for i in y_pred
        ],

        "correct": (
            y_true == y_pred
        )
    }
)

prediction_path = os.path.join(
    EVALUATION_DIR,
    "test_predictions.csv"
)

prediction_data.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

correct_predictions = np.sum(
    y_true == y_pred
)

total_predictions = len(
    y_true
)

incorrect_predictions = (
    total_predictions
    - correct_predictions
)

summary_path = os.path.join(
    EVALUATION_DIR,
    "evaluation_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BIRD CLASSIFICATION CNN - FINAL EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total test images : "
        f"{total_predictions}\n"
    )

    f.write(
        f"Correct predictions : "
        f"{correct_predictions}\n"
    )

    f.write(
        f"Incorrect predictions : "
        f"{incorrect_predictions}\n"
    )

    f.write(
        f"Test loss : "
        f"{test_loss:.4f}\n"
    )

    f.write(
        f"Test accuracy : "
        f"{test_accuracy * 100:.2f}%\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nTest Accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)

print("\nFiles generated:")

print(
    cm_path
)

print(
    report_path
)

print(
    prediction_path
)

print(
    summary_path
)
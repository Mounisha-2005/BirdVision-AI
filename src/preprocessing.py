from pathlib import Path
import tensorflow as tf

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "Desktop"
    / "Bird_Classification_CNN"
    / "dataset"
)

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

# ============================================================
# 2. PARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("=" * 70)
print("CNN DATA PREPROCESSING")
print("=" * 70)

print("\nDATASET PATHS")
print("-" * 70)
print("Training   :", TRAIN_DIR)
print("Validation :", VALID_DIR)
print("Testing    :", TEST_DIR)

# ============================================================
# 3. CHECK DATASET DIRECTORIES
# ============================================================

for directory in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{directory}"
        )

print("\nDataset directories verified successfully.")

# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# ============================================================
# 5. LOAD VALIDATION DATA
# ============================================================

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# 6. LOAD TEST DATA
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# 7. CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nCLASS INFORMATION")
print("-" * 70)
print("Number of classes :", len(class_names))

for i, class_name in enumerate(class_names):
    print(f"{i:2d} : {class_name}")

# ============================================================
# 8. VERIFY CLASS CONSISTENCY
# ============================================================

if class_names != valid_ds.class_names:
    raise ValueError("Training and validation class mappings do not match.")

if class_names != test_ds.class_names:
    raise ValueError("Training and test class mappings do not match.")

print("\nClass mapping verified successfully.")

# ============================================================
# 9. NORMALIZATION
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

valid_ds = valid_ds.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# 10. PERFORMANCE OPTIMIZATION
# ============================================================

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# 11. VERIFY ONE BATCH
# ============================================================

print("\nBATCH VERIFICATION")
print("-" * 70)

for images, labels in train_ds.take(1):

    print("Image batch shape :", images.shape)
    print("Label batch shape :", labels.shape)
    print("Image data type   :", images.dtype)

    print(
        "Pixel minimum     :",
        float(tf.reduce_min(images))
    )

    print(
        "Pixel maximum     :",
        float(tf.reduce_max(images))
    )

    print(
        "Pixel mean        :",
        float(tf.reduce_mean(images))
    )

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nPreprocessing operations:")
print("1. Images loaded from directory")
print("2. Labels automatically assigned")
print("3. Images resized/enforced to 224 x 224")
print("4. RGB images converted to TensorFlow tensors")
print("5. Pixel values normalized from [0,255] to [0,1]")
print("6. Training data shuffled")
print("7. Validation data not shuffled")
print("8. Test data not shuffled")
print("9. Dataset pipeline optimized using prefetch")
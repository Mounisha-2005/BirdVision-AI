import tensorflow as tf
from pathlib import Path

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

# ============================================================
# 2. PARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("=" * 70)
print("DATA AUGMENTATION")
print("=" * 70)

# ============================================================
# 3. LOAD TRAINING DATA
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
# 4. LOAD VALIDATION DATA
# ============================================================

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nNumber of classes:", len(class_names))

# ============================================================
# 5. NORMALIZATION
# ============================================================

normalization = tf.keras.layers.Rescaling(1.0 / 255)

# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        mode="horizontal"
    ),

    tf.keras.layers.RandomRotation(
        factor=0.08
    ),

    tf.keras.layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10
    ),

    tf.keras.layers.RandomContrast(
        factor=0.10
    )
], name="data_augmentation")

# ============================================================
# 7. APPLY AUGMENTATION + NORMALIZATION
# ============================================================

def preprocess_training(images, labels):
    images = normalization(images)
    images = data_augmentation(images, training=True)
    return images, labels


def preprocess_validation(images, labels):
    images = normalization(images)
    return images, labels


train_ds = train_ds.map(
    preprocess_training,
    num_parallel_calls=tf.data.AUTOTUNE
)

valid_ds = valid_ds.map(
    preprocess_validation,
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# 8. PREFETCH
# ============================================================

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# 9. VERIFY TRAINING BATCH
# ============================================================

print("\nTRAINING BATCH VERIFICATION")
print("-" * 70)

for images, labels in train_ds.take(1):

    print("Image batch shape :", images.shape)
    print("Label batch shape :", labels.shape)
    print("Data type         :", images.dtype)

    print(
        "Minimum pixel     :",
        float(tf.reduce_min(images))
    )

    print(
        "Maximum pixel     :",
        float(tf.reduce_max(images))
    )

# ============================================================
# 10. VERIFY VALIDATION BATCH
# ============================================================

print("\nVALIDATION BATCH VERIFICATION")
print("-" * 70)

for images, labels in valid_ds.take(1):

    print("Image batch shape :", images.shape)
    print("Label batch shape :", labels.shape)
    print("Data type         :", images.dtype)

    print(
        "Minimum pixel     :",
        float(tf.reduce_min(images))
    )

    print(
        "Maximum pixel     :",
        float(tf.reduce_max(images))
    )

# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATA AUGMENTATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nAugmentation applied to TRAINING data:")
print("1. Random horizontal flip")
print("2. Random rotation")
print("3. Random zoom")
print("4. Random contrast")

print("\nValidation data:")
print("No augmentation applied.")

print("\nNormalization:")
print("Pixel range converted from [0,255] to [0,1].")
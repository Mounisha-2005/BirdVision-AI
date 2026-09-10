import tensorflow as tf
from pathlib import Path

# ============================================================
# CNN MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 20

print("=" * 70)
print("CNN MODEL ARCHITECTURE")
print("=" * 70)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="data_augmentation")


# ============================================================
# CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    # Input
    tf.keras.layers.Input(
        shape=(224, 224, 3)
    ),

    # Normalization
    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # Augmentation
    data_augmentation,

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # BLOCK 4
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.40
    ),

    # 20 bird classes
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nMODEL SUMMARY")
print("-" * 70)

model.summary()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

print("\nMODEL CONFIGURATION")
print("-" * 70)

print("Input size       :", "224 × 224 × 3")
print("Number of classes:", NUM_CLASSES)
print("Optimizer        :", "Adam")
print("Learning rate    :", "0.001")
print("Loss function    :", "Sparse Categorical Crossentropy")
print("Output activation:", "Softmax")
print("Dropout          :", "0.40")


# ============================================================
# OUTPUT SHAPE
# ============================================================

print("\nOUTPUT VERIFICATION")
print("-" * 70)

print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)

print("\n" + "=" * 70)
print("CNN MODEL CREATED SUCCESSFULLY")
print("=" * 70)
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
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

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42


# ============================================================
# GPU / DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("BIRD CLASSIFICATION CNN - MODEL TRAINING")
print("=" * 70)

print("\nDEVICE INFORMATION")
print("-" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("No TensorFlow GPU detected.")
    print("Training will use CPU.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="int"
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nDATASET INFORMATION")
print("-" * 70)
print("Number of classes :", num_classes)

for i, class_name in enumerate(class_names):
    print(f"{i:2d} : {class_name}")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=4)

print("\nClass names saved to:")
print(class_names_path)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),

        tf.keras.layers.Lambda(
            clip_to_unit_range
        )
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD CNN MODEL
# ============================================================
# ============================================================
# LOAD PREVIOUSLY TRAINED MODEL
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_bird_cnn.keras"
)

if not os.path.exists(best_model_path):
    raise FileNotFoundError(
        f"Saved model not found: {best_model_path}"
    )

print("\n" + "=" * 70)
print("LOADING PREVIOUSLY TRAINED MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    best_model_path,
    safe_mode=False
)
# ------------------------------------------------------------
# FIX OLD SAVED MODEL'S LAMBDA LAYER
# ------------------------------------------------------------

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)

augmentation_model = model.get_layer("data_augmentation")
clip_layer = augmentation_model.layers[-1]

if isinstance(clip_layer, tf.keras.layers.Lambda):
    clip_layer.function = clip_to_unit_range

print("Old Lambda layer repaired successfully.")
print("Previously trained model loaded successfully.")
print("Training will continue from Epoch 32.")


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_bird_cnn.keras"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(
        RESULTS_DIR,
        "training_history.csv"
    )
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)

# Test the repaired model before starting training
for images, labels in train_ds.take(1):
    _ = model(images, training=False)

print("MODEL TEST PASSED - Ready to resume training.")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    initial_epoch=31,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_bird_cnn.keras"
)

model.save(final_model_path)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nBest model saved at:")
print(best_model_path)

print("\nFinal model saved at:")
print(final_model_path)


# ============================================================
# TRAINING HISTORY
# ============================================================

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(accuracy) + 1)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")

plt.legend()

plt.grid(True)

accuracy_path = os.path.join(
    RESULTS_DIR,
    "training_validation_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")

plt.legend()

plt.grid(True)

loss_path = os.path.join(
    RESULTS_DIR,
    "training_validation_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL TRAINING RESULTS
# ============================================================

best_epoch = (
    val_accuracy.index(
        max(val_accuracy)
    ) + 1
)

best_val_accuracy = max(val_accuracy)

best_train_accuracy = accuracy[
    best_epoch - 1
]

print("\n" + "=" * 70)
print("TRAINING RESULTS")
print("=" * 70)

print(
    f"Total epochs executed : {len(accuracy)}"
)

print(
    f"Best epoch            : {best_epoch}"
)

print(
    f"Training accuracy     : "
    f"{best_train_accuracy:.4f}"
)

print(
    f"Validation accuracy   : "
    f"{best_val_accuracy:.4f}"
)

print("\nTraining accuracy graph:")
print(accuracy_path)

print("\nTraining loss graph:")
print(loss_path)

print("\n" + "=" * 70)
print("STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 70)
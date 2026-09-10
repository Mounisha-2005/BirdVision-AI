from pathlib import Path
from shutil import copy2
from datetime import datetime

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "Desktop"
    / "Bird_Classification_CNN"
    / "dataset"
)

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

BACKUP_DIR = PROJECT_DIR / "results" / "dataset_cleanup_backup"

MARKER_FILE = PROJECT_DIR / "results" / "cleanup_complete.txt"

CLASS_NAME = "ABBOTTS BABBLER"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# SAFETY CHECK
# ============================================================

if MARKER_FILE.exists():
    print("=" * 70)
    print("CLEANUP ALREADY COMPLETED")
    print("=" * 70)
    print(f"Marker found: {MARKER_FILE}")
    print("\nNo files were changed.")
    print("If you need to run the cleanup again, contact me first.")
    raise SystemExit


# ============================================================
# REQUIRED DIRECTORIES
# ============================================================

train_class_dir = TRAIN_DIR / CLASS_NAME
valid_class_dir = VALID_DIR / CLASS_NAME
test_class_dir = TEST_DIR / CLASS_NAME

for directory in [
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    train_class_dir,
    valid_class_dir,
    test_class_dir
]:

    if not directory.exists():
        raise FileNotFoundError(
            f"Required directory not found:\n{directory}"
        )


# ============================================================
# FILES IDENTIFIED DURING DUPLICATE DETECTION
# ============================================================

TRAIN_DUPLICATE = train_class_dir / "005.jpg"

VALID_DUPLICATE_1 = valid_class_dir / "1.jpg"
VALID_DUPLICATE_2 = valid_class_dir / "2.jpg"


# ============================================================
# VERIFY EXPECTED DUPLICATES EXIST
# ============================================================

required_files = [
    TRAIN_DUPLICATE,
    VALID_DUPLICATE_1,
    VALID_DUPLICATE_2
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nExpected duplicate file was not found:\n{file_path}\n\n"
            "Cleanup stopped. No files were modified."
        )


# ============================================================
# CREATE BACKUP DIRECTORIES
# ============================================================

backup_train = BACKUP_DIR / "train" / CLASS_NAME
backup_valid = BACKUP_DIR / "valid" / CLASS_NAME

backup_train.mkdir(parents=True, exist_ok=True)
backup_valid.mkdir(parents=True, exist_ok=True)


# ============================================================
# BACKUP FUNCTION
# ============================================================

def backup_file(source, destination):

    copy2(source, destination)

    print(f"BACKUP CREATED:")
    print(f"  {destination}")


# ============================================================
# 1. BACKUP AND REMOVE TRAINING DUPLICATE
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — REMOVE TRAINING INTERNAL DUPLICATE")
print("=" * 70)

backup_train_duplicate = backup_train / TRAIN_DUPLICATE.name

backup_file(
    TRAIN_DUPLICATE,
    backup_train_duplicate
)

TRAIN_DUPLICATE.unlink()

print(f"\nRemoved from training:")
print(f"  {TRAIN_DUPLICATE}")


# ============================================================
# 2. BACKUP AND REMOVE VALIDATION DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — REMOVE VALIDATION/TEST DUPLICATES")
print("=" * 70)

validation_duplicates = [
    VALID_DUPLICATE_1,
    VALID_DUPLICATE_2
]

for duplicate_file in validation_duplicates:

    backup_file(
        duplicate_file,
        backup_valid / duplicate_file.name
    )

    duplicate_file.unlink()

    print(f"\nRemoved from validation:")
    print(f"  {duplicate_file}")


# ============================================================
# 3. SELECT TWO UNIQUE TRAINING IMAGES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — SELECT REPLACEMENT VALIDATION IMAGES")
print("=" * 70)

excluded_files = {
    "004.jpg",
    "005.jpg"
}

available_train_images = sorted(
    file
    for file in train_class_dir.iterdir()
    if file.is_file()
    and file.suffix.lower() in IMAGE_EXTENSIONS
    and file.name not in excluded_files
)

if len(available_train_images) < 2:

    raise RuntimeError(
        "Not enough training images available for replacement."
    )


replacement_images = available_train_images[:2]

print("\nSelected replacement images:")

for image in replacement_images:
    print(f"  {image}")


# ============================================================
# 4. BACKUP AND MOVE REPLACEMENT IMAGES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — MOVE REPLACEMENT IMAGES")
print("=" * 70)

for image in replacement_images:

    # Backup original training image
    backup_destination = backup_train / image.name

    backup_file(
        image,
        backup_destination
    )

    # Move image from train to validation
    destination = valid_class_dir / image.name

    if destination.exists():

        raise FileExistsError(
            f"\nDestination already exists:\n{destination}\n"
            "Cleanup stopped."
        )

    image.rename(destination)

    print("\nMoved:")
    print(f"  FROM: {image}")
    print(f"  TO  : {destination}")


# ============================================================
# 5. WRITE CLEANUP LOG
# ============================================================

log_file = PROJECT_DIR / "results" / "dataset_cleanup_log.txt"

log_file.parent.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(log_file, "w", encoding="utf-8") as file:

    file.write("BIRD CLASSIFICATION DATASET CLEANUP LOG\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Cleanup date/time: {timestamp}\n\n")

    file.write("TRAINING DUPLICATE REMOVED:\n")
    file.write(f"{TRAIN_DUPLICATE}\n\n")

    file.write("VALIDATION DUPLICATES REMOVED:\n")
    file.write(f"{VALID_DUPLICATE_1}\n")
    file.write(f"{VALID_DUPLICATE_2}\n\n")

    file.write("TRAINING IMAGES MOVED TO VALIDATION:\n")

    for image in replacement_images:
        file.write(f"{image}\n")

    file.write("\nTEST SET:\n")
    file.write("No test images were modified.\n")


# ============================================================
# 6. CREATE COMPLETION MARKER
# ============================================================

with open(MARKER_FILE, "w", encoding="utf-8") as file:

    file.write(
        f"Dataset cleanup completed successfully on {timestamp}\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATASET CLEANUP COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Changes made:

1. One internal training duplicate was removed.
2. Two validation images duplicated in the test set were removed.
3. Two unique training images were moved to validation.
4. TEST SET was NOT modified.
5. Backup copies were created.
6. Cleanup log was created.
7. Completion marker was created.

IMPORTANT:
Do NOT train the CNN yet.

The next step is to VERIFY the cleaned dataset.
""")
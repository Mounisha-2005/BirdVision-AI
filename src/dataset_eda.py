from pathlib import Path
from PIL import Image

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "Desktop" / "Bird_Classification_CNN" / "dataset" / "train"
VALID_DIR = DATASET_DIR / "Desktop" / "Bird_Classification_CNN" / "dataset" / "valid"
TEST_DIR = DATASET_DIR / "Desktop" / "Bird_Classification_CNN" / "dataset" / "test"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_class_folders(directory):
    """Return class folders inside a dataset split."""
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return []

    return sorted(
        folder for folder in directory.iterdir()
        if folder.is_dir()
    )


def count_images(folder):
    """Count supported image files recursively."""
    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def analyze_split(name, directory):
    """Analyze one dataset split."""
    print("\n" + "=" * 60)
    print(f"{name.upper()} DATASET")
    print("=" * 60)

    classes = get_class_folders(directory)

    print(f"Path: {directory}")
    print(f"Number of classes: {len(classes)}")

    total_images = 0

    for class_folder in classes:
        image_count = count_images(class_folder)
        total_images += image_count
        print(f"{class_folder.name}: {image_count}")

    print(f"\nTotal images in {name}: {total_images}")

    return classes, total_images


# Analyze train, validation and test
train_classes, train_total = analyze_split("train", TRAIN_DIR)
valid_classes, valid_total = analyze_split("valid", VALID_DIR)
test_classes, test_total = analyze_split("test", TEST_DIR)


# Overall summary
print("\n" + "=" * 60)
print("OVERALL DATASET SUMMARY")
print("=" * 60)

print(f"Training images   : {train_total}")
print(f"Validation images : {valid_total}")
print(f"Test images       : {test_total}")
print(f"Total images      : {train_total + valid_total + test_total}")

print(f"\nTrain classes     : {len(train_classes)}")
print(f"Valid classes     : {len(valid_classes)}")
print(f"Test classes      : {len(test_classes)}")


# Compare class names between splits
train_names = {folder.name for folder in train_classes}
valid_names = {folder.name for folder in valid_classes}
test_names = {folder.name for folder in test_classes}

print("\n" + "=" * 60)
print("CLASS CONSISTENCY CHECK")
print("=" * 60)

print(f"Classes in train but not valid: {len(train_names - valid_names)}")
print(f"Classes in train but not test : {len(train_names - test_names)}")
print(f"Classes in valid but not train: {len(valid_names - train_names)}")
print(f"Classes in test but not train : {len(test_names - train_names)}")
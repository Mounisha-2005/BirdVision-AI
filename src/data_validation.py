from pathlib import Path
from PIL import Image
from collections import Counter

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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(directory):

    return [
        file
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_images(name, directory):

    print("\n" + "=" * 70)
    print(f"{name.upper()} IMAGE VALIDATION")
    print("=" * 70)

    image_files = get_image_files(directory)

    print(f"Total images checked: {len(image_files)}")

    corrupted_images = []
    dimensions = Counter()
    modes = Counter()
    formats = Counter()

    for image_path in image_files:

        try:

            with Image.open(image_path) as img:

                # Verify image integrity
                img.verify()

            # Open again because verify() closes the image
            with Image.open(image_path) as img:

                dimensions[img.size] += 1
                modes[img.mode] += 1
                formats[img.format] += 1

        except Exception as e:

            corrupted_images.append(
                (str(image_path), str(e))
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nImage dimensions:")
    for dimension, count in dimensions.most_common():
        print(f"{dimension}: {count}")

    print("\nImage color modes:")
    for mode, count in modes.most_common():
        print(f"{mode}: {count}")

    print("\nImage formats:")
    for image_format, count in formats.most_common():
        print(f"{image_format}: {count}")

    print("\nCorrupted/unreadable images:")
    print(len(corrupted_images))

    if corrupted_images:

        print("\nCorrupted files:")

        for file_path, error in corrupted_images[:20]:
            print(file_path)
            print("Error:", error)

    return {
        "images": len(image_files),
        "corrupted": corrupted_images,
        "dimensions": dimensions,
        "modes": modes,
        "formats": formats
    }


# ============================================================
# RUN VALIDATION
# ============================================================

train_result = validate_images(
    "train",
    TRAIN_DIR
)

valid_result = validate_images(
    "valid",
    VALID_DIR
)

test_result = validate_images(
    "test",
    TEST_DIR
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION SUMMARY")
print("=" * 70)

print(
    f"Training corrupted images   : "
    f"{len(train_result['corrupted'])}"
)

print(
    f"Validation corrupted images : "
    f"{len(valid_result['corrupted'])}"
)

print(
    f"Test corrupted images       : "
    f"{len(test_result['corrupted'])}"
)

total_corrupted = (
    len(train_result["corrupted"])
    + len(valid_result["corrupted"])
    + len(test_result["corrupted"])
)

print(f"Total corrupted images      : {total_corrupted}")

if total_corrupted == 0:

    print("\nSTATUS: ALL IMAGES ARE VALID AND READABLE.")

else:

    print("\nSTATUS: CORRUPTED IMAGES FOUND.")
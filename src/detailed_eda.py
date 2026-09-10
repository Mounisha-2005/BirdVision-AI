from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
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

RESULTS_DIR = PROJECT_DIR / "results" / "eda"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# FUNCTIONS
# ============================================================

def get_images(directory):
    return [
        file
        for file in directory.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_class_counts(directory):
    counts = {}

    for class_folder in sorted(directory.iterdir()):
        if class_folder.is_dir():
            counts[class_folder.name] = len(get_images(class_folder))

    return counts


def analyze_dimensions(images):
    dimensions = []
    modes = Counter()
    formats = Counter()

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                dimensions.append(img.size)
                modes[img.mode] += 1
                formats[img.format] += 1
        except Exception:
            pass

    return dimensions, modes, formats


# ============================================================
# 1. BASIC DATASET STATISTICS
# ============================================================

train_counts = get_class_counts(TRAIN_DIR)
valid_counts = get_class_counts(VALID_DIR)
test_counts = get_class_counts(TEST_DIR)

train_images = get_images(TRAIN_DIR)
valid_images = get_images(VALID_DIR)
test_images = get_images(TEST_DIR)

print("=" * 70)
print("DETAILED EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDATASET SUMMARY")
print("-" * 70)

print(f"Number of classes : {len(train_counts)}")
print(f"Training images   : {len(train_images)}")
print(f"Validation images : {len(valid_images)}")
print(f"Test images       : {len(test_images)}")
print(
    f"Total images      : "
    f"{len(train_images) + len(valid_images) + len(test_images)}"
)


# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

print("\nTRAINING CLASS DISTRIBUTION")
print("-" * 70)

for class_name, count in train_counts.items():
    print(f"{class_name:<35} : {count}")


# ============================================================
# 3. CLASS BALANCE STATISTICS
# ============================================================

class_values = np.array(list(train_counts.values()))

minimum = class_values.min()
maximum = class_values.max()
average = class_values.mean()
std = class_values.std()
imbalance_ratio = maximum / minimum

print("\nCLASS BALANCE ANALYSIS")
print("-" * 70)

print(f"Minimum images/class : {minimum}")
print(f"Maximum images/class : {maximum}")
print(f"Average images/class : {average:.2f}")
print(f"Std deviation        : {std:.2f}")
print(f"Max/Min ratio        : {imbalance_ratio:.2f}")


# ============================================================
# 4. TRAINING CLASS DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(14, 7))

plt.bar(
    list(train_counts.keys()),
    list(train_counts.values())
)

plt.xlabel("Bird Class")
plt.ylabel("Number of Images")
plt.title("Training Dataset - Class Distribution")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. SPLIT DISTRIBUTION GRAPH
# ============================================================

split_names = [
    "Training",
    "Validation",
    "Test"
]

split_counts = [
    len(train_images),
    len(valid_images),
    len(test_images)
]

plt.figure(figsize=(8, 6))

plt.bar(
    split_names,
    split_counts
)

plt.xlabel("Dataset Split")
plt.ylabel("Number of Images")
plt.title("Dataset Split Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "split_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. IMAGE DIMENSIONS
# ============================================================

all_images = train_images + valid_images + test_images

dimensions, modes, formats = analyze_dimensions(all_images)

dimension_counter = Counter(dimensions)

print("\nIMAGE DIMENSIONS")
print("-" * 70)

for dimension, count in dimension_counter.items():
    print(f"{dimension} : {count} images")

print("\nCOLOR MODES")
print("-" * 70)

for mode, count in modes.items():
    print(f"{mode} : {count} images")

print("\nIMAGE FORMATS")
print("-" * 70)

for image_format, count in formats.items():
    print(f"{image_format} : {count} images")


# ============================================================
# 7. SAMPLE IMAGE GRID
# ============================================================

classes = sorted(train_counts.keys())

fig, axes = plt.subplots(
    4,
    5,
    figsize=(15, 12)
)

for ax, class_name in zip(axes.ravel(), classes):

    class_dir = TRAIN_DIR / class_name

    images = get_images(class_dir)

    if images:

        image_path = images[0]

        with Image.open(image_path) as img:
            ax.imshow(img.convert("RGB"))

        ax.set_title(
            class_name,
            fontsize=9
        )

    ax.axis("off")

plt.suptitle(
    "Representative Images from 20 Bird Classes",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "sample_images.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. PIXEL INTENSITY ANALYSIS
# ============================================================

sample_images = train_images[:500]

pixel_values = []

for image_path in sample_images:

    try:

        with Image.open(image_path) as img:

            img = img.convert("RGB")
            img_array = np.asarray(img)

            pixel_values.append(
                img_array.reshape(-1, 3)
            )

    except Exception:
        pass


if pixel_values:

    pixels = np.concatenate(pixel_values)

    mean_rgb = pixels.mean(axis=0)
    std_rgb = pixels.std(axis=0)

    print("\nPIXEL STATISTICS")
    print("-" * 70)

    print(
        f"Mean RGB : "
        f"R={mean_rgb[0]:.2f}, "
        f"G={mean_rgb[1]:.2f}, "
        f"B={mean_rgb[2]:.2f}"
    )

    print(
        f"Std RGB  : "
        f"R={std_rgb[0]:.2f}, "
        f"G={std_rgb[1]:.2f}, "
        f"B={std_rgb[2]:.2f}"
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        pixels[:, 0],
        bins=50,
        alpha=0.5,
        label="Red"
    )

    plt.hist(
        pixels[:, 1],
        bins=50,
        alpha=0.5,
        label="Green"
    )

    plt.hist(
        pixels[:, 2],
        bins=50,
        alpha=0.5,
        label="Blue"
    )

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("RGB Pixel Intensity Distribution")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pixel_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. FINAL EDA REPORT
# ============================================================

report_path = RESULTS_DIR / "eda_summary.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("DETAILED EDA SUMMARY\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Number of classes : {len(train_counts)}\n")
    file.write(f"Training images   : {len(train_images)}\n")
    file.write(f"Validation images : {len(valid_images)}\n")
    file.write(f"Test images       : {len(test_images)}\n")
    file.write(
        f"Total images      : "
        f"{len(all_images)}\n\n"
    )

    file.write("CLASS BALANCE\n")
    file.write("-" * 60 + "\n")

    file.write(
        f"Minimum images/class : {minimum}\n"
    )

    file.write(
        f"Maximum images/class : {maximum}\n"
    )

    file.write(
        f"Average images/class : {average:.2f}\n"
    )

    file.write(
        f"Standard deviation   : {std:.2f}\n"
    )

    file.write(
        f"Maximum/Minimum ratio: {imbalance_ratio:.2f}\n\n"
    )

    file.write("IMAGE DIMENSIONS\n")
    file.write("-" * 60 + "\n")

    for dimension, count in dimension_counter.items():
        file.write(
            f"{dimension}: {count} images\n"
        )

    file.write("\nCOLOR MODES\n")
    file.write("-" * 60 + "\n")

    for mode, count in modes.items():
        file.write(
            f"{mode}: {count} images\n"
        )

    file.write("\nIMAGE FORMATS\n")
    file.write("-" * 60 + "\n")

    for image_format, count in formats.items():
        file.write(
            f"{image_format}: {count} images\n"
        )


print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved in:")
print(RESULTS_DIR)
from pathlib import Path
from hashlib import md5
from collections import defaultdict

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

    return sorted(
        file
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# CALCULATE FILE HASH
# ============================================================

def calculate_hash(file_path):

    hash_md5 = md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(8192), b""):

            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# ============================================================
# COLLECT HASHES
# ============================================================

def collect_hashes(split_name, directory):

    print(f"\nScanning {split_name}...")

    image_files = get_image_files(directory)

    hashes = defaultdict(list)

    for image_path in image_files:

        image_hash = calculate_hash(image_path)

        hashes[image_hash].append(image_path)

    print(f"Images scanned: {len(image_files)}")
    print(f"Unique hashes : {len(hashes)}")

    return hashes


# ============================================================
# COLLECT TRAIN / VALID / TEST HASHES
# ============================================================

train_hashes = collect_hashes(
    "TRAIN",
    TRAIN_DIR
)

valid_hashes = collect_hashes(
    "VALIDATION",
    VALID_DIR
)

test_hashes = collect_hashes(
    "TEST",
    TEST_DIR
)


# ============================================================
# FIND DUPLICATES WITHIN A SPLIT
# ============================================================

def find_internal_duplicates(split_name, hashes):

    duplicate_groups = [
        paths
        for paths in hashes.values()
        if len(paths) > 1
    ]

    duplicate_image_count = sum(
        len(paths) - 1
        for paths in duplicate_groups
    )

    print("\n" + "=" * 70)
    print(f"{split_name} INTERNAL DUPLICATES")
    print("=" * 70)

    print(f"Duplicate groups : {len(duplicate_groups)}")
    print(f"Duplicate images : {duplicate_image_count}")

    if duplicate_groups:

        print("\nDuplicate files:")

        for group in duplicate_groups:

            print("\n---")

            for path in group:
                print(path)


# ============================================================
# FIND DUPLICATES BETWEEN TWO SPLITS
# ============================================================

def compare_splits(name1, hashes1, name2, hashes2):

    common_hashes = set(hashes1.keys()) & set(hashes2.keys())

    print("\n" + "=" * 70)
    print(f"{name1} vs {name2}")
    print("=" * 70)

    print(f"Common image hashes: {len(common_hashes)}")

    if common_hashes:

        total_matches = 0

        for image_hash in common_hashes:

            for path1 in hashes1[image_hash]:

                for path2 in hashes2[image_hash]:

                    total_matches += 1

                    print("\nDuplicate image:")
                    print(f"{name1}: {path1}")
                    print(f"{name2}: {path2}")

        print(f"\nTotal matching image pairs: {total_matches}")

    else:

        print("No exact duplicate images found.")


# ============================================================
# INTERNAL DUPLICATES
# ============================================================

find_internal_duplicates(
    "TRAIN",
    train_hashes
)

find_internal_duplicates(
    "VALIDATION",
    valid_hashes
)

find_internal_duplicates(
    "TEST",
    test_hashes
)


# ============================================================
# CROSS-SPLIT DUPLICATES
# ============================================================

compare_splits(
    "TRAIN",
    train_hashes,
    "VALIDATION",
    valid_hashes
)

compare_splits(
    "TRAIN",
    train_hashes,
    "TEST",
    test_hashes
)

compare_splits(
    "VALIDATION",
    valid_hashes,
    "TEST",
    test_hashes
)


# ============================================================
# FINAL RESULT
# ============================================================

train_valid = set(train_hashes.keys()) & set(valid_hashes.keys())
train_test = set(train_hashes.keys()) & set(test_hashes.keys())
valid_test = set(valid_hashes.keys()) & set(test_hashes.keys())

print("\n" + "=" * 70)
print("FINAL DUPLICATE CHECK")
print("=" * 70)

print(
    f"Train ↔ Validation duplicate hashes : "
    f"{len(train_valid)}"
)

print(
    f"Train ↔ Test duplicate hashes       : "
    f"{len(train_test)}"
)

print(
    f"Validation ↔ Test duplicate hashes : "
    f"{len(valid_test)}"
)

total_cross_split = (
    len(train_valid)
    + len(train_test)
    + len(valid_test)
)

print(
    f"\nTotal cross-split duplicate hash groups: "
    f"{total_cross_split}"
)

if total_cross_split == 0:

    print("\nSTATUS: NO EXACT DUPLICATES FOUND ACROSS DATA SPLITS.")

else:

    print("\nSTATUS: DUPLICATES FOUND — DATA LEAKAGE MUST BE INVESTIGATED.")
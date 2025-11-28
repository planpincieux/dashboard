"""
Script to update image file paths from old FMS structure to new GMG FTP structure.

Old structure: /ppcx/fms_data/Dati/HiRes/{Tele|Wide}/{Year}/{Month}/{Day}/{filename}
New structure: /data/ppcx_data/{Camera_1|Camera_2}/Archivio/{Year}/{filename}

Camera mapping:
- PPCX_Tele → Camera_2
- PPCX_Wide → Camera_1
"""

import os
import re
from pathlib import Path

import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()


from ppcx_app.models import Image

# Camera name to folder mapping
CAMERA_FOLDER_MAP = {
    "PPCX_Tele": "Camera_2",
    "PPCX_Wide": "Camera_1",
}

BASE_PATH = "/data/ppcx_data"


# Cache for all files indexed by camera folder
# Structure: {camera_folder: {filename: full_path, ...}}
_camera_files_cache = {}


def get_all_camera_files(base_path, camera_folder):
    """
    Get and cache all files from the camera's Archivio folder.

    Args:
        base_path: Base directory path
        camera_folder: Camera folder name (Camera_1 or Camera_2)

    Returns:
        Dictionary mapping filenames to full file paths
    """
    if camera_folder not in _camera_files_cache:
        archivio_path = Path(base_path) / camera_folder / "Archivio"

        if not archivio_path.exists():
            print(f"Warning: Archivio path does not exist: {archivio_path}")
            _camera_files_cache[camera_folder] = {}
            return _camera_files_cache[camera_folder]

        print(f"Building file cache for {camera_folder}...")
        file_dict = {}

        # Get all jpg files recursively
        for file_path in archivio_path.rglob("*.jpg"):
            filename = file_path.name
            # Store full path; if duplicate filename exists, keep the first one found
            if filename not in file_dict:
                file_dict[filename] = str(file_path)

        _camera_files_cache[camera_folder] = file_dict
        print(f"Cached {len(file_dict)} files for {camera_folder}")

    return _camera_files_cache[camera_folder]


def find_matching_image(
    base_path,
    camera_folder,
    original_filename,
):
    """
    Try to find an image with a matching date/time pattern.
    Handles cases where camera ID was added or removed from filename.

    Args:
        base_path: Base directory path
        camera_folder: Camera folder name (Camera_1 or Camera_2)
        original_filename: Original filename to match

    Example patterns:
    - PPCX_2015_02_02_15_30_03.jpg
    - PPCX_2_2015_02_02_15_30_03.jpg
    """
    # Extract date/time pattern from original filename
    # Pattern: YYYY_MM_DD_HH_MM_SS
    date_time_pattern = re.search(
        r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", original_filename
    )

    if not date_time_pattern:
        return None

    date_time = date_time_pattern.group(1)

    # Get all files for this camera
    all_files = get_all_camera_files(base_path, camera_folder)

    # Fast in-memory search through filenames
    for filename, full_path in all_files.items():
        if date_time in filename:
            return full_path

    return None


def update_image_paths(dry_run=True, delete_not_found=False):
    """Update all image paths from old to new structure.

    Args:
        dry_run: If True, don't make any changes to the database
        delete_not_found: If True, delete images from database if file not found
    """

    images = Image.objects.select_related("camera").all()
    total = images.count()
    updated = 0
    not_found = 0
    skipped = 0
    found_auto = 0
    deleted = 0
    not_found_paths = []

    print(f"Processing {total} images...")
    print(f"Dry run: {dry_run}")
    print(f"Delete not found: {delete_not_found}\n")

    for idx, image in enumerate(images, 1):
        old_path = image.file_path
        filename = os.path.basename(old_path)

        # Skip if already in new format
        if old_path.startswith(BASE_PATH):
            skipped += 1
            continue

        # Get camera folder
        camera_name = image.camera.camera_name
        camera_folder = CAMERA_FOLDER_MAP.get(camera_name)
        if not camera_folder:
            print(
                f"[{idx}/{total}] SKIP: Unknown camera '{camera_name}' for image {image.id}"
            )
            skipped += 1
            continue

        # Extract year from Image timestamp
        year = image.datetime.year

        # Build new path

        if year < 2025:
            new_path = f"{BASE_PATH}/{camera_folder}/Archivio/{year}/{filename}"
        else:
            new_path = f"{BASE_PATH}/{camera_folder}/Archivio/{filename}"

        # If same name is not found, try to find a matching file with partial name match
        if not os.path.exists(new_path):
            new_path = find_matching_image(
                BASE_PATH,
                camera_folder,
                filename,
            )
            if new_path and os.path.exists(new_path):
                found_auto += 1

        # Check if new path exists
        if not new_path or not os.path.exists(new_path):
            print(
                f"[{idx}/{total}] NOT FOUND: {old_path} → {new_path if new_path else 'No match found'}"
            )
            not_found += 1
            not_found_paths.append(old_path)

            # Delete from database if option is enabled
            if delete_not_found and not dry_run:
                image.delete()
                deleted += 1
                print(f"[{idx}/{total}] DELETED from database: Image ID {image.id}")

            continue

        # Update database
        if not dry_run:
            image.file_path = new_path
            image.save(update_fields=["file_path"])

        updated += 1
        if idx % 500 == 0:
            print(
                f"[{idx}/{total}] Processed: {updated} updated ({found_auto} auto found), {not_found} not found, {skipped} skipped, {deleted} deleted"
            )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"Total images:     {total}")
    print(f"Updated:          {updated}")
    print(f"  - Exact match:  {updated - found_auto}")
    print(f"  - Found auto:   {found_auto}")
    print(f"Not found:        {not_found}")
    print(f"Skipped:          {skipped}")
    if delete_not_found:
        print(f"Deleted:          {deleted}")
    print(f"{'=' * 60}")

    # Log all not found paths
    if not_found_paths:
        print(f"\n{'=' * 60}")
        print(f"NOT FOUND IMAGES ({len(not_found_paths)}):")
        print(f"{'=' * 60}")
        for path in not_found_paths:
            print(f"  {path}")
        print(f"{'=' * 60}")

    if dry_run:
        print("\n⚠️  DRY RUN - No changes were made to the database")
        print("Run with dry_run=False to apply changes")
        if delete_not_found:
            print(f"⚠️  delete_not_found=True: {not_found} images would be deleted")


if __name__ == "__main__":
    # First run with dry_run=True to check what will happen
    # update_image_paths(dry_run=True, delete_not_found=False)

    # Uncomment to actually update the database
    update_image_paths(dry_run=False, delete_not_found=False)

    # Uncomment to update AND delete not found images
    # update_image_paths(dry_run=False, delete_not_found=True)

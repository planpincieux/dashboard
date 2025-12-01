"""
One-time script to populate file_size_bytes for existing images.
"""

import os
from pathlib import Path

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from ppcx_app.models import Image


def update_file_sizes(dry_run=True):
    """Update file_size_bytes for all images."""
    images = Image.objects.all()
    total = images.count()
    updated = 0
    not_found = 0

    print(f"Processing {total} images...")
    print(f"Dry run: {dry_run}\n")

    for idx, img in enumerate(images.iterator(), 1):
        if img.file_size_bytes is not None:
            continue  # Already has size

        try:
            file_path = Path(img.file_path)
            if file_path.exists():
                size = file_path.stat().st_size
                if not dry_run:
                    img.file_size_bytes = size
                    img.save(update_fields=["file_size_bytes"])
                updated += 1
            else:
                not_found += 1
        except (OSError, PermissionError) as e:
            print(f"[{idx}/{total}] ERROR: {img.file_path}: {e}")
            not_found += 1
            continue

        if idx % 100 == 0:
            print(f"[{idx}/{total}] Updated: {updated}, Not found: {not_found}")

    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"Total: {total}")
    print(f"Updated: {updated}")
    print(f"Not found: {not_found}")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n⚠️  DRY RUN - Run with dry_run=False to apply changes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update file_size_bytes for existing images in the database."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry run without making changes to the database.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Running in DRY RUN mode. No changes will be made to the database.\n")
    else:
        input(
            "WARNING: You are about to update the database. Press Enter to continue or Ctrl+C to abort..."
        )

    update_file_sizes(dry_run=args.dry_run)

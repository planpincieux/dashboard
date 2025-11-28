import os
from pathlib import Path

import django
import numpy as np
import pandas as pd
from django.contrib.gis.geos import GEOSGeometry
from django.db import IntegrityError
from ppcollapse import logger
from ppcollapse.utils.database import (
    fetch_image_ids,
    fetch_image_metadata_by_ids,
)
from shapely import wkt as shapely_wkt
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

# ---------------------------
# Initialize Django ORM
# ---------------------------
# Make sure this points to your project settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()


from ppcx_app.models import Collapse, Image


def write_geom_wkt_to_db(
    db_engine,
    image_id: int,
    geom: BaseGeometry,
    area: float | None = None,
    volume: float | None = None,
    table: str = "ppcx_app_collapse",
) -> int:
    """Insert geometry using Django ORM (uses ppcx_app.Collapse model). Returns inserted id."""
    # Convert shapely geom to WKT and then to GEOSGeometry for GeoDjango
    wkt = shapely_wkt.dumps(geom)
    try:
        # Get image instance via ORM
        img = Image.objects.get(pk=int(image_id))
    except Image.DoesNotExist:
        raise ValueError(f"Image with id {image_id} not found in Django DB")

    geos = GEOSGeometry(wkt)
    # ensure srid matches model (model uses srid=0)
    try:
        geos.srid = 0
    except Exception:
        # some GEOS versions may not allow setting srid for certain geometries; ignore if fails
        pass

    # Use provided area if available, otherwise compute from shapely
    area_val = float(area) if area is not None else float(geom.area)
    volume_val = float(volume) if volume is not None else None

    try:
        collapse = Collapse.objects.create(
            image=img,
            geom=geos,
            area=area_val,
            volume=volume_val,
        )
    except IntegrityError as e:
        logger.error("IntegrityError creating Collapse for image %s: %s", image_id, e)
        raise

    logger.debug("Created Collapse id=%s (image=%s)", collapse.pk, image_id)
    return int(collapse.pk)


def read_collapse_volume_file(path: Path) -> pd.DataFrame:
    import re

    # regex: 6 mandatory columns then optional trailing flag that starts with %
    # allow numeric type or literal "NaN" for missing type
    pattern = re.compile(
        r"^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+((?:\d+|NaN))\s+(\d+)(?:\s+(%.*))?\s*$",
        re.IGNORECASE,
    )
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("%"):
            continue
        m = pattern.match(ln)
        if m:
            year, month, day_start, day_end, typ, size, flag = m.groups()
        else:
            parts = ln.split()
            if len(parts) < 6:
                raise ValueError(f"Unparseable line: {ln!r}")
            year, month, day_start, day_end, typ, size = parts[:6]
            flag = None
            if len(parts) > 6:
                tail = " ".join(parts[6:])
                flag = tail if tail.startswith("%") else None

        # normalize type: treat NaN (any case) as missing -> None
        typ_val = None if isinstance(typ, str) and typ.lower() == "nan" else int(typ)

        rows.append(
            {
                "date": pd.to_datetime(
                    f"{year}-{int(month):02d}-{int(day_end):02d}", format="%Y-%m-%d"
                ),
                "year": int(year),
                "month": int(month),
                "day_start": int(day_start),
                "day_end": int(day_end),
                "type": typ_val,
                "volume": int(size),
                "flag_raw": flag,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    for year_dir in sorted(shape_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        files = sorted(year_dir.glob(f"*{file_ext}"))
        logger.info(f"Processing year directory: {year_dir}, found {len(files)} files")

        # Check if a volume file exists
        year = year_dir.name.split("_")[0]
        volume_file = volume_file_dir / f"crolli_{year}.txt"
        if volume_file.exists():
            try:
                collapse_volume_df = read_collapse_volume_file(volume_file)
            except Exception as e:
                logger.error(f"Error reading volume file {volume_file}: {e}")
                raise e
            has_volume_file = True
        else:
            has_volume_file = False
        if not has_volume_file:
            logger.warning(
                f"No volume file found for year {year}, proceeding without volume data."
            )

        if not files:
            logger.warning(f"No shapefiles found in {year_dir}, skipping...")
            continue

        for file in tqdm(files, desc="Processing shapefiles"):
            logger.debug(f"Reading {file}...")

            # Skip shapefile with _vol in the name (they don't have geometry)
            if "_vol" in file.name:
                logger.info(f"Skipping {file} (volume shapefile)")
                continue

            # Extract date from filename
            try:
                date_str = file.stem.split("-")[0]
                date = pd.to_datetime(date_str, format="%Y%m%d").date()
            except Exception as e:
                logger.error(f"Error parsing date from {file}: {e}")
                raise ValueError(
                    f"Filename {file.name} does not match expected format."
                ) from e
                # continue

            # Fetch image IDs for the given date, ordered by acquisition time descending
            try:
                image_ids = fetch_image_ids(
                    db_engine,
                    date=date.isoformat(),
                    order_by="datetime DESC",
                )
                images_metadata = fetch_image_metadata_by_ids(
                    db_engine=db_engine, image_id=image_ids
                )
            except Exception as e:
                logger.error(f"Error fetching image IDs for {file}: {e}")
                continue

            # Pick up center image in the list
            idx = len(image_ids) // 2
            image_id = image_ids[idx]
            logger.debug(
                f"Selected image ID: {image_id} - {images_metadata.iloc[idx].datetime}"
            )

            # Read geometry from file
            try:
                geom = read_shapely_geom_from_file(file, invert_y=True)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue

            # Get volume from collapse_volume_df if available
            volume = None
            if has_volume_file:
                match = collapse_volume_df.loc[
                    (collapse_volume_df["date"].dt.date == date)
                ]
                if not match.empty:
                    volume = float(match.iloc[0]["volume"])
                    logger.debug(f"Found volume {volume} for date {date}")
                else:
                    logger.debug(f"No volume entry found for date {date}")

            # Insert geometry into DB and get id (using Django ORM now)
            area = np.round(geom.area, 5)
            try:
                collapse_id = write_geom_wkt_to_db(
                    db_engine=db_engine,
                    image_id=image_id,
                    geom=geom,
                    area=area,
                    volume=volume,
                )
                logger.debug(f"Inserted collapse id: {collapse_id}")
            except Exception as e:
                logger.error(f"Failed to insert collapse for {file}: {e}")
                continue

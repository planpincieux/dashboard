import logging
import os
from pathlib import Path

import h5py
import numpy as np
from django.http import Http404
from ppcx_app.models import DIC

logger = logging.getLogger(__name__)


def read_dic_h5(dic: DIC) -> dict[str, np.ndarray]:
    """
    Read the required datasets from the DIC HDF5 file.

    Returns: (points, vectors or None, magnitudes)
    Raises Http404 on errors or missing required datasets.
    """
    h5 = dic.result_file_path
    if not h5 or not os.path.exists(h5):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5, "r") as f:
            points = f["points"][()]
            vectors = f["vectors"][()]
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e

    if points is None or vectors is None:
        raise Http404("DIC HDF5 missing required datasets ('points' or 'vectors')")

    if len(points) == 0:
        raise Http404("DIC HDF5 contains empty datasets")

    # Compute magnitudes if missing
    if magnitudes is None:
        magnitudes = np.sqrt(np.sum(np.square(vectors), axis=1))

    data = {
        "points": points,
        "vectors": vectors,
        "magnitudes": magnitudes,
    }

    return data


def create_dic_h5(dic_data: dict, h5_file_path: Path) -> bool:
    """
    Create HDF5 file from DIC data dictionary.

    Args:
        dic_data: Dictionary with DIC data. Required keys: 'points', 'magnitudes'
                  Optional keys: 'vectors', 'max_magnitude', and any custom datasets
        h5_file_path: Path where to save the H5 file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        h5_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate required fields
        if "points" not in dic_data or "magnitudes" not in dic_data:
            logger.error("Missing required fields: 'points' and 'magnitudes'")
            return False

        with h5py.File(h5_file_path, "w") as f:
            # Required datasets
            f.create_dataset(
                "points",
                data=np.array(dic_data["points"], dtype="int32"),
            )
            f.create_dataset(
                "magnitudes",
                data=np.array(dic_data["magnitudes"], dtype="float32"),
            )

            # Optional standard datasets
            if "vectors" in dic_data:
                f.create_dataset(
                    "vectors",
                    data=np.array(dic_data["vectors"], dtype="float32"),
                )

            if "max_magnitude" in dic_data:
                max_mag = dic_data["max_magnitude"]
                if not isinstance(max_mag, (list, np.ndarray)):
                    max_mag = [max_mag]
                f.create_dataset(
                    "max_magnitude",
                    data=np.array(max_mag, dtype="float32"),
                )

            # Store any additional custom datasets
            reserved_keys = {"points", "vectors", "magnitudes", "max_magnitude"}
            for key, value in dic_data.items():
                if key not in reserved_keys:
                    try:
                        # Try to store as numpy array
                        if isinstance(value, (list, tuple)):
                            f.create_dataset(key, data=np.array(value))
                        elif isinstance(value, np.ndarray):
                            f.create_dataset(key, data=value)
                        elif isinstance(value, (int, float, str, bool)):
                            # Store scalar as attribute
                            f.attrs[key] = value
                        else:
                            logger.warning(
                                f"Skipping unsupported data type for key '{key}': {type(value)}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to store custom field '{key}': {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to create H5 file: {e}", exc_info=True)
        return False


def filter_dic_arrays(
    points: np.ndarray,
    vectors: np.ndarray | None,
    magnitudes: np.ndarray,
    *,
    filter_outliers: bool = True,
    tails_percentile: float = 0.01,
    min_velocity: float | None = None,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply filtering and subsampling to DIC arrays.

    Returns: x, y, u, v, mag (all float arrays)
    """
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    if vectors is not None:
        u = vectors[:, 0].astype(float)
        v = vectors[:, 1].astype(float)
    else:
        u = np.zeros_like(x)
        v = np.zeros_like(x)

    mag = magnitudes.astype(float)

    # Build mask and apply filters
    mask = np.ones_like(mag, dtype=bool)
    if filter_outliers and 0.0 < tails_percentile < 0.5:
        lo, hi = np.percentile(
            mag, [100.0 * tails_percentile, 100.0 * (1.0 - tails_percentile)]
        )
        mask &= (mag >= lo) & (mag <= hi)
    if min_velocity is not None and min_velocity > 0:
        mask &= mag >= min_velocity

    x = x[mask]
    y = y[mask]
    u = u[mask]
    v = v[mask]
    mag = mag[mask]

    if subsample is None or subsample < 1:
        subsample = 1
    if subsample > 1:
        idx = np.arange(0, len(x), subsample)
        x = x[idx]
        y = y[idx]
        u = u[idx]
        v = v[idx]
        mag = mag[idx]

    return x, y, u, v, mag


def load_and_filter_dic(
    dic: DIC,
    *,
    filter_outliers: bool = True,
    tails_percentile: float = 0.01,
    min_velocity: float | None = None,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read DIC HDF5 for `dic` and apply filtering + subsampling.

    Returns: x, y, u, v, mag (all numpy arrays float).
    Raises Http404 on errors or missing required datasets.
    """
    data = read_dic_h5(dic)

    return filter_dic_arrays(
        data["points"],
        data["vectors"],
        data["magnitudes"],
        filter_outliers=filter_outliers,
        tails_percentile=tails_percentile,
        min_velocity=min_velocity,
        subsample=subsample,
    )

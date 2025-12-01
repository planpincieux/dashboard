"""
Unified DIC uploader:
- Supports per-timestamp DIC text files (day_dic_YYYYMMDD-YYYYMMDD.txt)
- Supports inversion time-series (EW / NS stacks)
Builds a flexible JSON payload for the API irrespective of source.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

# Configuration
DIC_FILES_PATTERN = "day_dic_*.txt"
DJANGO_API_URL = "http://150.145.51.193:8080/API/dic/upload/"
SOFTWARE_USED = "pylamma"


# -------------------------------
# CLI / Interactive
# -------------------------------
def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload DIC data (per-timestamp text files or inversion stacks) to API."
    )
    parser.add_argument(
        "--mode",
        choices=["dic", "inversion"],
        help="Operation mode: dic (day_dic_*.txt) or inversion (EW/NS stacks).",
    )
    parser.add_argument(
        "--dic-dir",
        type=Path,
        help="Directory containing day_dic_*.txt files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DIC_FILES_PATTERN,
        help="Glob pattern for DIC text files.",
    )
    parser.add_argument("--ew", type=Path, help="EW inversion stack file path.")
    parser.add_argument("--ns", type=Path, help="NS inversion stack file path.")
    parser.add_argument(
        "--invert-ns",
        action="store_true",
        help="Invert NS component sign (default False).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DJANGO_API_URL,
        help="Override API upload URL.",
    )
    return parser.parse_args(argv)


# -------------------------------
# Data readers
# -------------------------------
def parse_dic_filename(dic_filename: Path) -> tuple[datetime, datetime] | None:
    """Parse DIC filename: day_dic_YYYYMMDD-YYYYMMDD.txt -> (slave_date, master_date)."""
    try:
        if not dic_filename.name.startswith("day_dic_"):
            return None
        date_part = dic_filename.stem.split("_")[2]
        slave_str, master_str = date_part.split("-")
        return (
            datetime.strptime(slave_str, "%Y%m%d"),
            datetime.strptime(master_str, "%Y%m%d"),
        )
    except Exception:
        return None


def read_dic_text_file(dic_file: Path, invert_y: bool = False) -> dict:
    """
    Read per-timestamp DIC text file with columns: x,y,dx,dy,magnitude.
    Returns dict containing points, vectors, magnitudes, max_magnitude.
    """
    data = np.loadtxt(dic_file, delimiter=",")
    if data.size == 0:
        raise ValueError("Empty DIC file")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    dx = data[:, 2].astype(np.float32)
    dy = data[:, 3].astype(np.float32)
    mag = data[:, 4].astype(np.float32)

    if invert_y:
        y = -y
        dy = -dy  # maintain consistency if inversion needed

    dic_data = {
        "points": np.column_stack((x, y)).tolist(),
        "vectors": np.column_stack((dx, dy)).tolist(),
        "magnitudes": mag.tolist(),
        "max_magnitude": float(np.max(mag)),
    }
    return dic_data


def read_inversion_file(
    file_path: Path, invert_y: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read inversion stack:
    Header: # X Y <date tokens...>
    Data: X Y d1 d2 ... (displacement components per date)
    Returns (dates[datetime64[D]], x, y, vectors[shape=(n_points, n_dates)]).
    """
    with open(file_path, "r") as f:
        header = f.readline().strip()

    header_parts = header.split()
    date_strings = header_parts[3:]  # Skip '#', 'X', 'Y'

    dates_py: list[datetime] = []
    for date_str in date_strings:
        try:
            dates_py.append(datetime.strptime(date_str[:8], "%Y%m%d"))
        except ValueError:
            # Skip malformed tokens
            continue
    dates = np.array(dates_py, dtype="datetime64[D]")

    data = np.loadtxt(file_path, skiprows=1)
    x = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    vectors = data[:, 2:].astype(np.float32)  # shape: (n_points, n_dates)

    if invert_y:
        y = -y
        vectors[:, 1] = -vectors[:, 1]  # maintain consistency if inversion needed

    return dates, x, y, vectors


# -------------------------------
# Builder and helpers to upload DIC data
# -------------------------------
def compute_magnitude_from_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    vectors shape: (n_points, 2) -> returns magnitude shape (n_points,)
    """
    return np.sqrt(np.sum(vectors**2, axis=1))


def build_dic_payload(dic_input: dict) -> dict:
    """
    Unified builder.
    dic_input may contain:
      master_date (YYYY-MM-DD) | master_image_id
      slave_date  (YYYY-MM-DD) | slave_image_id
      reference_date (optional)
      with_inversion (bool, optional)
      software_used (optional)
      dic_data: {
          points: [[x,y], ...] (required)
          magnitudes: [..] (required OR vectors to derive)
          vectors: [[dx,dy], ...] (optional -> used to derive magnitudes if missing)
          max_magnitude: float (optional -> computed if missing)
          any extra custom fields
      }
    Auto-computes missing magnitudes / max_magnitude.
    Returns payload ready for POST.
    """
    if "dic_data" not in dic_input:
        raise ValueError("dic_input must include 'dic_data'")

    dic_data = dic_input["dic_data"]
    points = dic_data.get("points")
    magnitudes = dic_data.get("magnitudes")
    vectors = dic_data.get("vectors")

    if points is None:
        raise ValueError("dic_data missing 'points'")

    # Derive magnitudes if absent but vectors provided
    if magnitudes is None:
        if vectors is None:
            raise ValueError("dic_data requires 'magnitudes' or 'vectors'")
        vec_arr = np.asarray(vectors, dtype=np.float32)
        magnitudes = compute_magnitude_from_vectors(vec_arr).astype(np.float32).tolist()
        dic_data["magnitudes"] = magnitudes

    # Compute max_magnitude if absent
    if "max_magnitude" not in dic_data:
        dic_data["max_magnitude"] = float(
            np.max(np.asarray(magnitudes, dtype=np.float32))
        )

    # Defaults
    payload = {
        "master_date": dic_input.get("master_date"),
        "slave_date": dic_input.get("slave_date"),
        "reference_date": dic_input.get("reference_date")
        or dic_input.get("slave_date"),
        "dic_data": dic_data,
        "with_inversion": bool(dic_input.get("with_inversion", False)),
        "software_used": dic_input.get("software_used", SOFTWARE_USED),
    }

    # Include any additional arbitrary fields (e.g., notes, label, processing_parameters)
    for k, v in dic_input.items():
        if k not in payload and k != "dic_data":
            payload[k] = v

    return payload


def upload_dic_payload(
    session: requests.Session, payload: dict, api_url: str = DJANGO_API_URL
) -> dict:
    """POST a prepared payload. Returns status dict."""
    result = {
        "reference_date": payload.get("reference_date"),
        "status": "failed",
        "message": "",
    }
    try:
        response = session.post(
            api_url, json=payload, headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result["status"] = "success"
            result["message"] = response.text
        else:
            result["message"] = f"{response.status_code} - {response.text}"
    except Exception as e:
        result["message"] = str(e)
    return result


# -------------------------------
# High-level processes
# -------------------------------
def process_dic_text_files(dic_dir: Path, pattern: str = DIC_FILES_PATTERN) -> None:
    """Process and upload per-timestamp DIC text files."""
    files = sorted(dic_dir.glob(pattern))
    if not files:
        print(f"No DIC files found in {dic_dir}")
        return

    session = requests.Session()
    stats = {"success": 0, "failed": 0}

    for f in tqdm(files, desc="Uploading DIC text files"):
        dates = parse_dic_filename(f)
        if not dates:
            print(f"Skip invalid filename: {f.name}")
            stats["failed"] += 1
            continue
        slave_date, master_date = dates

        try:
            dic_data = read_dic_text_file(f)
        except Exception as e:
            print(f"Read error {f.name}: {e}")
            stats["failed"] += 1
            continue

        payload = build_dic_payload(
            {
                "master_date": master_date.strftime("%Y-%m-%d"),
                "slave_date": slave_date.strftime("%Y-%m-%d"),
                "dic_data": dic_data,
                "with_inversion": False,
            }
        )
        res = upload_dic_payload(session, payload)
        stats[res["status"]] += 1
        if res["status"] == "failed":
            print(f"Failed {f.name}: {res['message']}")

    print(f"\nText files upload complete: {stats}")


def process_inversion_stack(
    ew_file: Path,
    ns_file: Path,
    invert_ns: bool = False,
) -> None:
    """
    Process inversion time-series EW / NS stacks and upload each timestamp as a DIC entry.
    Assumes EW file and NS file share identical date header and node layout.
    """
    dates, x, y, ew_vectors = read_inversion_file(ew_file)
    dates_ns, x_ns, y_ns, ns_vectors = read_inversion_file(ns_file)

    if (
        not np.array_equal(dates, dates_ns)
        or not np.array_equal(x, x_ns)
        or not np.array_equal(y, y_ns)
    ):
        raise ValueError("EW / NS files mismatch in dates or node coordinates")

    if invert_ns:  # Invert NS displacements if needed
        y = -y
        ns_vectors = -ns_vectors

    session = requests.Session()
    stats = {"success": 0, "failed": 0}

    # Iterate timestamps (columns)
    for col_idx in tqdm(range(ew_vectors.shape[1]), desc="Uploading inversion stack"):
        dx = ew_vectors[:, col_idx]
        dy = ns_vectors[:, col_idx]
        reference_date = dates[col_idx].astype("M8[D]").astype(str)

        dic_data = {
            "points": np.column_stack((x, y)).tolist(),
            "vectors": np.column_stack((dx, dy)).tolist(),
            # magnitudes omitted -> auto-computed by builder
        }

        payload = build_dic_payload(
            {
                "master_date": reference_date,
                "slave_date": reference_date,
                "reference_date": reference_date,
                "dic_data": dic_data,
                "with_inversion": True,
                "software_used": SOFTWARE_USED,
                "label": "inversion_stack",
            }
        )
        res = upload_dic_payload(session, payload)
        stats[res["status"]] += 1
        if res["status"] == "failed":
            print(f"Failed {reference_date}: {res['message']}")

    print(f"\nInversion stack upload complete: {stats}")


def interactive_run() -> None:
    DIC_RESULTS_DIR = Path.cwd() / "dummy"
    MODE = "inversion"
    INVERT_Y = False

    if MODE == "dic":
        dir_in = DIC_RESULTS_DIR
        pattern = DIC_FILES_PATTERN
        process_dic_text_files(dir_in, pattern)

    elif MODE == "inversion":
        invert_ns = INVERT_Y
        ew = input("EW file path: ").strip()
        ns = input("NS file path: ").strip()
        if not ew or not ns:
            print("Both EW and NS file paths required.")
            return

        process_inversion_stack(Path(ew), Path(ns), invert_ns=invert_ns)
    else:
        print("Invalid mode.")


def main(argv: list[str] | None = None) -> None:
    # Allow override of API URL dynamically
    args = parse_cli_args(argv)
    global DJANGO_API_URL
    DJANGO_API_URL = args.api_url

    if not args.mode:
        interactive_run()
        return
    if args.mode == "dic":
        process_dic_text_files(args.dic_dir, args.pattern)

    elif args.mode == "inversion":
        if not args.ew or not args.ns:
            print("Provide both --ew and --ns for inversion mode.")
            return
        process_inversion_stack(
            args.ew,
            args.ns,
            invert_ns=args.invert_ns,
        )

    else:
        print("Unknown mode. Use --mode dic or inversion.")


if __name__ == "__main__":
    main(sys.argv[1:])

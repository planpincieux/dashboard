"""
Load new DIC results to the database.

This script reads daily DIC files and stores them in the database with their associated metadata.
Expected filename format: day_dic_YYYYMMDD-YYYYMMDD.txt
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

# Configuration
DIC_RESULTS_DIR = Path("./temp")
DIC_FILES_PATTERN = "day_dic_*.txt"

# DJANGO_API_URL = "http://150.145.51.193:8080/API/dic/upload/"
DJANGO_API_URL = "http://127.0.0.1:9999/API/dic/upload/"  # Local testing
SOFTWARE_USED = "pylamma"


def parse_dic_filename(dic_filename: Path) -> tuple[datetime, datetime] | None:
    """Parse DIC filename: day_dic_YYYYMMDD-YYYYMMDD.txt -> (slave_date, master_date)"""
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


def read_dic_file(dic_file: Path, invert_y: bool = False) -> dict:
    """Read DIC results: x, y, dx, dy, magnitude"""
    try:
        data = np.loadtxt(dic_file, delimiter=",")
        if data.size == 0:
            raise ValueError("Empty data")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        x = data[:, 0]
        y = data[:, 1]
        if invert_y:
            y = -y
        dx = data[:, 2]
        dy = data[:, 3]
        if invert_y:
            dy = -dy

        # Return data as JSON-serializable dict
        dic_data = {
            "points": np.column_stack(
                (x.astype(np.int32), -y.astype(np.int32))
            ).tolist(),
            "vectors": np.column_stack(
                (dx.astype(np.float32), -dy.astype(np.float32))
            ).tolist(),
            "magnitudes": data[:, 4].astype(np.float32).tolist(),
            "max_magnitude": float(np.max(data[:, 4])),
        }
        return dic_data
    except Exception:
        raise ValueError(f"Failed to read DIC file: {dic_file}")


def upload_dic_file(dic_file: Path, session: requests.Session) -> dict:
    """Upload single DIC file via API"""
    result = {"filename": dic_file.name, "status": "failed", "message": ""}

    # Parse filename
    dates = parse_dic_filename(dic_file)
    if not dates:
        result["message"] = "Invalid filename"
        return result

    slave_date, master_date = dates

    # Read DIC data
    try:
        dic_data = read_dic_file(dic_file)
    except Exception as e:
        result["message"] = f"Read error: {e}"
        return result

    try:
        # Send as JSON
        response = session.post(
            DJANGO_API_URL,
            json={
                "master_date": master_date.strftime("%Y-%m-%d"),
                "slave_date": slave_date.strftime("%Y-%m-%d"),
                "software_used": SOFTWARE_USED,
                "with_inversion": True,
                "dic_data": dic_data,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            resp_data = response.json()
            result["status"] = "success"
            result["message"] = f"DIC ID: {resp_data['dic_id']}"
        else:
            result["message"] = f"API error: {response.status_code} - {response.text}"

    except Exception as e:
        result["message"] = str(e)

    return result


def upload_dic_files(dic_dir: Path, pattern: str = "*.txt") -> None:
    """Load all DIC files via API"""
    dic_files = sorted(dic_dir.glob(pattern))

    if not dic_files:
        print(f"No files found in {dic_dir}")
        return

    print(f"Found {len(dic_files)} files")

    session = requests.Session()
    results = {"success": 0, "failed": 0}

    for dic_file in tqdm(dic_files, desc="Uploading"):
        result = upload_dic_file(dic_file, session)

        if result["status"] == "success":
            results["success"] += 1
        else:
            results["failed"] += 1
            print(f"\nFailed: {result['filename']} - {result['message']}")

    print(f"\n{'=' * 60}")
    print(f"Success: {results['success']}, Failed: {results['failed']}")


if __name__ == "__main__":
    upload_dic_files(
        dic_dir=DIC_RESULTS_DIR,
        pattern=DIC_FILES_PATTERN,
    )

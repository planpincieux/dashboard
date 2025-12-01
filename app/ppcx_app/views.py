import io
import json
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from django.db import transaction
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_http_methods
from PIL import Image
from PIL import Image as PILImage

from ppcx_app.functions.h5dic import create_dic_h5, filter_dic_arrays, read_dic_h5
from ppcx_app.functions.visualization import (
    draw_quiver_on_image_cv2,
    plot_dic_scatter,
    plot_dic_vectors,
)
from ppcx_app.models import DIC, Collapse, Image

matplotlib.use("Agg")  # Use non-interactive backend

logger = logging.getLogger(__name__)

H5_FILES_DIR = Path("/ppcx/data/dic_h5")

# optional presence of cv2 for encoding / fallbacks
try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    HAS_CV2 = False


def _parse_float(s: str | None, default: float | None = None) -> float | None:
    try:
        return float(s) if s is not None and s != "" else default
    except (ValueError, TypeError):
        return default


def _parse_int(s: str | None, default: int | None = None) -> int | None:
    try:
        return int(s) if s is not None and s != "" else default
    except (ValueError, TypeError):
        return default


def _parse_bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).lower() in ("1", "true", "yes", "y")


@require_http_methods(["GET"])
def home(request) -> HttpResponse:
    return HttpResponse("Welcome to the Planpincieux API. Use /API/ for endpoints.")


# ===================== IMAGES =======================
@require_http_methods(["GET"])
def serve_image(request, image_id: int) -> HttpResponse:
    """Serve image files by Image.id (inline)."""
    image = get_object_or_404(Image, id=image_id)
    if not os.path.exists(image.file_path):
        raise Http404("Image file not found")
    try:
        with open(image.file_path, "rb") as f:
            content = f.read()
        content_type, _ = mimetypes.guess_type(image.file_path)
        content_type = content_type or "application/octet-stream"
        response = HttpResponse(content, content_type=content_type)
        response["Content-Disposition"] = (
            f'inline; filename="{os.path.basename(image.file_path)}"'
        )
        return response
    except OSError:
        raise Http404("Could not read image file")


# ===================== DIC =======================
@require_http_methods(["POST"])
@csrf_exempt
def upload_dic_h5(request, h5_dir: str | None = None) -> JsonResponse:
    """
    Upload DIC data and create database entry with H5 file.

    POST JSON body (required):
      - master_date: YYYY-MM-DD (master image date) OR master_image_id: int
      - slave_date: YYYY-MM-DD (slave image date) OR slave_image_id: int
      - dic_data: dict with keys: points, magnitudes (required); vectors, max_magnitude (optional)

    Optional fields (match DIC model):
      - reference_date: YYYY-MM-DD (defaults to slave_date)
      - software_used: str (default: "pylamma")
      - software_version: str
      - with_inversion: bool (default: false)
      - use_ensemble_correlation: bool (default: false)
      - ensemble_size: int
      - ensemble_image_pairs: list of {"master": id, "slave": id}
      - processing_parameters: dict (stored as JSON)
      - label: str
      - notes: str
      - dt_hours: int (auto-calculated if not provided)
    """
    try:
        h5_directory = Path(h5_dir) if h5_dir else H5_FILES_DIR
        data = json.loads(request.body)
        dic_data = data.get("dic_data")
        if not dic_data:
            return JsonResponse(
                {"error": "Missing required field: dic_data"}, status=400
            )

        # Helper function to get image by date or ID
        def get_image(
            id_key: str, date_key: str, take_first: bool = True
        ) -> Image | None:
            if id_key in data:
                return Image.objects.filter(id=data[id_key]).first()
            if date_key in data:
                try:
                    date = datetime.strptime(data[date_key], "%Y-%m-%d").date()
                    images = Image.objects.filter(
                        acquisition_timestamp__date=date
                    ).all()
                    if take_first:
                        return images.first()
                    else:
                        mid_id = len(images) // 2
                        return images[mid_id]
                except ValueError:
                    return None
            return None

        # Get master and slave images
        master_image = get_image("master_image_id", "master_date", take_first=False)
        slave_image = get_image("slave_image_id", "slave_date")
        if not master_image or not slave_image:
            return JsonResponse(
                {"error": "Master or slave image not found"}, status=404
            )

        # Use transaction to ensure atomicity
        with transaction.atomic():
            # Prepare DIC fields with defaults
            dic_fields = {
                "master_image": master_image,
                "slave_image": slave_image,
                "result_file_path": "",  # Temporary, updated after H5 creation
            }

            # Reference date (default to slave image date)
            if "reference_date" in data:
                try:
                    dic_fields["reference_date"] = datetime.strptime(
                        data["reference_date"], "%Y-%m-%d"
                    ).date()
                except ValueError:
                    dic_fields["reference_date"] = (
                        slave_image.acquisition_timestamp.date()
                    )
            else:
                dic_fields["reference_date"] = slave_image.acquisition_timestamp.date()

            # Map JSON fields to DIC model fields
            optional_fields = {
                "software_used": (str, "pylamma"),
                "software_version": (str, None),
                "with_inversion": (bool, False),
                "use_ensemble_correlation": (bool, False),
                "ensemble_size": (int, None),
                "ensemble_image_pairs": (list, None),
                "processing_parameters": (dict, None),
                "label": (str, None),
                "notes": (str, None),
                "dt_hours": (int, None),
            }
            for field_name, (field_type, default_value) in optional_fields.items():
                if field_name in data:
                    value = data[field_name]
                    if value is not None and not isinstance(value, field_type):
                        try:
                            value = field_type(value)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid type for {field_name}, using default: {default_value}"
                            )
                            value = default_value
                    dic_fields[field_name] = value
                elif default_value is not None:
                    dic_fields[field_name] = default_value

            # Create DIC entry to get ID for filename
            dic = DIC.objects.create(**dic_fields)

            # Generate H5 filename with DIC ID
            reference_date = dic_fields["reference_date"]
            master_stem = Path(master_image.file_path).stem
            slave_stem = Path(slave_image.file_path).stem
            h5_filename = (
                f"{dic.pk:08d}_{reference_date.strftime('%Y%m%d')}_"
                f"{master_stem}_{slave_stem}.h5"
            )
            h5_path = h5_directory / h5_filename

            # Create the H5 file
            if not create_dic_h5(dic_data, h5_path):
                raise Exception("Failed to create H5 file")

            # Update DIC entry with h5 path
            dic.result_file_path = str(h5_path)
            dic.save(update_fields=["result_file_path"])

        return JsonResponse(
            {
                "success": True,
                "dic_id": dic.id,
                "h5_path": str(h5_path),
                "reference_date": dic.reference_date.isoformat(),
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Error uploading DIC: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def serve_dic_h5(request, dic_id: int) -> HttpResponse:
    """
    Return DIC HDF5 content as JSON for the specified DIC id.
    Query params: none required.
    """
    dic = get_object_or_404(DIC, id=dic_id)

    try:
        data = read_dic_h5(dic)
        # convert numpy arrays to lists for JSON serialization
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        return JsonResponse(data)
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e


@require_http_methods(["GET"])
def serve_dic_h5_as_csv(request, dic_id: int) -> HttpResponse:
    """Return DIC data as CSV (x,y,u,v,magnitude)."""
    dic = get_object_or_404(DIC, id=dic_id)
    try:
        data = read_dic_h5(dic)
        points = data["points"]  # Nx2 array
        vectors = data["vectors"]  # Nx2 array
        magnitudes = data["magnitudes"]  # N array

        csv_lines = ["x,y,u,v,magnitude"]
        for i in range(len(points)):
            x, y = points[i]
            u, v = vectors[i] if vectors is not None else (0.0, 0.0)
            magnitude = magnitudes[i]
            csv_lines.append(f"{x},{y},{u},{v},{magnitude}")

        # Build informative filename
        master_ts = (
            dic.master_timestamp.strftime("%Y-%m-%d-%H-%M")
            if dic.master_timestamp
            else "unknown"
        )
        slave_ts = (
            dic.slave_timestamp.strftime("%Y-%m-%d-%H-%M")
            if dic.slave_timestamp
            else "unknown"
        )
        dt_days = getattr(dic, "dt_days", None)
        if dt_days is None and dic.dt_hours is not None:
            dt_days = int(round(dic.dt_hours / 24.0))
        filename = f"dic_{dic_id}_{master_ts}_{slave_ts}_{dt_days}days.csv"

        response = HttpResponse("\n".join(csv_lines), content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e


@require_http_methods(["GET"])
def visualize_dic(request, dic_id: int) -> HttpResponse:
    """
    Generate PNG visualization (matplotlib) for a DIC record.

    Query parameters (all parsed at start):
      - plot_type: 'quiver'|'scatter' (default 'quiver')
      - background: true/false (default true)
      - cmap: colormap (default 'viridis')
      - vmin / vmax: min/max magnitude for colormap (float)
      - min_velocity: minimum magnitude to keep (float)
      - filter_outliers: true/false (default true)
      - tails_percentile: percentile for trimming (default 0.01)
      - subsample: int (default 1)
      - scale / scale_units / width / headwidth / quiver_alpha / image_alpha
      - figsize: "W,H" and dpi
    """
    # Plotting type
    plot_type = request.GET.get("plot_type", "quiver").lower()
    if plot_type not in ("quiver", "scatter"):
        return HttpResponse("Invalid plot_type. Use 'quiver' or 'scatter'.", status=400)

    # Normalize by time difference if requested
    plot_velocity = _parse_bool(request.GET.get("plot_velocity"), True)

    # Plotting params
    show_background = _parse_bool(request.GET.get("background"), True)
    cmap_name = request.GET.get("cmap", "viridis")
    vmin = _parse_float(request.GET.get("vmin"), None)
    vmax = _parse_float(request.GET.get("vmax"), None)
    figsize_param = request.GET.get("figsize", "")
    if figsize_param:
        try:
            w, h = [float(x) for x in figsize_param.split(",", 1)]
            figsize = (w, h)
        except Exception:
            figsize = (10.0, 8.0)
    else:
        figsize = (10.0, 8.0)
    dpi = _parse_int(request.GET.get("dpi"), 150) or 150

    # quiver params
    scale_raw = request.GET.get("scale", None)
    scale = (
        None
        if scale_raw is None or scale_raw.lower() == "none"
        else _parse_float(scale_raw, None)
    )
    scale_units = request.GET.get("scale_units", "xy")
    width = _parse_float(request.GET.get("width"), 0.005) or 0.005
    headwidth = _parse_float(request.GET.get("headwidth"), 2.5) or 2.5
    quiver_alpha = _parse_float(request.GET.get("quiver_alpha"), 1.0) or 1.0
    image_alpha = _parse_float(request.GET.get("image_alpha"), 0.7) or 0.7

    # filtering params
    filter_outliers = _parse_bool(request.GET.get("filter_outliers"), False)
    tails_percentile = _parse_float(request.GET.get("tails_percentile"), 0.01) or 0.01
    filter_min_velocity = _parse_float(request.GET.get("filter_min_velocity"), None)
    subsample = _parse_int(request.GET.get("subsample"), 1) or 1

    # load DIC record
    dic = get_object_or_404(DIC, id=dic_id)

    # load and filter data
    data = read_dic_h5(dic)
    points = data["points"]
    vectors = data["vectors"]
    magnitudes = data["magnitudes"]
    x, y, u, v, mag = filter_dic_arrays(
        points,
        vectors,
        magnitudes,
        min_velocity=filter_min_velocity,
        filter_outliers=filter_outliers,
        tails_percentile=tails_percentile,
        subsample=subsample,
    )

    # Normalize displacement by the time difference and plot daily velocity if requested
    dt_hours = dic.dt_hours if dic.dt_hours is not None and dic.dt_hours > 0 else 1.0
    if plot_velocity:
        u = u / dt_hours * 24.0  # px -> px/day
        v = v / dt_hours * 24.0  # px -> px/day
        mag = mag / dt_hours * 24.0  # px -> px/day
        clabel = "Velocity Magnitude [px/day]"
    else:
        clabel = "Displacement Magnitude [px]"

    # background image (may be rotated for Tele cameras) - only rotate image, not vectors
    background_image: np.ndarray | None = None
    if show_background:
        try:
            image_path = dic.master_image.file_path if dic.master_image else None
            if image_path and os.path.exists(image_path):
                pil_image = PILImage.open(image_path)
                # Use the rotation field from the model
                if dic.master_image.rotation and dic.master_image.rotation != 0:
                    pil_image = pil_image.rotate(
                        -dic.master_image.rotation, expand=True
                    )
                background_image = np.array(pil_image)
        except Exception:
            background_image = None

    # plotting
    plt.figure(figsize=figsize, dpi=dpi)
    if plot_type == "quiver":
        fig, ax, _ = plot_dic_vectors(
            x,
            y,
            u,
            v,
            mag,
            background_image=background_image,
            vmin=vmin,
            vmax=vmax,
            scale=scale,
            scale_units=scale_units,
            width=width,
            headwidth=headwidth,
            quiver_alpha=quiver_alpha,
            image_alpha=image_alpha,
            cmap_name=cmap_name,
            clabel=clabel,
            dpi=dpi,
        )
    else:
        fig, ax, _ = plot_dic_scatter(
            x,
            y,
            mag,
            background_image=background_image,
            vmin=vmin if vmin is not None else 0.0,
            vmax=vmax,
            cmap_name=cmap_name,
            dpi=dpi,
        )

    # title
    if dic.master_timestamp and dic.slave_timestamp:
        master_date = dic.master_timestamp.strftime("%Y-%m-%d")
        slave_date = dic.slave_timestamp.strftime("%Y-%m-%d")
        title = f"DIC #{dic_id}: {master_date} → {slave_date}"
        if dic.dt_hours:
            title += f" ({dic.dt_hours} hours)"
        ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="jpg",
        bbox_inches="tight",
    )
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/jpeg")


@require_http_methods(["GET"])
def serve_dic_quiver(request, dic_id: int) -> HttpResponse:
    """
    Generate and return a quiver PNG drawn with OpenCV (or PIL fallback).

    Query parameters parsed at start:
      - colormap (str)
      - arrow_length_scale (float) or omitted for auto
      - arrow_thickness (int) or omitted for auto
      - subsample (int)
      - rotate_tele (bool) default True (rotate background image only)
      - filter_outliers (bool), tails_percentile (float), min_velocity (float)
      - vmin / vmax (float) clip for color mapping
    """
    # parse params
    colormap = request.GET.get("colormap", "viridis")
    arrow_length_scale = _parse_float(request.GET.get("arrow_length_scale"), None)
    arrow_thickness = _parse_int(request.GET.get("arrow_thickness"), None)
    subsample = _parse_int(request.GET.get("subsample"), 1) or 1
    rotate_tele = _parse_bool(request.GET.get("rotate_tele"), True)

    vmin = _parse_float(request.GET.get("vmin"), None)
    vmax = _parse_float(request.GET.get("vmax"), None)
    min_velocity = _parse_float(request.GET.get("min_velocity"), None)
    filter_outliers = _parse_bool(request.GET.get("filter_outliers"), True)
    tails_percentile = _parse_float(request.GET.get("tails_percentile"), 0.01) or 0.01

    dic = get_object_or_404(DIC, id=dic_id)
    x, y, u, v, mag = load_and_filter_dic(
        dic,
        filter_outliers=filter_outliers,
        tails_percentile=tails_percentile,
        min_velocity=min_velocity
        if (min_velocity is not None and min_velocity > 0)
        else None,
        subsample=subsample,
    )

    # apply clip for color mapping if requested (does not change geometry)
    if (vmin is not None) or (vmax is not None):
        clip_lo = vmin if vmin is not None else (mag.min() if mag.size else 0.0)
        clip_hi = vmax if vmax is not None else (mag.max() if mag.size else clip_lo)
        if clip_hi < clip_lo:
            clip_lo, clip_hi = clip_hi, clip_lo
        mag = np.clip(mag, clip_lo, clip_hi)

    # background image (rotated for Tele cameras only)
    background_image = None
    try:
        image_path = dic.master_image.file_path if dic.master_image else None
        if image_path and os.path.exists(image_path):
            pil_image = PILImage.open(image_path)
            # Use rotation field instead of camera name check
            if (
                rotate_tele
                and dic.master_image.rotation
                and dic.master_image.rotation != 0
            ):
                pil_image = pil_image.rotate(-dic.master_image.rotation, expand=True)
            background_image = np.array(pil_image)
            # ensure BGR uint8 for OpenCV helper
            if background_image.ndim == 3 and background_image.shape[2] == 3:
                background_image = background_image[:, :, ::-1].copy()
            elif background_image.ndim == 2:
                background_image = np.stack([background_image] * 3, axis=2)
    except Exception:
        background_image = None

    # if no background create white canvas based on extents
    if background_image is None:
        max_x = int(np.ceil(np.max(x))) if x.size else 200
        max_y = int(np.ceil(np.max(y))) if y.size else 200
        canvas_w = max(200, max_x + 10)
        canvas_h = max(200, max_y + 10)
        background_image = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # draw using helper
    try:
        out_bgr = draw_quiver_on_image_cv2(
            background_image,
            x,
            y,
            u,
            v,
            mag,
            colormap_name=colormap,
            arrow_length_scale=arrow_length_scale,
            arrow_thickness=arrow_thickness,
            alpha=1.0,
        )
    except Exception as e:
        raise Http404(f"Error drawing quiver: {e}") from e

    # encode PNG (prefer cv2 if available)
    try:
        if HAS_CV2 and cv2 is not None:
            ok, buf = cv2.imencode(".png", out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            png_bytes = buf.tobytes()
        else:
            out_rgb = out_bgr[:, :, ::-1]
            pil_out = PILImage.fromarray(out_rgb)
            bio = io.BytesIO()
            pil_out.save(bio, format="PNG", optimize=True, compress_level=3)
            png_bytes = bio.getvalue()
    except Exception as e:
        raise Http404(f"Failed to encode PNG: {e}") from e

    # Build informative filename
    master_ts = (
        dic.master_timestamp.strftime("%Y-%m-%d-%H-%M")
        if dic.master_timestamp
        else "unknown"
    )
    slave_ts = (
        dic.slave_timestamp.strftime("%Y-%m-%d-%H-%M")
        if dic.slave_timestamp
        else "unknown"
    )
    dt_days = getattr(dic, "dt_days", None)
    if dt_days is None and dic.dt_hours is not None:
        dt_days = int(round(dic.dt_hours / 24.0))
    filename = f"quiver_{dic_id}_{master_ts}_{slave_ts}_{dt_days}days.png"

    response = HttpResponse(png_bytes, content_type="image/png")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@require_http_methods(["GET"])
def dic_visualizer(request, dic_id: int | None = None) -> HttpResponse:
    """
    Small DIC visualizer UI.

    - Supports simple filtering of DIC results (date range, camera, month, label).
    - Defaults to the most recent matching DIC when none selected.
    - Exposes Prev/Next links (keyboard arrows also navigate).
    - Forwards visualization parameters to visualize/quiver endpoints.

    Query params considered for filtering:
      reference_date, master_timestamp_start, master_timestamp_end,
      camera_id, camera_name, dt_hours_min, dt_hours_max,
      dt_min_days, dt_max_days, dt_min_hours, dt_max_hours,
      month, label

    Visualization params forwarded:
      plot_type, background, cmap, vmin, vmax, scale, scale_units,
      width, headwidth, quiver_alpha, image_alpha, figsize, dpi,
      filter_outliers, tails_percentile, min_velocity, subsample,
      arrow_length_scale, arrow_thickness, rotate_tele
    """
    # --- parse simple filters (kept early) ---
    q_reference_date = request.GET.get("reference_date")
    q_master_start = request.GET.get("master_timestamp_start")
    q_master_end = request.GET.get("master_timestamp_end")
    q_camera_id = request.GET.get("camera_id")
    q_camera_name = request.GET.get("camera_name")
    q_time_diff_min = request.GET.get("dt_hours_min")
    q_time_diff_max = request.GET.get("dt_hours_max")
    q_dt_min_days = request.GET.get("dt_min_days")
    q_dt_max_days = request.GET.get("dt_max_days")
    q_dt_min_hours = request.GET.get("dt_min_hours")
    q_dt_max_hours = request.GET.get("dt_max_hours")
    q_month = request.GET.get("month")
    q_label = request.GET.get("label")
    q_year = request.GET.get("year")  # new year filter

    qs = DIC.objects.select_related("master_image__camera", "slave_image").all()

    if q_reference_date:
        try:
            dt = datetime.fromisoformat(q_reference_date)
            qs = qs.filter(reference_date=dt.date())
        except Exception:
            pass

    if q_master_start:
        try:
            dt = datetime.fromisoformat(q_master_start)
            qs = qs.filter(master_timestamp__gte=dt)
        except Exception:
            pass

    if q_master_end:
        try:
            dt = datetime.fromisoformat(q_master_end)
            qs = qs.filter(master_timestamp__lte=dt)
        except Exception:
            pass

    if q_camera_id:
        try:
            qs = qs.filter(master_image__camera__id=int(q_camera_id))
        except Exception:
            pass

    if q_camera_name:
        qs = qs.filter(master_image__camera__camera_name__icontains=q_camera_name)

    if q_time_diff_min:
        try:
            qs = qs.filter(dt_hours__gte=float(q_time_diff_min))
        except Exception:
            pass

    if q_time_diff_max:
        try:
            qs = qs.filter(dt_hours__lte=float(q_time_diff_max))
        except Exception:
            pass

    # dt days -> hours conversion
    try:
        if q_dt_min_days:
            qs = qs.filter(dt_hours__gte=float(q_dt_min_days) * 24.0)
    except Exception:
        pass
    try:
        if q_dt_max_days:
            qs = qs.filter(dt_hours__lte=float(q_dt_max_days) * 24.0)
    except Exception:
        pass
    try:
        if q_dt_min_hours:
            qs = qs.filter(dt_hours__gte=float(q_dt_min_hours))
    except Exception:
        pass
    try:
        if q_dt_max_hours:
            qs = qs.filter(dt_hours__lte=float(q_dt_max_hours))
    except Exception:
        pass

    if q_month:
        try:
            m = int(q_month)
            if 1 <= m <= 12:
                qs = qs.filter(reference_date__month=m)
        except Exception:
            pass

    if q_label:
        qs = qs.filter(label__icontains=q_label)

    if q_year:
        try:
            y = int(q_year)
            qs = qs.filter(reference_date__year=y)
        except Exception:
            pass

    dic_list = list(qs.order_by("-master_timestamp")[:200])  # list for indexing

    # --- visualization params forwarded to endpoints ---
    viz_keys = {
        "plot_type",
        "background",
        "cmap",
        "vmin",
        "vmax",
        "scale",
        "scale_units",
        "width",
        "headwidth",
        "quiver_alpha",
        "image_alpha",
        "figsize",
        "dpi",
        "filter_outliers",
        "tails_percentile",
        "min_velocity",
        "subsample",
        "arrow_length_scale",
        "arrow_thickness",
        "rotate_tele",
    }

    selected_dic = request.GET.get("dic_id") or (
        str(dic_id) if dic_id is not None else None
    )
    if not selected_dic and dic_list:
        selected_dic = str(dic_list[0].id)  # default most recent

    # build forward params (visualization only)
    forward_params = {}
    for k, v in request.GET.items():
        if k in viz_keys and v is not None and v != "":
            forward_params[k] = v

    visualize_url = None
    quiver_url = None
    prev_url = None
    next_url = None
    selected_index = None
    selected_obj = None
    master_thumb = None
    slave_thumb = None

    if selected_dic:
        try:
            selected_id_int = int(selected_dic)
        except Exception:
            selected_id_int = None

        if selected_id_int is not None:
            for i, d in enumerate(dic_list):
                if d.id == selected_id_int:
                    selected_index = i
                    break

        if selected_index is None and dic_list:
            selected_index = 0
            selected_id_int = dic_list[0].id

        if selected_index is not None:
            selected_obj = dic_list[selected_index]
            try:
                if selected_obj.master_image:
                    master_thumb = reverse(
                        "serve_image", kwargs={"image_id": selected_obj.master_image.id}
                    )
            except Exception:
                master_thumb = None
            try:
                if selected_obj.slave_image:
                    slave_thumb = reverse(
                        "serve_image", kwargs={"image_id": selected_obj.slave_image.id}
                    )
            except Exception:
                slave_thumb = None

        if selected_id_int is not None:
            visualize_base = reverse(
                "visualize_dic", kwargs={"dic_id": selected_id_int}
            )
            quiver_base = reverse(
                "serve_dic_quiver", kwargs={"dic_id": selected_id_int}
            )
            qs_str = urlencode(forward_params, doseq=True)
            visualize_url = visualize_base + ("?" + qs_str if qs_str else "")
            quiver_url = quiver_base + ("?" + qs_str if qs_str else "")

            # admin change URL for this DIC
            try:
                admin_change_url = reverse(
                    "admin:ppcx_app_dic_change", args=[selected_id_int]
                )
            except Exception:
                admin_change_url = None

            # admin change URLs for master/slave images (if present)
            admin_master_image_url = None
            admin_slave_image_url = None
            try:
                if selected_obj and selected_obj.master_image_id:
                    admin_master_image_url = reverse(
                        "admin:ppcx_app_image_change",
                        args=[selected_obj.master_image_id],
                    )
            except Exception:
                admin_master_image_url = None
            try:
                if selected_obj and selected_obj.slave_image_id:
                    admin_slave_image_url = reverse(
                        "admin:ppcx_app_image_change",
                        args=[selected_obj.slave_image_id],
                    )
            except Exception:
                admin_slave_image_url = None

        # Prev/Next preserve all current GET params and only change dic_id
        if selected_index is not None:
            if selected_index + 1 < len(dic_list):
                prev_id = dic_list[selected_index + 1].id
                prev_params = dict(request.GET.items())
                prev_params["dic_id"] = str(prev_id)
                prev_url = reverse("dic_visualizer") + "?" + urlencode(prev_params)
            if selected_index - 1 >= 0:
                next_id = dic_list[selected_index - 1].id
                next_params = dict(request.GET.items())
                next_params["dic_id"] = str(next_id)
                next_url = reverse("dic_visualizer") + "?" + urlencode(next_params)

    default_options = {k: request.GET.get(k, "") for k in sorted(viz_keys)}
    # pass current GET lists so template can preserve all values (including duplicates)
    current_get_lists = list(request.GET.lists())

    context = {
        "dic_list": dic_list,
        "selected_dic_id": int(selected_dic) if selected_dic else None,
        "selected_dic_obj": selected_obj,
        "master_thumb": master_thumb,
        "slave_thumb": slave_thumb,
        "visualize_url": visualize_url,
        "quiver_url": quiver_url,
        "prev_url": prev_url,
        "next_url": next_url,
        "admin_change_url": admin_change_url,
        "admin_master_image_url": admin_master_image_url,
        "admin_slave_image_url": admin_slave_image_url,
        "plot_opts": default_options,
        "filters": {
            "reference_date": q_reference_date or "",
            "master_timestamp_start": q_master_start or "",
            "master_timestamp_end": q_master_end or "",
            "camera_id": q_camera_id or "",
            "camera_name": q_camera_name or "",
            "dt_hours_min": q_time_diff_min or "",
            "dt_hours_max": q_time_diff_max or "",
            "dt_min_days": q_dt_min_days or "",
            "dt_max_days": q_dt_max_days or "",
            "dt_min_hours": q_dt_min_hours or "",
            "dt_max_hours": q_dt_max_hours or "",
            "month": q_month or "",
            "label": q_label or "",
            "year": q_year or "",
        },
        "current_get_lists": current_get_lists,
    }
    return render(request, "ppcx_app/dic_visualizer.html", context)


@require_http_methods(["POST"])
@csrf_protect
def set_dic_label(request, dic_id: int):
    """
    Set or clear the label for a DIC result.
    Accepts application/json or form-encoded POST with parameter 'label'.
    Returns JSON {status: 'ok', id: <id>, label: <new_label>}.
    """
    dic = get_object_or_404(DIC, id=dic_id)
    label = None
    try:
        if request.content_type and "application/json" in request.content_type:
            body = json.loads(request.body.decode("utf-8") or "{}")
            label = body.get("label", None)
        else:
            label = request.POST.get("label", None)
    except Exception:
        label = None

    if label is not None:
        label = label.strip() or None

    dic.label = label
    dic.save(update_fields=["label"])
    return JsonResponse({"status": "ok", "id": dic.id, "label": dic.label})


# ===================== COLLAPSES =======================
@require_http_methods(["GET"])
def visualize_collapse(request, collapse_id: int) -> HttpResponse:
    """
    Generate PNG visualization showing collapse geometry overlaid on the associated image.

    Query parameters:
      - figsize: "W,H" in inches (default "10,8")
      - dpi: output resolution (default 150)
      - outline_color: matplotlib color for polygon outline (default "red")
      - outline_width: line width for polygon outline (default 2)
      - fill_alpha: transparency for polygon fill (default 0.3)
    """
    from shapely import wkt as shapely_wkt

    collapse = get_object_or_404(Collapse, id=collapse_id)

    # Parse visualization parameters
    figsize_param = request.GET.get("figsize", "")
    if figsize_param:
        try:
            w, h = [float(x) for x in figsize_param.split(",", 1)]
            figsize = (w, h)
        except Exception:
            figsize = (10.0, 8.0)
    else:
        figsize = (10.0, 8.0)

    dpi = _parse_int(request.GET.get("dpi"), 150) or 150
    outline_color = request.GET.get("outline_color", "red")
    outline_width = _parse_float(request.GET.get("outline_width"), 2.0) or 2.0
    fill_alpha = _parse_float(request.GET.get("fill_alpha"), 0.0) or 0.0

    # Load the associated image
    if not collapse.image or not collapse.image.file_path:
        raise Http404("No image associated with this collapse")
    image_path = collapse.image.file_path
    if not os.path.exists(image_path):
        raise Http404("Image file not found on disk")
    try:
        pil_image = PILImage.open(image_path)
        # Use rotation field from the model
        if collapse.image.rotation and collapse.image.rotation != 0:
            pil_image = pil_image.rotate(-collapse.image.rotation, expand=True)
        image_array = np.array(pil_image)
    except Exception as e:
        raise Http404(f"Could not load image: {e}") from e

    # Extract geometry
    if not collapse.geom:
        raise Http404("No geometry associated with this collapse")

    try:
        # Convert Django GEOSGeometry to Shapely for easier coordinate extraction
        geom_wkt = collapse.geom.wkt
        shapely_geom = shapely_wkt.loads(geom_wkt)
        if shapely_geom.geom_type == "Polygon":
            xs, ys = shapely_geom.exterior.xy
        elif shapely_geom.geom_type == "MultiPolygon":
            for poly in shapely_geom.geoms:
                xs, ys = poly.exterior.xy
        # xs, ys = shapely_geom.exterior.xy
    except Exception as e:
        raise Http404(f"Could not extract geometry coordinates: {e}") from e

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display image
    ax.imshow(image_array)

    # Draw polygon outline
    ax.plot(xs, ys, color=outline_color, linewidth=outline_width)

    # Fill polygon with transparency
    ax.fill(xs, ys, facecolor=outline_color, edgecolor="none", alpha=fill_alpha)

    # Remove axes for cleaner look
    ax.set_axis_off()

    # Add title with metadata
    title_parts = [f"Collapse #{collapse_id}"]
    if collapse.image.datetime:
        date_str = collapse.image.datetime.strftime("%Y-%m-%d %H:%M")
        title_parts.append(f"Date: {date_str}")
    if collapse.area is not None:
        title_parts.append(f"Area: {collapse.area:.1f} px²")
    if collapse.volume is not None:
        title_parts.append(f"Volume: {collapse.volume:.1f} m³")

    fig.suptitle(" | ".join(title_parts), fontsize=12)
    fig.tight_layout()

    # Render to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    # Build informative filename
    date_str = "unknown"
    if collapse.image and collapse.image.datetime:
        date_str = collapse.image.datetime.strftime("%Y-%m-%d-%H-%M")
    filename = f"collapse_{collapse_id}_{date_str}.png"

    response = HttpResponse(buf.getvalue(), content_type="image/png")
    response["Content-Disposition"] = f'inline; filename="{filename}"'
    return response


@require_http_methods(["GET"])
def collapses_geojson(request) -> HttpResponse:
    """
    Serve all collapses as GeoJSON for use in QGIS or other GIS software.
    Uses geom_qgis field which has Y-coordinates inverted for proper QGIS display.
    """
    qs = Collapse.objects.select_related("image__camera").all()

    # Apply filters (same as before)
    # ...existing filter code...

    # Build GeoJSON using geom_qgis instead of geom
    features = []
    for collapse in qs:
        if not collapse.geom_qgis:
            continue

        # Handle MultiPolygon
        if collapse.geom_qgis.geom_type == "MultiPolygon":
            coordinates = [
                [[[x, y] for x, y in polygon[0].coords]]
                for polygon in collapse.geom_qgis
            ]
        else:
            coordinates = [[[x, y] for x, y in collapse.geom_qgis.coords[0]]]

        feature = {
            "type": "Feature",
            "id": collapse.id,
            "geometry": {"type": "MultiPolygon", "coordinates": coordinates},
            "properties": {
                "collapse_id": collapse.id,
                "image_id": collapse.image_id,
                "camera_name": collapse.image.camera.camera_name
                if collapse.image.camera
                else None,
                "acquisition_date": collapse.image.datetime.isoformat()
                if collapse.image.datetime
                else None,
                "area_px": collapse.area,
                "volume_m3": collapse.volume,
                "created_at": collapse.created_at.isoformat()
                if collapse.created_at
                else None,
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    response = JsonResponse(geojson, safe=False)
    response["Content-Disposition"] = 'attachment; filename="collapses.geojson"'
    return response


# ===================== VELOCITY TIMESERIES VIEWER =======================

import io
import json
import mimetypes
import os
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_http_methods
from PIL import Image as PILImage

from .functions.visualization import (
    draw_quiver_on_image_cv2,
    plot_dic_scatter,
    plot_dic_vectors,
)
from .models import DIC, Collapse, Image

matplotlib.use("Agg")  # Use non-interactive backend

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


@require_http_methods(["GET"])
def serve_dic_h5(request, dic_id: int) -> HttpResponse:
    """
    Return DIC HDF5 content as JSON for the specified DIC id.
    Query params: none required.
    """
    dic = get_object_or_404(DIC, id=dic_id)
    h5_path = dic.result_file_path
    if not h5_path or not os.path.exists(h5_path):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5_path, "r") as f:
            data: dict[str, Any] = {
                "points": f["points"][()].tolist() if "points" in f else None,
                "vectors": f["vectors"][()].tolist() if "vectors" in f else None,
                "magnitudes": f["magnitudes"][()].tolist()
                if "magnitudes" in f
                else None,
                "max_magnitude": float(f["max_magnitude"][()])
                if "max_magnitude" in f
                else None,
            }
        return JsonResponse(data)
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e


@require_http_methods(["GET"])
def serve_dic_h5_as_csv(request, dic_id: int) -> HttpResponse:
    """Return DIC data as CSV (x,y,u,v,magnitude)."""
    dic = get_object_or_404(DIC, id=dic_id)
    h5_path = dic.result_file_path
    if not h5_path or not os.path.exists(h5_path):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5_path, "r") as f:
            points = f["points"][()] if "points" in f else None
            vectors = f["vectors"][()] if "vectors" in f else None
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None

        if points is None or magnitudes is None:
            return HttpResponse("No DIC data available", status=404)

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
    h5 = dic.result_file_path
    if not h5 or not os.path.exists(h5):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5, "r") as f:
            points = f["points"][()] if "points" in f else None
            vectors = f["vectors"][()] if "vectors" in f else None
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e

    if points is None or magnitudes is None:
        raise Http404("DIC HDF5 missing required datasets ('points' or 'magnitudes')")

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
    # parse params
    plot_type = request.GET.get("plot_type", "quiver").lower()
    if plot_type not in ("quiver", "scatter"):
        return HttpResponse("Invalid plot_type. Use 'quiver' or 'scatter'.", status=400)

    show_background = _parse_bool(request.GET.get("background"), True)
    cmap_name = request.GET.get("cmap", "viridis")
    vmin = _parse_float(request.GET.get("vmin"), None)
    vmax = _parse_float(request.GET.get("vmax"), None)

    scale_raw = request.GET.get("scale", None)
    scale = (
        None
        if scale_raw is None or scale_raw.lower() == "none"
        else _parse_float(scale_raw, None)
    )
    scale_units = request.GET.get("scale_units", "xy")
    width = _parse_float(request.GET.get("width"), 0.003) or 0.003
    headwidth = _parse_float(request.GET.get("headwidth"), 2.5) or 2.5
    quiver_alpha = _parse_float(request.GET.get("quiver_alpha"), 1.0) or 1.0
    image_alpha = _parse_float(request.GET.get("image_alpha"), 0.7) or 0.7

    # filtering params
    min_velocity = _parse_float(request.GET.get("min_velocity"), None)
    filter_outliers = _parse_bool(request.GET.get("filter_outliers"), False)
    tails_percentile = _parse_float(request.GET.get("tails_percentile"), 0.01) or 0.01
    subsample = _parse_int(request.GET.get("subsample"), 1) or 1

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

    dic = get_object_or_404(DIC, id=dic_id)

    # load and filter data
    x, y, u, v, mag = load_and_filter_dic(
        dic,
        filter_outliers=filter_outliers,
        tails_percentile=tails_percentile,
        min_velocity=min_velocity
        if (min_velocity is not None and min_velocity > 0)
        else None,
        subsample=subsample,
    )

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
            vmin=vmin if vmin is not None else 0.0,
            vmax=vmax,
            scale=scale,
            scale_units=scale_units,
            width=width,
            headwidth=headwidth,
            quiver_alpha=quiver_alpha,
            image_alpha=image_alpha,
            cmap_name=cmap_name,
            figsize=figsize,
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
            figsize=figsize,
            dpi=dpi,
        )

    # title
    if dic.master_timestamp and dic.slave_timestamp:
        master_date = dic.master_timestamp.strftime("%Y-%m-%d %H:%M")
        slave_date = dic.slave_timestamp.strftime("%Y-%m-%d %H:%M")
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
        xs, ys = shapely_geom.exterior.xy
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
    if collapse.image.acquisition_timestamp:
        date_str = collapse.image.acquisition_timestamp.strftime("%Y-%m-%d %H:%M")
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
    if collapse.image and collapse.image.acquisition_timestamp:
        date_str = collapse.image.acquisition_timestamp.strftime("%Y-%m-%d-%H-%M")
    filename = f"collapse_{collapse_id}_{date_str}.png"

    response = HttpResponse(buf.getvalue(), content_type="image/png")
    response["Content-Disposition"] = f'inline; filename="{filename}"'
    return response

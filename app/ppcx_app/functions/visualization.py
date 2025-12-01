import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image as PILImage

# ================ Plotting Functions with matplotlib ================


def plot_dic_vectors(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | PILImage.Image | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    scale: float | None = None,
    scale_units: str = "xy",
    width: float = 0.005,
    headwidth: float = 2.5,
    quiver_alpha: float = 1,
    image_alpha: float = 0.7,
    cmap_name: str = "viridis",
    clabel: str = "Magnitude",
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object]:
    """Plot DIC displacement vectors using numpy arrays."""
    # Input validation
    arrays = [x, y, u, v, magnitudes]
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("Input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays are empty")

    # Set up color normalization
    min_magnitude = vmin if vmin is not None else 0.0
    max_magnitude = vmax if vmax is not None else np.percentile(magnitudes, 99)
    norm = Normalize(vmin=min_magnitude, vmax=max_magnitude)

    # Set up figure and axes
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display background image if provided
    if background_image is not None:
        ax.imshow(background_image, alpha=image_alpha)

    # Create quiver plot
    q = ax.quiver(
        x,
        y,
        u,
        v,
        magnitudes,
        scale=scale,
        scale_units=scale_units,
        angles="xy",
        cmap=cmap_name,
        norm=norm,
        width=width,
        headwidth=headwidth,
        alpha=quiver_alpha,
    )

    # Add colorbar
    cbar = fig.colorbar(q, ax=ax)
    cbar.set_label(clabel)

    # Set title and labels
    if title:
        ax.set_title(title)

    # Disable axis grid and labels
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return fig, ax, q


def plot_dic_scatter(
    x: np.ndarray,
    y: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap_name: str = "viridis",
    s: float = 20,
    alpha: float = 0.8,
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object]:
    """Plot DIC displacement data as a scatter plot colored by magnitude."""
    # Input validation
    if len(x) != len(y) or len(x) != len(magnitudes):
        raise ValueError("Input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays are empty")

    # Set up figure and axes
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display background image if provided
    if background_image is not None:
        ax.imshow(background_image, alpha=0.7)

    # Create scatter plot
    scatter = ax.scatter(
        x,
        y,
        c=magnitudes,
        cmap=cmap_name,
        s=s,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax or np.max(magnitudes),
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Displacement Magnitude (pixels)")

    # Set title and labels
    if title:
        ax.set_title(title)

    # Disable axis grid and labels
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return fig, ax, scatter


# ================ Plotting Functions with OpenCV ================


def draw_quiver_on_image_cv2(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    magnitudes: np.ndarray,
    colormap_name: str = "viridis",
    arrow_length_scale: float | None = None,
    arrow_thickness: int | None = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Draw arrows directly on the image using OpenCV. Returns a BGR image (uint8).

    If arrow_length_scale is None or <= 0 it will be auto-estimated from the
    image size and the typical vector length (median). The goal is to produce
    arrows that are visible but not oversized regardless of image resolution.

    arrow_thickness if None is auto-computed from image diagonal.
    """
    # ensure uint8 BGR image
    out = image.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    h, w = out.shape[:2]
    diag = np.hypot(w, h)

    # Safe arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    mag = np.asarray(magnitudes, dtype=float)

    # Auto-estimate arrow_length_scale when not provided or invalid.
    # Target arrow pixel length is a small fraction of image diagonal (3% by default),
    # but at least a few pixels so arrows are visible on small images.
    median_vec_len = 0.0
    if x.size:
        vec_lengths = np.hypot(u, v)
        median_vec_len = float(np.median(vec_lengths)) if vec_lengths.size else 0.0

    if arrow_length_scale is None or arrow_length_scale <= 0:
        target_px = max(6.0, diag * 0.005)  # target arrow length in pixels
        if median_vec_len <= 0:
            # no meaningful vector lengths -> use a small absolute scale
            arrow_length_scale = float(target_px)
        else:
            # scale such that typical vectors map to ~target_px
            arrow_length_scale = float(target_px / median_vec_len)
        # keep scale in reasonable bounds
        arrow_length_scale = float(np.clip(arrow_length_scale, 0.05, 2000.0))

    # Auto-estimate thickness if not provided
    if arrow_thickness is None or arrow_thickness <= 0:
        # thickness roughly proportional to image diagonal
        arrow_thickness = max(1, int(round(diag / 1000.0)))

    # Prepare colormap colors (BGR)
    cmap = cm.get_cmap(colormap_name)
    mag_min = mag.min() if mag.size else 0.0
    mag_max = mag.max() if mag.size else mag_min
    denom = max(1e-12, (mag_max - mag_min))
    mag_norm = (mag - mag_min) / denom
    colors_rgba = cmap(mag_norm)  # Nx4 in 0..1
    colors_bgr = (colors_rgba[:, :3] * 255)[:, ::-1].astype(np.uint8)  # RGB->BGR

    # Draw arrows - clip coordinates to image bounds
    for xi, yi, ui, vi, col in zip(x, y, u, v, colors_bgr, strict=False):
        sx = int(round(xi))
        sy = int(round(yi))
        ex = int(round(xi + ui * arrow_length_scale))
        ey = int(round(yi + vi * arrow_length_scale))

        # Clip to image bounds
        sx = int(np.clip(sx, 0, w - 1))
        sy = int(np.clip(sy, 0, h - 1))
        ex = int(np.clip(ex, 0, w - 1))
        ey = int(np.clip(ey, 0, h - 1))

        # skip degenerate arrows
        if sx == ex and sy == ey:
            # optionally draw a small dot for zero-length vectors
            cv2.circle(
                out,
                (sx, sy),
                radius=max(1, arrow_thickness),
                color=tuple(int(c) for c in col),
                thickness=-1,
            )
            continue

        # draw arrow - use tipLength relative to arrow pixel length (clamped)
        arrow_pixel_len = np.hypot(ex - sx, ey - sy)
        tipLength = float(np.clip(0.25, 0.08, 0.4))  # fixed reasonable tip size
        cv2.arrowedLine(
            out,
            (sx, sy),
            (ex, ey),
            color=tuple(int(c) for c in col),
            thickness=int(arrow_thickness),
            tipLength=tipLength,
        )

    # Return BGR uint8 image ready to encode/save
    return out

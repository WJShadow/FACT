"""
ImageJ-style Auto Brightness/Contrast display range estimation for video (T, H, W).

Estimates a suitable clipped min/max for video data with shape (T, H, W). Two strategies:
- per_frame_union: uniformly sample n frames (all T frames if n is None or 0), estimate
  per frame, then aggregate (minmax / mean / percentile).
- merged_histogram: uniform sampling as above, merge histograms, then estimate once.

Self-contained; does not depend on imagej_auto_brightness_contrast.py.
"""

import numpy as np
from typing import Tuple, Optional, Literal

# Matches ImageJ ContrastAdjuster
AUTO_THRESHOLD = 600
N_BINS = 256


# ---------------------------------------------------------------------------
# Single-frame estimation helpers (aligned with imagej_auto_brightness_contrast)
# ---------------------------------------------------------------------------


def _compute_histogram(
    img: np.ndarray,
    hist_min: float,
    hist_max: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute a 256-bin histogram consistent with ImageJ getRawStatistics().
    """
    if hist_max <= hist_min:
        hist_max = hist_min + 1.0
    bin_size = (hist_max - hist_min) / N_BINS

    if mask is not None:
        img = np.asarray(img, dtype=np.float64)
        valid = img[mask]
    else:
        valid = np.asarray(img, dtype=np.float64).ravel()

    valid = valid - hist_min
    valid = valid / bin_size
    valid = np.clip(valid, 0, N_BINS - 1).astype(np.int32)

    histogram = np.bincount(valid, minlength=N_BINS)
    return histogram, bin_size, hist_min


def _display_range_from_histogram(
    histogram: np.ndarray,
    bin_size: float,
    hmin_val: float,
    hist_min: float,
    hist_max: float,
    pixel_count: int,
    auto_threshold: int,
) -> Tuple[float, float]:
    """
    From a computed histogram and parameters, yield ImageJ-style display_min, display_max.
    Used for single-frame estimation and the merged-histogram path.
    """
    threshold = max(1, pixel_count // auto_threshold)
    limit = max(1, pixel_count // 10)

    hmin_bin = 0
    for i in range(N_BINS):
        count = int(histogram[i])
        if count > limit:
            count = 0
        if count > threshold:
            hmin_bin = i
            break
    else:
        hmin_bin = 0

    hmax_bin = N_BINS - 1
    for i in range(N_BINS - 1, -1, -1):
        count = int(histogram[i])
        if count > limit:
            count = 0
        if count > threshold:
            hmax_bin = i
            break
    else:
        hmax_bin = N_BINS - 1

    display_min = hmin_val + hmin_bin * bin_size
    display_max = hmin_val + hmax_bin * bin_size

    if display_max <= display_min:
        display_min = hist_min
        display_max = hist_max

    return (float(display_min), float(display_max))


def _estimate_display_range_single(
    img: np.ndarray,
    auto_threshold: int = AUTO_THRESHOLD,
    mask: Optional[np.ndarray] = None,
    hist_min: Optional[float] = None,
    hist_max: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Estimate display range for one frame (internal use; matches ImageJ Auto).
    """
    img = np.asarray(img)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != img.shape:
            raise ValueError("mask shape must match image shape")
        pixel_count = int(np.sum(mask))
    else:
        pixel_count = int(img.size)

    if pixel_count == 0:
        return (float(np.min(img)), float(np.max(img)))

    if hist_min is None or hist_max is None:
        data_min = float(np.min(img))
        data_max = float(np.max(img))
        if img.dtype == np.uint8:
            hist_min = 0.0
            hist_max = 256.0
        else:
            hist_min = data_min if hist_min is None else hist_min
            hist_max = data_max if hist_max is None else hist_max
    if hist_max <= hist_min:
        hist_max = hist_min + 1.0

    histogram, bin_size, hmin_val = _compute_histogram(img, hist_min, hist_max, mask)
    return _display_range_from_histogram(
        histogram, bin_size, hmin_val, hist_min, hist_max,
        pixel_count, auto_threshold,
    )


# ---------------------------------------------------------------------------
# Video sampling and the two estimation strategies
# ---------------------------------------------------------------------------


def _get_sampled_frame_indices(T: int, n_frames: int) -> np.ndarray:
    """Uniformly pick n_frames indices in [0, T-1]."""
    if T <= 0:
        return np.array([], dtype=np.int64)
    if n_frames >= T:
        return np.arange(T, dtype=np.int64)
    if n_frames <= 1:
        return np.array([T // 2], dtype=np.int64)
    indices = np.linspace(0, T - 1, n_frames, dtype=np.int64)
    return indices


def _estimate_per_frame_union(
    video: np.ndarray,
    indices: np.ndarray,
    aggregation: Literal["minmax", "mean", "percentile"],
    percentile: float,
    auto_threshold: int,
    mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    """
    Strategy I: estimate each sampled frame, then aggregate by aggregation mode.
    """
    dmins, dmaxes = [], []
    for t in indices:
        frame = video[t]
        frame_mask = mask[t] if mask is not None else None
        dmin, dmax = _estimate_display_range_single(
            frame,
            auto_threshold=auto_threshold,
            mask=frame_mask,
        )
        dmins.append(dmin)
        dmaxes.append(dmax)

    dmins = np.array(dmins)
    dmaxes = np.array(dmaxes)

    if aggregation == "minmax":
        return (float(np.max(dmins)), float(np.max(dmaxes)))
    if aggregation == "mean":
        return (float(np.mean(dmins)), float(np.mean(dmaxes)))
    if aggregation == "percentile":
        p_lo = percentile
        p_hi = 100.0 - percentile
        return (
            float(np.percentile(dmins, p_lo)),
            float(np.percentile(dmaxes, p_hi)),
        )
    raise ValueError(f"Unknown aggregation: {aggregation}")


def _estimate_merged_histogram(
    video: np.ndarray,
    indices: np.ndarray,
    auto_threshold: int,
    mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    """
    Strategy II: use a shared hist_min/hist_max per sampled frame, sum histograms,
    then estimate on the merged histogram.
    """
    # Global value range over sampled frames
    frames = [video[t] for t in indices]
    if mask is not None:
        valid = np.concatenate([frames[i][mask[indices[i]]] for i in range(len(indices))])
    else:
        valid = np.concatenate([f.ravel() for f in frames])

    if valid.size == 0:
        return (float(np.min(video)), float(np.max(video)))

    hist_min = float(np.min(valid))
    hist_max = float(np.max(valid))
    if hist_max <= hist_min:
        hist_max = hist_min + 1.0

    merged_hist = np.zeros(N_BINS, dtype=np.int64)
    total_pixel_count = 0

    for i, t in enumerate(indices):
        frame = video[t]
        frame_mask = mask[t] if mask is not None else None
        hist, bin_size, hmin_val = _compute_histogram(
            frame, hist_min, hist_max, frame_mask
        )
        merged_hist += hist
        if frame_mask is not None:
            total_pixel_count += int(np.sum(frame_mask))
        else:
            total_pixel_count += frame.size

    if total_pixel_count == 0:
        return (hist_min, hist_max)

    return _display_range_from_histogram(
        merged_hist,
        (hist_max - hist_min) / N_BINS,
        hist_min,
        hist_min,
        hist_max,
        total_pixel_count,
        auto_threshold,
    )


def estimate_display_range_video(
    video: np.ndarray,
    n_frames: Optional[int] = None,
    method: Literal["per_frame_union", "merged_histogram"] = "per_frame_union",
    aggregation: Literal["minmax", "mean", "percentile"] = "minmax",
    percentile: float = 10.0,
    auto_threshold: int = AUTO_THRESHOLD,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Estimate a suitable clipped min/max for video with shape (T, H, W).

    Parameters
    ----------
    video : ndarray, shape (T, H, W)
        Grayscale video; uint8, uint16, float, etc.
    n_frames : int, optional
        Number of frames sampled uniformly. If omitted (None) or 0, use all T frames;
        if n_frames >= T, use all frames.
    method : "per_frame_union" | "merged_histogram"
        - per_frame_union: ImageJ-style estimate per frame, then aggregate.
        - merged_histogram: merge histograms over sampled frames, estimate once.
    aggregation : "minmax" | "mean" | "percentile"
        Only when method="per_frame_union".
        - minmax: display_min = max(per-frame dmin), display_max = max(per-frame dmax).
        - mean: average dmin/dmax; tighter range.
        - percentile: use percentile and (100 - percentile) for robustness to outlier frames.
    percentile : float
        Only when aggregation="percentile"; e.g. 10 means 10th and 90th percentiles.
    auto_threshold : int
        ImageJ-style parameter, default 5000; smaller yields stronger contrast.
    mask : ndarray, optional, shape (T, H, W)
        Boolean array aligned with video; True = valid pixel; None = full frame.

    Returns
    -------
    display_min, display_max : float
        Suggested display minimum and maximum for clipping or normalizing the whole video.
    """
    video = np.asarray(video)
    if video.ndim != 3:
        raise ValueError("video must be 3D (T, H, W)")

    T, H, W = video.shape
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != video.shape:
            raise ValueError("mask shape must match video shape (T, H, W)")

    if n_frames is None or n_frames == 0:
        n_frames_eff = T
    else:
        n_frames_eff = int(n_frames)
        if n_frames_eff < 0:
            raise ValueError("n_frames must be non-negative")

    indices = _get_sampled_frame_indices(T, n_frames_eff)

    if method == "per_frame_union":
        return _estimate_per_frame_union(
            video, indices, aggregation, percentile, auto_threshold, mask,
        )
    if method == "merged_histogram":
        return _estimate_merged_histogram(video, indices, auto_threshold, mask)
    raise ValueError(f"Unknown method: {method}")


def apply_display_range(
    img: np.ndarray,
    display_min: float,
    display_max: float,
    out_dtype: type = np.uint8,
) -> np.ndarray:
    """
    Linearly map image from [display_min, display_max] to the output dtype range (e.g. 0–255),
    consistent with ImageJ display mapping. For a single frame or per-frame video processing.
    """
    img = np.asarray(img, dtype=np.float64)
    if display_max <= display_min:
        display_max = display_min + 1.0
    out = (img - display_min) / (display_max - display_min)
    out = np.clip(out, 0, 1)

    if out_dtype == np.uint8:
        return (out * 255).astype(np.uint8)
    if out_dtype == np.uint16:
        return (out * 65535).astype(np.uint16)
    return out.astype(out_dtype)
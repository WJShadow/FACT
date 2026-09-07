from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops


def _sample_indices(total_frames: int, sample_frames: int = 96) -> np.ndarray:
    if total_frames <= sample_frames:
        return np.arange(total_frames, dtype=np.int32)
    return np.unique(np.linspace(0, total_frames - 1, sample_frames, dtype=np.int32))


def _trimmed_mean(values: list[float], q_low: float = 0.20, q_high: float = 0.80) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    low, high = np.quantile(arr, [q_low, q_high])
    trimmed = arr[(arr >= low) & (arr <= high)]
    if trimmed.size == 0:
        trimmed = arr
    return float(np.mean(trimmed))


def _threshold_metrics(
    diff: np.ndarray,
    threshold: float,
    min_area: int,
    sample_frames: int,
) -> dict[str, float]:
    semimajors: list[float] = []
    solidities: list[float] = []
    eccentricities: list[float] = []
    mean_areas: list[float] = []

    for frame_index in _sample_indices(diff.shape[0], sample_frames):
        mask = diff[frame_index] > threshold
        labels, _ = ndimage.label(mask)
        props = [region for region in regionprops(labels) if region.area >= min_area]
        if not props:
            semimajors.append(float("nan"))
            solidities.append(float("nan"))
            eccentricities.append(float("nan"))
            mean_areas.append(float("nan"))
            continue

        semimajors.append(float(np.mean([region.major_axis_length / 2.0 for region in props])))
        solidities.append(float(np.mean([region.solidity for region in props])))
        eccentricities.append(float(np.mean([region.eccentricity for region in props])))
        mean_areas.append(float(np.mean([region.area for region in props])))

    return {
        "trimmed_semimajor": _trimmed_mean(semimajors, q_low=0.20, q_high=0.80),
        "mean_solidity": float(np.nanmean(solidities)),
        "mean_eccentricity": float(np.nanmean(eccentricities)),
        "mean_component_area": float(np.nanmean(mean_areas)),
    }


def auto_sens_targetrad(
    diff: np.ndarray,
    target_rad: float = 5.0,
    min_area: int = 4,
    sample_frames: int = 96,
) -> int:
    """
    Select the binary threshold for a (T, H, W) diff stack.

    Parameters
    ----------
    diff
        Input ndarray with shape (T, H, W).
    target_rad
        Desired semi-major axis for the dataset being processed.
    min_area
        Minimum connected-component area kept before ellipse fitting.
    sample_frames
        Number of evenly spaced frames sampled across time.

    Returns
    -------
    int
        Selected threshold, either 0 or -1.
    """
    diff = np.asarray(diff)
    if diff.ndim != 3:
        raise ValueError(f"`diff` must have shape (T, H, W), got {diff.shape!r}")
    if diff.shape[0] == 0:
        raise ValueError("`diff` must contain at least one frame.")

    metrics_0 = _threshold_metrics(diff, threshold=0.0, min_area=min_area, sample_frames=sample_frames)
    metrics_n1 = _threshold_metrics(diff, threshold=-1.0, min_area=min_area, sample_frames=sample_frames)

    size_error_0 = abs(metrics_0["trimmed_semimajor"] - float(target_rad))
    size_error_n1 = abs(metrics_n1["trimmed_semimajor"] - float(target_rad))
    size_improvement = size_error_0 - size_error_n1

    compactness_gain = (
        (metrics_n1["mean_solidity"] - metrics_0["mean_solidity"])
        + (metrics_n1["mean_eccentricity"] - metrics_0["mean_eccentricity"])
        - 0.001 * (metrics_n1["mean_component_area"] - metrics_0["mean_component_area"])
    )

    return -1 if (size_improvement > 0.0 and compactness_gain > 0.0) else 0

"""Measure original TIFF fluorescence on final neuronal masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from PostSlice.PostProcess_Pipeline import Post_FACT_guided_backgroundremoving


def measure_raw_tiff_traces(
    masks_final,
    raw_video: np.ndarray,
    *,
    infer_mask3d: np.ndarray | None = None,
    run_overlap_demix: bool = True,
    use_nnls: bool = True,
    alpha_default: float = 0.7,
    alpha_max: float = 0.9,
    alpha_min: float | None = None,
    eps: float = 1e-8,
    sigma_frames: float = 5.0,
    strength: float = 0.35,
    ov_cap: float = 5.0,
    min_overlap_pixels: int = 1,
    preserve_amplitude: bool = True,
    amplitude_method: str = "rms",
) -> dict[str, Any]:
    """Extract ROI, background and final traces without modifying inputs.

    Supply a TIFF array shaped (T, H, W) and masks shaped (N, H*W).
    Every mask is measured by its pixel mean, including demixed masks.
    Overlap correction requires the aligned network mask, infer_mask3d.
    """
    video = np.asarray(raw_video)
    if video.ndim != 3 or any(size == 0 for size in video.shape):
        raise ValueError("raw_video must have nonempty shape (T, H, W).")
    if video.dtype.kind not in "biuf" or not np.all(np.isfinite(video)):
        raise ValueError("raw_video must contain finite real intensities.")
    frames, height, width = video.shape
    if not hasattr(masks_final, "shape") or len(masks_final.shape) != 2:
        raise ValueError("masks_final must have shape (N, H*W).")
    if masks_final.shape[1] != height * width:
        raise ValueError("Mask pixels must match the TIFF spatial dimensions.")
    masks = (sp.csr_matrix(masks_final, copy=True) > 0).astype(np.int64)
    if run_overlap_demix:
        if infer_mask3d is None or np.shape(infer_mask3d) != video.shape:
            raise ValueError("Overlap correction requires infer_mask3d with shape (T, H, W).")

    result = Post_FACT_guided_backgroundremoving(
        masks,
        None,
        np.zeros(masks.shape[0], dtype=bool),
        infer_mask3d=infer_mask3d,
        input_img=video,
        mask_shape_2d=(height, width),
        run_overlap_demix=run_overlap_demix,
        use_nnls=use_nnls,
        alpha_default=alpha_default,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        eps=eps,
        sigma_frames=sigma_frames,
        strength=strength,
        ov_cap=ov_cap,
        min_overlap_pixels=min_overlap_pixels,
        preserve_amplitude=preserve_amplitude,
        amplitude_method=amplitude_method,
    )
    result["metadata"] = {
        "format_version": 1,
        "trace_group": "original_tiff",
        "source": "raw",
        "units": "raw_intensity",
        "video_shape": [int(frames), int(height), int(width)],
        "mask_shape_2d": [int(height), int(width)],
        "row_indices": list(range(masks.shape[0])),
        "roi_identity": "Zero-based row index in the supplied final masks",
        "extraction_method": "mask_mean",
        "background_mode": "automatic" if use_nnls else "fixed",
        "alpha_default": float(alpha_default),
        "alpha_max": float(alpha_max) if np.isfinite(alpha_max) else None,
        "alpha_min": None if alpha_min is None else float(alpha_min),
        "eps": float(eps),
        "coefficients": np.asarray(result["alpha_nnls"]).tolist(),
        "coefficient_status": np.asarray(result["alpha_status"]).tolist(),
        "empty_background": np.asarray(result["empty_background"]).tolist(),
        "overlap_demixing": bool(run_overlap_demix),
        "overlap_applied": np.asarray(result["sources"], dtype=bool).tolist(),
        "overlap_settings": {
            "sigma_frames": float(sigma_frames),
            "strength": float(strength),
            "ov_cap": float(ov_cap),
            "min_overlap_pixels": int(min_overlap_pixels),
            "preserve_amplitude": bool(preserve_amplitude),
            "amplitude_method": amplitude_method,
        },
        "stages": {
            "traces_cell": "roi_before_background_subtraction",
            "traces_bg": "local_background_mean",
            "traces_final": "final_corrected",
        },
    }
    return result


__all__ = ["measure_raw_tiff_traces"]

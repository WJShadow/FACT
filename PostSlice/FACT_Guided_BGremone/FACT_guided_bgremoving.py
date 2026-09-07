"""FACT-guided background removal followed by overlap demixing."""

from typing import Any, Dict, Tuple

import numpy as np

from PostSlice.com.Overlap_cancel import fact_guided_overlap_demixing
from PostSlice.FACT_Guided_BGremone.Fact_Guided_Surround_Subtraction import (
    fact_guided_bgremove,
)


def fact_guided_bgremoving(
    masks_final,
    traces_final,
    sources,
    *,
    infer_mask3d: np.ndarray,
    input_img: np.ndarray,
    mask_shape_2d: Tuple[int, int],
    use_nnls: bool = True,
    alpha_default: float = 0.7,
    alpha_max: float = 0.9,
    alpha_min: float = None,
    eps: float = 1e-8,
    sigma_frames: float = 5.0,
    strength: float = 0.35,
    ov_cap: float = 5.0,
    min_overlap_pixels: int = 1,
    preserve_amplitude: bool = True,
    amplitude_method: str = "rms",
) -> Dict[str, Any]:
    """
    Run ``fact_guided_bgremove`` then ``fact_guided_overlap_demixing``.

    Returns a dict with:
        masks_final, traces_final, sources,
        traces_bg, traces_cell, alpha_nnls (from background removal).
    """
    bg_results = fact_guided_bgremove(
        masks=masks_final,
        img_shape_2d=mask_shape_2d,
        input_vid=input_img,
        traces_cell_override=traces_final,
        override_valid_mask=sources,
        use_nnls=use_nnls,
        alpha_default=alpha_default,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        eps=eps,
    )

    traces_bg = bg_results["traces_bg"]
    traces_cell = bg_results["traces_cell"]
    alpha_nnls = bg_results["alpha_nnls"]
    traces_final = bg_results["traces_bg_rm"].astype(np.float32)

    masks_final, traces_final, sources = fact_guided_overlap_demixing(
        masks_final,
        traces_final,
        sources,
        infer_mask3d=infer_mask3d,
        input_img=input_img,
        mask_shape_2d=mask_shape_2d,
        sigma_frames=sigma_frames,
        strength=strength,
        ov_cap=ov_cap,
        min_overlap_pixels=min_overlap_pixels,
        preserve_amplitude=preserve_amplitude,
        amplitude_method=amplitude_method,
    )

    masks_final = (masks_final.tocsr() > 0).astype(np.int64)

    return {
        "masks_final": masks_final,
        "traces_final": traces_final,
        "sources": sources,
        "traces_bg": traces_bg,
        "traces_cell": traces_cell,
        "alpha_nnls": alpha_nnls,
    }

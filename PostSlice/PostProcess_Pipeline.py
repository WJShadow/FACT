"""Unified post-processing entry points. Original PostSlice functions are unchanged."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def empty_masks(pixel_count: int, dtype: Any = np.int64) -> sp.csr_matrix:
    return sp.csr_matrix((0, int(pixel_count)), dtype=dtype)


def timestamp_list(timestamps: Any, row_count: int) -> list[np.ndarray]:
    row_count = int(row_count)
    if row_count == 0:
        return []
    if isinstance(timestamps, (list, tuple)):
        if len(timestamps) != row_count:
            if row_count == 1:
                return [np.asarray(timestamps, dtype=np.int64).reshape(-1)]
            raise ValueError(
                f"Timestamp rows ({len(timestamps)}) do not match masks ({row_count})."
            )
        return [np.asarray(value, dtype=np.int64).reshape(-1) for value in timestamps]

    array = np.asarray(timestamps)
    if row_count == 1:
        return [array.astype(np.int64, copy=False).reshape(-1)]
    if array.ndim == 1 and array.size == row_count:
        return [np.asarray([value], dtype=np.int64) for value in array]
    if array.ndim >= 2 and array.shape[0] == row_count:
        return [np.asarray(array[index], dtype=np.int64).reshape(-1) for index in range(row_count)]
    raise ValueError(
        f"Cannot align timestamp shape {array.shape!r} to {row_count} mask rows."
    )


def threshold_sparse_rows(masks: sp.csr_matrix, threshold_fraction: float) -> sp.csr_matrix:
    masks = masks.tocsr()
    if masks.shape[0] == 0:
        return empty_masks(masks.shape[1])
    rows: list[sp.csr_matrix] = []
    for index in range(masks.shape[0]):
        row = masks.getrow(index)
        maximum = float(row.max()) if row.nnz else 0.0
        if maximum <= 0:
            rows.append(sp.csr_matrix((1, masks.shape[1]), dtype=np.int64))
        else:
            rows.append((row >= maximum * float(threshold_fraction)).astype(np.int64))
    return sp.vstack(rows, format="csr").astype(np.int64)


def refine_masks_safe(
    masks: sp.csr_matrix,
    timestamps: Any,
    *,
    thresh_dist: float,
    thresh_border: float,
    mask_shape_2d: tuple[int, int],
) -> tuple[sp.csr_matrix, list[np.ndarray]]:
    masks = masks.tocsr()
    timestamp_rows = timestamp_list(timestamps, masks.shape[0])
    if masks.shape[0] <= 1:
        return threshold_sparse_rows(masks, thresh_border), timestamp_rows

    from PostSlice.refine_masks import refine_masks

    refined, refined_timestamps = refine_masks(
        masks=masks,
        thresh_dist=float(thresh_dist),
        thresh_boarder=float(thresh_border),
        img_shape_2d=mask_shape_2d,
        valid_arr=None,
        timestamps=timestamp_rows,
    )
    return refined.tocsr(), timestamp_list(refined_timestamps, refined.shape[0])


def filter_active_safe(
    masks: sp.csr_matrix,
    timestamps: Any,
    *,
    thresh_active: int,
    thresh_refine: float,
) -> tuple[sp.csr_matrix, list[np.ndarray]]:
    masks = masks.tocsr()
    timestamp_rows = timestamp_list(timestamps, masks.shape[0])
    keep: list[int] = []
    if int(thresh_active) > 1:
        width = int(thresh_active)
        for index, values in enumerate(timestamp_rows):
            values = np.asarray(values, dtype=np.int64).reshape(-1)
            active = False
            if values.size >= width:
                active = bool(np.any(values[width - 1 :] - values[: 1 - width] == width - 1))
            if active:
                keep.append(index)
    else:
        keep = list(range(masks.shape[0]))

    if not keep:
        return empty_masks(masks.shape[1]), []
    selected = masks[keep].tocsr()
    selected_timestamps = [timestamp_rows[index] for index in keep]
    return threshold_sparse_rows(selected, thresh_refine), selected_timestamps


def coerce_trace_matrix(
    values: Any,
    row_count: int,
    frame_count: int,
    *,
    name: str,
) -> np.ndarray:
    if sp.issparse(values):
        values = values.toarray()
    array = np.asarray(values, dtype=np.float32)
    expected = (int(row_count), int(frame_count))
    if array.size == 0 and row_count == 0:
        return np.zeros(expected, dtype=np.float32)
    if array.ndim == 1 and row_count == 1:
        array = array.reshape(1, -1)
    if array.shape != expected:
        raise ValueError(f"{name} has shape {array.shape!r}; expected {expected!r}.")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def extract_roi_mean_traces(masks: sp.spmatrix, video: np.ndarray) -> np.ndarray:
    mask_rows = (masks.tocsr() > 0).astype(np.float32)
    movie = np.asarray(video, dtype=np.float32)
    frame_count = int(movie.shape[0])
    if mask_rows.shape[0] == 0:
        return np.zeros((0, frame_count), dtype=np.float32)
    if mask_rows.shape[1] != int(movie.shape[1] * movie.shape[2]):
        raise ValueError("Spatial masks and postprocess trace video have different shapes.")
    areas = np.asarray(mask_rows.sum(axis=1), dtype=np.float32).reshape(-1)
    areas = np.maximum(areas, 1.0)
    flat_frames = movie.reshape(frame_count, -1)
    traces = np.asarray(mask_rows @ flat_frames.T, dtype=np.float32)
    traces /= areas[:, None]
    return np.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)


def apply_sc4_small_mask_filter(
    masks: sp.csr_matrix,
    traces: Any,
    sources: Any,
    mask_shape_2d: tuple[int, int],
    *,
    enabled: bool,
    sc4_min_area: int,
) -> tuple[sp.csr_matrix, Any, Any]:
    if not enabled:
        return masks, traces, sources
    from PostSlice.Morphology import filter_small_tr

    return filter_small_tr(
        masks,
        sc4_min_area,
        mask_shape_2d,
        traces=traces,
        sources=sources,
    )


def post_slice_connect(
    infer_3d=None,
    min_area=None,
    max_area=None,
    num_workers=8,
    infer_mask3d=None,
):
    from PostSlice.slice_connect import slice_connect

    mask = infer_mask3d if infer_mask3d is not None else infer_3d
    return slice_connect(
        infer_3d=mask,
        min_area=min_area,
        max_area=max_area,
        num_workers=num_workers,
    )


def post_merge_com(masks, coms, timestamps, thresh_com, *, skip_if_empty: bool = False):
    if skip_if_empty and int(masks.shape[0]) == 0:
        return masks, timestamps
    from PostSlice.merger import merge_COM

    return merge_COM(masks, coms, timestamps, thresh_com)


def post_refine_masks(
    masks,
    timestamps=None,
    *,
    thresh_dist,
    thresh_border=None,
    mask_shape_2d=None,
    empty_safe: bool = False,
    thresh_boarder=None,
    img_shape_2d=None,
    valid_arr=None,
):
    if thresh_border is None:
        thresh_border = thresh_boarder
    if mask_shape_2d is None:
        mask_shape_2d = img_shape_2d
    if empty_safe:
        return refine_masks_safe(
            masks.tocsr() if sp.issparse(masks) else masks,
            timestamps,
            thresh_dist=thresh_dist,
            thresh_border=thresh_border,
            mask_shape_2d=mask_shape_2d,
        )
    from PostSlice.refine_masks import refine_masks

    return refine_masks(
        masks=masks,
        thresh_dist=thresh_dist,
        thresh_boarder=thresh_border,
        img_shape_2d=mask_shape_2d,
        valid_arr=valid_arr,
        timestamps=timestamps,
    )


def post_merge_iou(masks, timestamps, *, thresh_refine, thresh_iou=None, thresh_IoU=None):
    from PostSlice.further_merge import merge_IoU

    if thresh_iou is None:
        thresh_iou = thresh_IoU
    return merge_IoU(
        masks=masks,
        timestamps=timestamps,
        thresh_refine=thresh_refine,
        thresh_IoU=thresh_iou,
    )


def post_merge_consume(masks, timestamps, *, thresh_refine, thresh_consume, max_area):
    from PostSlice.further_merge import merge_consume

    return merge_consume(
        masks=masks,
        timestamps=timestamps,
        thresh_refine=thresh_refine,
        thresh_consume=thresh_consume,
        max_area=max_area,
    )


def post_merge_overlap(
    masks,
    timestamps,
    *,
    thresh_refine,
    thresh_iou,
    thresh_consume,
    max_area,
    skip_if_empty: bool = False,
):
    if skip_if_empty and int(masks.shape[0]) == 0:
        return masks, timestamps
    masks, timestamps = post_merge_iou(
        masks,
        timestamps,
        thresh_refine=thresh_refine,
        thresh_iou=thresh_iou,
    )
    return post_merge_consume(
        masks,
        timestamps,
        thresh_refine=thresh_refine,
        thresh_consume=thresh_consume,
        max_area=max_area,
    )


def post_filter_active(
    masks,
    timestamps,
    thresh_acti,
    thresh_refine,
    *,
    empty_safe: bool = False,
):
    if empty_safe:
        return filter_active_safe(
            masks.tocsr() if sp.issparse(masks) else masks,
            timestamps,
            thresh_active=thresh_acti,
            thresh_refine=thresh_refine,
        )
    from PostSlice.filter_active import filter_active

    return filter_active(masks, timestamps, thresh_acti, thresh_refine)


def Post_standard_spatial_consolidation(
    infer_mask3d: np.ndarray,
    *,
    mask_shape_2d: tuple[int, int],
    min_area,
    max_area,
    thresh_com,
    thresh_com_refine,
    thresh_refine,
    thresh_iou,
    thresh_consume,
    thresh_acti,
    num_workers=8,
    empty_safe: bool = False,
):
    masks, coms, _areas, timestamps = post_slice_connect(
        infer_mask3d,
        min_area=min_area,
        max_area=max_area,
        num_workers=num_workers,
    )
    coms = coms.astype(np.float32)
    masks, timestamps = post_merge_com(masks, coms, timestamps, thresh_com)
    masks, timestamps = post_refine_masks(
        masks,
        timestamps,
        thresh_dist=thresh_com_refine,
        thresh_border=thresh_refine,
        mask_shape_2d=mask_shape_2d,
        empty_safe=empty_safe,
    )
    masks, timestamps = post_merge_overlap(
        masks,
        timestamps,
        thresh_refine=thresh_refine,
        thresh_iou=thresh_iou,
        thresh_consume=thresh_consume,
        max_area=max_area,
    )
    masks, timestamps = post_filter_active(
        masks,
        timestamps,
        thresh_acti,
        thresh_refine,
        empty_safe=empty_safe,
    )
    return masks, timestamps


def Post_FACT_guided_demixing(
    masks,
    infer_mask3d,
    input_img,
    mask_shape_2d,
    *,
    enabled: bool = True,
    **demix_kwargs,
):
    if not enabled:
        traces = extract_roi_mean_traces(masks, input_img)
        sources = np.zeros((masks.shape[0],), dtype=bool)
        return masks, traces, sources
    from PostSlice.FACT_Guided_Demix_Sp import fact_guided_demixing

    return fact_guided_demixing(
        masks_final=masks,
        infer_mask3d=infer_mask3d,
        input_img=input_img,
        mask_shape_2d=mask_shape_2d,
        **demix_kwargs,
    )


def Post_additional_morphological_blocks(
    masks,
    mask_shape_2d: tuple[int, int],
    *,
    timestamps=None,
    traces=None,
    sources=None,
    sc1: bool = False,
    sc1_area_min=None,
    sc1_area_max=None,
    sc1_kernel_open=None,
    sc1_kernel_close=None,
    sc2: bool = False,
    sc2_params=None,
    sc3: bool = False,
    sc3_area_min=0,
    sc3_area_max=None,
    sc3_kernel=None,
    sc4: bool = False,
    sc4_min_area=None,
    binarize_after_each: bool = False,
    copy_inputs: bool = False,
):
    from PostSlice.Morphology import (
        enlarge_control_tr,
        filter_masks_ellipse_tr,
        filter_small_tr,
        openclose_tr,
    )

    if sc1:
        sc_masks = masks.copy() if copy_inputs else masks
        sc_traces = np.asarray(traces).copy() if copy_inputs and traces is not None else traces
        sc_sources = np.asarray(sources).copy() if copy_inputs and sources is not None else sources
        masks, traces, sources = openclose_tr(
            sc_masks,
            timestamps,
            sc1_area_min,
            sc1_area_max,
            mask_shape_2d,
            sc1_kernel_open,
            sc1_kernel_close,
            traces=sc_traces,
            sources=sc_sources,
        )
    if binarize_after_each:
        masks = (masks.tocsr() > 0).astype(np.int64)

    if sc2:
        sc_masks = masks.copy() if copy_inputs else masks
        sc_traces = np.asarray(traces).copy() if copy_inputs and traces is not None else traces
        sc_sources = np.asarray(sources).copy() if copy_inputs and sources is not None else sources
        masks, traces, sources = filter_masks_ellipse_tr(
            sc_masks,
            mask_shape_2d,
            params=list(sc2_params) if sc2_params is not None else sc2_params,
            traces=sc_traces,
            sources=sc_sources,
        )
    if binarize_after_each:
        masks = (masks.tocsr() > 0).astype(np.int64)

    if sc3:
        sc_masks = masks.copy() if copy_inputs else masks
        sc_traces = np.asarray(traces).copy() if copy_inputs and traces is not None else traces
        sc_sources = np.asarray(sources).copy() if copy_inputs and sources is not None else sources
        masks, traces, sources = enlarge_control_tr(
            sc_masks,
            sc3_area_min,
            sc3_area_max,
            mask_shape_2d,
            sc3_kernel,
            traces=sc_traces,
            sources=sc_sources,
        )
    if binarize_after_each:
        masks = (masks.tocsr() > 0).astype(np.int64)

    if sc4:
        sc_masks = masks.copy() if copy_inputs else masks
        sc_traces = np.asarray(traces).copy() if copy_inputs and traces is not None else traces
        sc_sources = np.asarray(sources).copy() if copy_inputs and sources is not None else sources
        masks, traces, sources = filter_small_tr(
            sc_masks,
            sc4_min_area,
            mask_shape_2d,
            traces=sc_traces,
            sources=sc_sources,
        )
    if binarize_after_each:
        masks = (masks.tocsr() > 0).astype(np.int64)

    return masks, traces, sources


def Post_FACT_guided_backgroundremoving(
    masks,
    traces,
    sources,
    *,
    infer_mask3d,
    input_img,
    mask_shape_2d: tuple[int, int],
    run_overlap_demix: bool = True,
    binarize_final: bool = True,
    **kwargs,
):
    if run_overlap_demix and binarize_final:
        from PostSlice.FACT_Guided_BGremone.FACT_guided_bgremoving import (
            fact_guided_bgremoving,
        )

        return fact_guided_bgremoving(
            masks,
            traces,
            sources,
            infer_mask3d=infer_mask3d,
            input_img=input_img,
            mask_shape_2d=mask_shape_2d,
            **kwargs,
        )

    from PostSlice.FACT_Guided_BGremone.Fact_Guided_Surround_Subtraction import (
        fact_guided_bgremove,
    )

    bg_kwargs = {
        key: kwargs[key]
        for key in ("use_nnls", "alpha_default", "alpha_max", "alpha_min", "eps")
        if key in kwargs
    }
    bg_results = fact_guided_bgremove(
        masks=masks,
        img_shape_2d=mask_shape_2d,
        input_vid=input_img,
        traces_cell_override=traces,
        override_valid_mask=sources,
        **bg_kwargs,
    )
    traces_out = bg_results["traces_bg_rm"].astype(np.float32)
    if not run_overlap_demix:
        return {
            "masks_final": masks,
            "traces_final": traces_out,
            "sources": sources,
            "traces_bg": bg_results["traces_bg"],
            "traces_cell": bg_results["traces_cell"],
            "alpha_nnls": bg_results["alpha_nnls"],
            "alpha_fallback_count": bg_results.get("alpha_fallback_count", 0),
        }

    from PostSlice.com.Overlap_cancel import fact_guided_overlap_demixing

    overlap_kwargs = {
        key: kwargs[key]
        for key in (
            "sigma_frames",
            "strength",
            "ov_cap",
            "min_overlap_pixels",
            "preserve_amplitude",
            "amplitude_method",
        )
        if key in kwargs
    }
    masks_out, traces_out, sources_out = fact_guided_overlap_demixing(
        masks,
        traces_out,
        sources,
        infer_mask3d=infer_mask3d,
        input_img=input_img,
        mask_shape_2d=mask_shape_2d,
        **overlap_kwargs,
    )
    if binarize_final:
        masks_out = (masks_out.tocsr() > 0).astype(np.int64)
    return {
        "masks_final": masks_out,
        "traces_final": traces_out,
        "sources": sources_out,
        "traces_bg": bg_results["traces_bg"],
        "traces_cell": bg_results["traces_cell"],
        "alpha_nnls": bg_results["alpha_nnls"],
        "alpha_fallback_count": bg_results.get("alpha_fallback_count", 0),
    }

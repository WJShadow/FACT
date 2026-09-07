## Integrated ARBS in FACT

import numpy as np
import scipy.sparse as sp
from skimage import morphology, measure
import cv2
from scipy.optimize import nnls

def fact_guided_bgremove(
    masks,
    img_shape_2d,
    input_vid,
    traces_cell_override=None,
    override_valid_mask=None,
    use_nnls=True,
    alpha_default=0.7,
    alpha_max=0.9,
    alpha_min=None,
    eps=1e-8,
):

    ''' Remove background fluctuation using the adaptive-annualar background removal methods 
    masks: all masks segmented 
    img_shape_2d : the spatial shape of input_vid
    input_vid: TxHxW inp
    
    
    '''
    if len(img_shape_2d) != 2:
        raise ValueError("img_shape_2d must be (H, W).")
    (H, W) = img_shape_2d
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive.")

    if not hasattr(masks, "shape") or len(masks.shape) != 2:
        raise ValueError("masks must be 2D with shape (n, H*W).")
    n, p = masks.shape
    if p != H * W:
        raise ValueError("masks second dimension must be H*W.")

    if input_vid.ndim != 3:
        raise ValueError("input_vid must have shape (T, H, W).")
    if input_vid.shape[1] != H or input_vid.shape[2] != W:
        raise ValueError("input_vid spatial shape must match img_shape_2d.")

    if alpha_min is not None and alpha_min > alpha_max:
        raise ValueError("alpha_min must be <= alpha_max.")

    t_len = input_vid.shape[0]
    if traces_cell_override is not None:
        traces_cell_override = np.asarray(traces_cell_override, dtype=np.float64)
        if traces_cell_override.ndim != 2:
            raise ValueError("traces_cell_override must be 2D with shape (n, T).")
        if traces_cell_override.shape[0] != n or traces_cell_override.shape[1] != t_len:
            raise ValueError("traces_cell_override shape must match (n, T).")
    if override_valid_mask is not None:
        override_valid_mask = np.asarray(override_valid_mask, dtype=bool).reshape(-1)
        if override_valid_mask.shape[0] != n:
            raise ValueError("override_valid_mask length must be n.")
    elif traces_cell_override is not None:
        override_valid_mask = np.all(np.isfinite(traces_cell_override), axis=1)

    if n == 0:
        return {
            'traces_cell': np.zeros((0, t_len), dtype=np.float64),
            'traces_bg': np.zeros((0, t_len), dtype=np.float64),
            'traces_bg_rm': np.zeros((0, t_len), dtype=np.float64),
            'alpha_nnls': np.zeros((0,), dtype=np.float64),
            'alpha_method': 'nnls' if use_nnls else 'fixed_default',
            'alpha_default': float(alpha_default),
            'alpha_max': float(alpha_max),
            'alpha_min': None if alpha_min is None else float(alpha_min),
            'alpha_fallback_count': 0,
        }

    # Build per-neuron exclusion zones and rings with guaranteed length n.
    exclusion_zones = np.zeros((n, H, W), dtype=bool)
    background_rings = np.zeros((n, H, W), dtype=bool)
    exclusion_zone_collection = np.zeros((H, W), dtype=bool)
    mask_planes = np.zeros((n, H, W), dtype=bool)
    geometry = []
    for i in range(n):
        if sp.issparse(masks):
            mask_vec = masks.getrow(i).toarray().ravel()
        else:
            mask_vec = np.asarray(masks[i]).ravel()
        if mask_vec.size != H * W:
            raise ValueError("Each mask row must have H*W elements.")
        mask = mask_vec.reshape(H, W) > 0
        mask_planes[i] = mask

        # Safe fallback for empty masks.
        if not np.any(mask):
            geometry.append(None)
            continue

        labeled_mask = measure.label(mask)
        props = measure.regionprops(labeled_mask)

        if props:
            ellipse = max(props, key=lambda r: r.area)
            if hasattr(ellipse, "axis_major_length"):
                major_axis_length = float(ellipse.axis_major_length)
                minor_axis_length = float(ellipse.axis_minor_length)
            else:
                major_axis_length = float(ellipse.major_axis_length)
                minor_axis_length = float(ellipse.minor_axis_length)
        else:
            # Robust fallback when regionprops is unavailable.
            coords = np.argwhere(mask)
            y_span = float(coords[:, 0].max() - coords[:, 0].min() + 1)
            x_span = float(coords[:, 1].max() - coords[:, 1].min() + 1)
            major_axis_length = max(y_span, x_span)
            minor_axis_length = min(y_span, x_span)

        dilation_radius_exclusion = int((major_axis_length + minor_axis_length) / 8)
        dilation_radius_exclusion = max(1, dilation_radius_exclusion)
        dilation_radius_background = int((major_axis_length + minor_axis_length) / 6)
        dilation_radius_background = max(3, dilation_radius_background)
        geometry.append((dilation_radius_exclusion, dilation_radius_background))

    radii = sorted({radius for pair in geometry if pair is not None for radius in pair})
    disk_cache = {radius: morphology.disk(radius) for radius in radii}

    def build_ring(i):
        pair = geometry[i]
        if pair is None:
            return i, None, None
        exclusion_radius, background_radius = pair
        mask = mask_planes[i]
        exclusion_zone = morphology.dilation(mask, disk_cache[exclusion_radius])
        background_ring = morphology.dilation(mask, disk_cache[background_radius])
        return i, exclusion_zone, background_ring & ~exclusion_zone

    nonempty_indices = [i for i, pair in enumerate(geometry) if pair is not None]
    if len(nonempty_indices) >= 8:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(nonempty_indices))) as executor:
            ring_results = executor.map(build_ring, nonempty_indices)
            for i, exclusion_zone, background_ring in ring_results:
                exclusion_zones[i] = exclusion_zone
                background_rings[i] = background_ring
                exclusion_zone_collection |= exclusion_zone
    else:
        for i in nonempty_indices:
            _, exclusion_zone, background_ring = build_ring(i)
            exclusion_zones[i] = exclusion_zone
            background_rings[i] = background_ring
            exclusion_zone_collection |= exclusion_zone

    for i in range(n):
        background_rings[i] = background_rings[i] & ~exclusion_zone_collection

    background_ring_sparse = sp.csr_matrix(background_rings.reshape(n, -1))
    current_masks = masks.tocsr() if sp.issparse(masks) else sp.csr_matrix(masks)
    current_bg_masks = background_ring_sparse

    areas_current = np.asarray(current_masks.astype(np.int64).sum(axis=1)).ravel().astype(np.float64)
    areas_bg_current = np.asarray(current_bg_masks.astype(np.int64).sum(axis=1)).ravel().astype(np.float64)
    safe_areas_current = np.where(areas_current > eps, areas_current, 1.0)
    safe_areas_bg_current = np.where(areas_bg_current > eps, areas_bg_current, 1.0)

    traces_current = np.zeros((n, t_len), dtype=np.float64)
    traces_bg_current = np.zeros((n, t_len), dtype=np.float64)
    if override_valid_mask is None:
        cell_rows_to_compute = np.arange(n, dtype=np.int64)
    else:
        cell_rows_to_compute = np.flatnonzero(~override_valid_mask)
    current_masks_for_trace = current_masks[cell_rows_to_compute]
    input_video_all_finite = bool(np.all(np.isfinite(input_vid)))

    for iter_fr in range(t_len):
        frame_vec = np.asarray(input_vid[iter_fr, :, :], dtype=np.float64).reshape(-1)
        if not input_video_all_finite:
            frame_vec = np.nan_to_num(frame_vec, nan=0.0, posinf=0.0, neginf=0.0)

        if cell_rows_to_compute.size:
            value_fr = np.asarray(
                current_masks_for_trace.dot(frame_vec.reshape(-1, 1))
            ).ravel()
            traces_current[cell_rows_to_compute, iter_fr] = (
                value_fr / safe_areas_current[cell_rows_to_compute]
            )
        value_bg_fr = np.asarray(current_bg_masks.dot(frame_vec.reshape(-1, 1))).ravel()

        traces_bg_current[:, iter_fr] = value_bg_fr / safe_areas_bg_current

    # Force empty-mask/empty-ring rows to zero trace.
    traces_current[areas_current <= eps, :] = 0.0
    traces_bg_current[areas_bg_current <= eps, :] = 0.0
    if traces_cell_override is not None:
        traces_current = traces_current.copy()
        traces_current[override_valid_mask, :] = traces_cell_override[override_valid_mask, :]

    alpha_nnls = np.full(current_masks.shape[0], float(alpha_default), dtype=np.float64)

    if use_nnls:
        fallback_count = 0
        for i in range(n):
            y_i = traces_current[i, :].astype(np.float64)
            b_i = traces_bg_current[i, :].astype(np.float64)

            # Use fallback when the ring trace is degenerate.
            if (
                areas_bg_current[i] <= eps
                or np.linalg.norm(b_i) <= eps
                or not np.all(np.isfinite(y_i))
                or not np.all(np.isfinite(b_i))
            ):
                alpha_i = float(alpha_default)
                fallback_count += 1
            else:
                try:
                    alpha_i, _ = nnls(b_i.reshape(-1, 1), y_i)
                    alpha_i = float(alpha_i[0])
                except Exception:
                    alpha_i = float(alpha_default)
                    fallback_count += 1

            if alpha_min is not None:
                alpha_i = max(float(alpha_min), alpha_i)
            alpha_i = min(float(alpha_max), alpha_i)
            alpha_nnls[i] = alpha_i
    else:
        fallback_count = n

    traces_bg_rm = traces_current - alpha_nnls[:, None] * traces_bg_current

    traces_results = {
        'traces_cell': traces_current,
        'traces_bg': traces_bg_current,
        'traces_bg_rm': traces_bg_rm,
        'override_used': np.zeros((n,), dtype=bool) if override_valid_mask is None else override_valid_mask.astype(bool),
        'alpha_nnls': alpha_nnls,
        'alpha_method': 'nnls' if use_nnls else 'fixed_default',
        'alpha_default': float(alpha_default),
        'alpha_max': float(alpha_max),
        'alpha_min': None if alpha_min is None else float(alpha_min),
        'alpha_fallback_count': int(fallback_count),
    }

    return traces_results

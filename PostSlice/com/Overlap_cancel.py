import numpy as np
import scipy.sparse as sp
import time
from typing import List, Sequence, Tuple

from scipy.ndimage import gaussian_filter1d


def olcancel(masks_init: sp.dok_matrix):
    ''' 
    Cancel the overlapping area in each mask
    
    Inputs:
    -------
        **masks_init** dok_matrix NxP [int]: all masks in dok_matrix

    '''
    overlap_mat =  masks_init.dot(masks_init.T).toarray()
    overlap_mat = overlap_mat - np.diag(overlap_mat.diagonal())  # delete the diagonal numbers for cancelling the self-overlap

    olstatus = overlap_mat.sum(axis=1)  
    list_olmasks = olstatus.nonzero()[0]  # locate the masks which have overlap area with other masks
    
    num_olmasks = list_olmasks.shape[0]
    clear_container = sp.dok_matrix((num_olmasks, masks_init.shape[1]))  # container for storing positions to be cleared in each mask with overlap
    
    lab_cont = 0
    for iter_olmask in list_olmasks:  # Iteratively calculate the overlap area of each pair of overlap mask
        # if (not overlap_mat[iter_olmask,:].sum()>0):  # skip when the current mask has been calculated all overlap area
        #     lab_cont += 1
        #     continue
        list_currentol = overlap_mat[iter_olmask,:].nonzero()[0]  # masks has overlapping area with current mask
        for iter_currentol in list_currentol:
            clear_container[lab_cont,:] += masks_init[iter_olmask,:].multiply(masks_init[iter_currentol,:])
            # clear_container[iter_currentol,:] += masks_init[iter_olmask,:].multiply(masks_init[iter_currentol,:])
            # overlap_mat[iter_olmask, iter_currentol], overlap_mat[iter_currentol, iter_olmask] = 0, 0  # label the two masks in the overlap mat

        lab_cont += 1
    
    print(clear_container.toarray())

    lab_cont = 0
    for iter_olmask in list_olmasks:  # Iteratively clear the overlapping area of each mask
        clearposis = clear_container[lab_cont,:].nonzero()[1]
        print(clearposis)
        masks_init[iter_olmask, clearposis] = 0
        lab_cont += 1
    
    return


def cancelol_corr(raw_vid:np.ndarray, masks_added_oneinitmask:sp.dok_matrix, olratio_mat:np.ndarray, olratio_thresh:np.float64):
    # **raw_vid** ndarray HxWxT: raw video
    (H,W,T) = raw_vid.shape
    processed_labmat = np.zeros(olratio_mat.shape)  # matrix for labeling all processed mask pairs
    targ_posis = np.where((olratio_mat>0) & (olratio_mat<olratio_thresh))  # locate all mask pairs to be processed
    
    time_store_all = 0
    for iter_m1, iter_m2 in zip(targ_posis[0],targ_posis[1]):  # Iteratively process each pair of masks
        if (not (processed_labmat[iter_m1, iter_m2]==0 and processed_labmat[iter_m2, iter_m1]==0)): # skip when this pair has been processed
            continue
        mask1_mat = masks_added_oneinitmask[iter_m1,:].reshape((H,W)).toarray()
        mask2_mat = masks_added_oneinitmask[iter_m2,:].reshape((H,W)).toarray()
        posis1 = mask1_mat.nonzero()
        posis2 = mask2_mat.nonzero()
        
        overlap_mat = (masks_added_oneinitmask[iter_m1,:].multiply(masks_added_oneinitmask[iter_m2,:])).reshape((H,W)).toarray()  # locate the overlapping pixels among these two masks
        posis_ol = overlap_mat.nonzero()
        
        mask1_mat[posis_ol] = 0  # keep only the non-overlapping pixels
        mask2_mat[posis_ol] = 0

        meantrace1 = raw_vid[posis1[0], posis1[1],:].mean(axis=0)
        meantrace2 = raw_vid[posis2[0], posis2[1],:].mean(axis=0)

        overlap_traces = raw_vid[posis_ol[0], posis_ol[1]]

        corr_mat = np.corrcoef(np.vstack((meantrace1, meantrace2, overlap_traces)),rowvar=True)
        corr_status = corr_mat[2:, 0:2]  # Nx2 correlation status
        
        targs = corr_status.argmax(axis=1)  # target mask of each overlapping pixel

        mask1_mat[posis_ol[0][np.where(targs==0)[0]], posis_ol[1][np.where(targs==0)[0]]] = 1
        mask2_mat[posis_ol[0][np.where(targs==1)[0]], posis_ol[1][np.where(targs==1)[0]]] = 1
        
        mask1_dok = sp.dok_matrix(mask1_mat.reshape((1,-1)))
        mask2_dok = sp.dok_matrix(mask2_mat.reshape((1,-1)))

        time_store_st = time.time()   
        
        masks_added_oneinitmask[iter_m1,masks_added_oneinitmask[iter_m1,:].nonzero()[1]] = 0  
        masks_added_oneinitmask[iter_m2,masks_added_oneinitmask[iter_m2,:].nonzero()[1]] = 0
        masks_added_oneinitmask[iter_m1,mask1_dok.nonzero()[1]] = mask1_dok[mask1_dok.nonzero()]
        masks_added_oneinitmask[iter_m2,mask2_dok.nonzero()[1]] = mask2_dok[mask2_dok.nonzero()]
        ## this is the main cause of the long processing time interval, already fixed
        # masks_added_oneinitmask[iter_m1,:] = mask1_dok  # update the masks in the mask dok matrix
        # masks_added_oneinitmask[iter_m2,:] = mask2_dok
        time_store_all += time.time() - time_store_st

        processed_labmat[iter_m1, iter_m2],processed_labmat[iter_m2, iter_m1] = 1,1  # label the mask pair as processed
        
    # print(f'time spent for storing masks in the olcancel function is {time_store_all}')

    return


def _mask_flat(masks_final, j: int) -> np.ndarray:
    if hasattr(masks_final, "getrow"):
        row = masks_final.getrow(j)
        row = row.toarray() if hasattr(row, "toarray") else np.asarray(row)
    else:
        row = np.asarray(masks_final[j])
    return (np.asarray(row).ravel() > 0).astype(np.uint8)


def _trace_row(traces_final, j: int) -> np.ndarray:
    if traces_final is None:
        raise ValueError("traces_final is None; run demix / trace extraction first.")
    if hasattr(traces_final, "getrow"):
        tr = traces_final.getrow(j)
        tr = tr.toarray() if hasattr(tr, "toarray") else np.asarray(tr)
    else:
        tr = np.asarray(traces_final[j])
    return np.asarray(tr, dtype=np.float64).ravel()


def _ensure_sources(sources, n_masks: int) -> np.ndarray:
    if sources is None:
        return np.zeros((n_masks,), dtype=bool)
    arr = np.asarray(sources, dtype=bool).reshape(-1)
    if arr.shape[0] != n_masks:
        raise ValueError(f"sources length {arr.shape[0]} != masks rows {n_masks}.")
    return arr.copy()


def _overlap_pairs(
    masks_final,
    min_overlap_pixels: int = 1,
) -> List[Tuple[int, int]]:
    n_masks = int(masks_final.shape[0])
    if n_masks < 2:
        return []
    overlap_mat = masks_final.dot(masks_final.T)
    threshold = int(min_overlap_pixels)
    if sp.issparse(overlap_mat) and threshold > 0:
        upper = sp.triu(overlap_mat, k=1, format="coo")
        pairs = [
            (int(i), int(j))
            for i, j, value in zip(upper.row, upper.col, upper.data)
            if int(value) >= threshold
        ]
        pairs.sort()
        return pairs

    overlap_arr = overlap_mat.toarray() if sp.issparse(overlap_mat) else np.asarray(overlap_mat)
    rows, cols = np.nonzero(np.triu(overlap_arr >= threshold, k=1))
    return [(int(i), int(j)) for i, j in zip(rows, cols)]


def _prepare_infer_flat(infer_mask3d: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    infer_ = np.asarray(infer_mask3d)
    if infer_.ndim != 3:
        raise ValueError("infer_mask3d must have shape (T, H, W)")
    infer_bin = (infer_ == 1).astype(np.uint8)
    t_inf, h_inf, w_inf = infer_bin.shape
    n_pix = h_inf * w_inf
    flat_inf = infer_bin.reshape(t_inf, n_pix)
    return flat_inf, t_inf, h_inf, w_inf


def _frac_infer_in_mask(
    masks_final,
    j_mask: int,
    flat_inf: np.ndarray,
    n_pix: int,
    h: int,
    w: int,
) -> np.ndarray:
    mflat = _mask_flat(masks_final, j_mask).astype(np.uint8).ravel()
    if mflat.size != n_pix:
        mflat = mflat.reshape(h, w).ravel()
        if mflat.size != n_pix:
            raise ValueError(
                f"flattened mask length {mflat.size} does not match infer_mask3d spatial pixel count {n_pix}"
            )
    area = int(mflat.sum())
    if area == 0:
        raise ValueError(f"masks_final[{j_mask}] has zero area.")
    overlap = (flat_inf * mflat.reshape(1, -1)).sum(axis=1).astype(np.float64)
    return overlap / float(area)


def _cached_frac_infer_in_mask(
    j_mask: int,
    flat_inf: np.ndarray,
    n_pix: int,
    h: int,
    w: int,
    mask_flats: dict[int, np.ndarray],
    frac_cache: dict[int, np.ndarray],
) -> np.ndarray:
    cached = frac_cache.get(j_mask)
    if cached is not None:
        return cached
    mflat = mask_flats[j_mask]
    if mflat.size != n_pix:
        mflat = mflat.reshape(h, w).ravel()
        if mflat.size != n_pix:
            raise ValueError(
                f"flattened mask length {mflat.size} does not match infer_mask3d spatial pixel count {n_pix}"
            )
    area = int(mflat.sum())
    if area == 0:
        raise ValueError(f"masks_final[{j_mask}] has zero area.")
    pixel_indices = np.flatnonzero(mflat)
    overlap = flat_inf[:, pixel_indices].sum(axis=1).astype(np.float64)
    fraction = overlap / float(area)
    frac_cache[j_mask] = fraction
    return fraction


def _region_mean_trace(
    vid3d: np.ndarray,
    region_hw: np.ndarray,
    *,
    all_finite: bool = None,
) -> Tuple[np.ndarray, int]:
    region_hw = np.asarray(region_hw, dtype=bool)
    n = int(region_hw.sum())
    if n == 0:
        return np.zeros(vid3d.shape[0], dtype=np.float64), 0
    if all_finite is None:
        all_finite = bool(np.all(np.isfinite(vid3d)))
    if all_finite:
        selected = np.asarray(vid3d).reshape(vid3d.shape[0], -1)[:, region_hw.ravel()]
        return selected.sum(axis=1) / float(n), n
    w = region_hw.astype(np.float64)
    return (vid3d * w).sum(axis=(1, 2)) / float(n), n


def _set_trace_rows(
    traces_final,
    row_updates: Sequence[Tuple[int, np.ndarray]],
):
    if not row_updates:
        return traces_final
    if hasattr(traces_final, "tolil") and sp.issparse(traces_final):
        m = traces_final.tolil()
        ncols = int(m.shape[1])
        for j, vec in row_updates:
            out = np.asarray(vec, dtype=np.float32).ravel()
            if out.size < ncols:
                out = np.concatenate([out, np.zeros(ncols - out.size, dtype=np.float32)])
            elif out.size > ncols:
                out = out[:ncols]
            m[j, :] = out
        return m.tocsr()
    tf = np.asarray(traces_final)
    if tf.ndim != 2:
        raise ValueError(f"traces_final must be 2D; received shape={tf.shape}")
    if not tf.flags.writeable:
        tf = tf.copy()
    for j, vec in row_updates:
        out = np.asarray(vec, dtype=np.float32).ravel()
        n = min(tf.shape[1], out.size)
        tf[j, :n] = out[:n]
    return tf


def _copy_traces_final(traces_final):
    if hasattr(traces_final, "tocsr") and sp.issparse(traces_final):
        return traces_final.tocsr().copy()
    return np.asarray(traces_final, dtype=np.float32).copy()


def _amplitude_stat(vec: np.ndarray, method: str, eps: float) -> float:
    x = np.asarray(vec, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    method = str(method).lower()
    if method == "rms":
        return float(np.sqrt(np.mean(x * x)))
    if method == "median":
        return float(np.median(np.abs(x)))
    if method == "mean":
        return float(np.mean(np.abs(x)))
    raise ValueError(f"amplitude_method must be 'rms' | 'median' | 'mean'; received {method!r}")


def _align_trace_amplitude(
    t_new: np.ndarray,
    t_ref: np.ndarray,
    active_mask: np.ndarray = None,
    method: str = "rms",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Scale t_new so its amplitude statistic on active frames matches t_ref.
    Preserves temporal shape; only adjusts global scale.
    """
    t_new = np.asarray(t_new, dtype=np.float64).ravel()
    t_ref = np.asarray(t_ref, dtype=np.float64).ravel()
    n = int(min(t_new.size, t_ref.size))
    if n <= 0:
        return np.asarray(t_new, dtype=np.float32).ravel()

    t_new = t_new[:n]
    t_ref = t_ref[:n]
    out = t_new.astype(np.float32, copy=True)

    if active_mask is not None:
        am = np.asarray(active_mask, dtype=bool).ravel()[:n]
        if int(am.sum()) > 0:
            use = am
        else:
            use = np.ones(n, dtype=bool)
    else:
        use = np.ones(n, dtype=bool)

    s_ref = _amplitude_stat(t_ref[use], method, eps)
    s_new = _amplitude_stat(t_new[use], method, eps)
    if s_ref <= eps or s_new <= eps:
        return out

    scale = s_ref / (s_new + eps)
    out[:n] = (t_new * scale).astype(np.float32)
    return out


def _modulate_overlap_pair(
    masks_final,
    traces_final,
    j0: int,
    j1: int,
    flat_inf: np.ndarray,
    n_pix: int,
    h: int,
    w: int,
    vid: np.ndarray,
    sigma_frames: float,
    strength: float,
    ov_cap: float,
    mask_flats: dict[int, np.ndarray],
    frac_cache: dict[int, np.ndarray],
    video_all_finite: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    f0 = _cached_frac_infer_in_mask(
        j0, flat_inf, n_pix, h, w, mask_flats, frac_cache
    )
    f1 = _cached_frac_infer_in_mask(
        j1, flat_inf, n_pix, h, w, mask_flats, frac_cache
    )
    if f0.size != f1.size:
        raise ValueError(f"the two mask-fraction sequences have different frame counts: {f0.size} vs {f1.size}")

    diff = (f0 - f1).astype(np.float64)
    m0 = mask_flats[j0].reshape(h, w) > 0
    m1 = mask_flats[j1].reshape(h, w) > 0
    overlap_hw = m0 & m1

    tr_overlap, n_ov = _region_mean_trace(
        vid, overlap_hw, all_finite=video_all_finite
    )
    if n_ov == 0:
        raise ValueError(f"masks_final[{j0}] and [{j1}] have no overlapping pixels.")

    t0_full = _trace_row(traces_final, j0)
    t1_full = _trace_row(traces_final, j1)
    t_use = int(min(diff.size, tr_overlap.size, t0_full.size, t1_full.size))
    if t_use <= 0:
        raise ValueError("No common frame length is available.")

    diff = diff[:t_use]
    tr_overlap = np.asarray(tr_overlap[:t_use], dtype=np.float64)
    t0 = np.asarray(t0_full[:t_use], dtype=np.float64).copy()
    t1 = np.asarray(t1_full[:t_use], dtype=np.float64).copy()

    diff_s = gaussian_filter1d(diff, sigma=float(sigma_frames), mode="nearest")
    mag = np.abs(diff_s)
    rel = mag / (float(np.nanmax(mag)) + 1e-12)
    rel = np.clip(rel, 0.0, 1.0)

    den = float(np.nanmedian(np.abs(tr_overlap))) + 1e-12
    ov_scale = np.clip(np.abs(tr_overlap) / den, 0.0, float(ov_cap))

    g = np.clip(float(strength) * rel * ov_scale, 0.0, 0.95)
    sign = np.sign(diff_s)
    mask_nz = sign != 0
    pos = np.maximum(sign, 0.0)
    neg = np.maximum(-sign, 0.0)

    t0_new = np.where(mask_nz, t0 * (1.0 + g * pos - g * neg), t0)
    t1_new = np.where(mask_nz, t1 * (1.0 + g * neg - g * pos), t1)

    t0_out = np.asarray(t0_full, dtype=np.float32).ravel().copy()
    t1_out = np.asarray(t1_full, dtype=np.float32).ravel().copy()
    t0_out[:t_use] = t0_new.astype(np.float32)
    t1_out[:t_use] = t1_new.astype(np.float32)
    return t0_out, t1_out, t_use


def fact_guided_overlap_demixing(
    masks_final,
    traces_final,
    sources,
    infer_mask3d: np.ndarray,
    input_img: np.ndarray,
    mask_shape_2d: Tuple[int, int],
    sigma_frames: float = 5.0,
    strength: float = 0.35,
    ov_cap: float = 5.0,
    min_overlap_pixels: int = 1,
    preserve_amplitude: bool = True,
    amplitude_method: str = "rms",
):
    """
    FACT-guided overlap demixing: modulate traces for all spatially overlapping mask pairs.

    Uses per-frame infer_mask3d footprint fractions and overlap-region raw-movie mean
    traces to enhance the higher-fraction trace and attenuate the lower-fraction trace.

    When preserve_amplitude is True, each overlap-optimized trace is scaled after
    modulation so its amplitude (rms/median/mean) matches the pre-modulation trace.

    Returns masks_final, traces_final, sources (sources[i]=True if overlap-optimized).
    """
    if traces_final is None:
        raise ValueError("traces_final is None; run demix / trace extraction first.")

    n_masks = int(masks_final.shape[0])
    sources = _ensure_sources(sources, n_masks)
    overlap_optimized = np.zeros((n_masks,), dtype=bool)
    traces_ref = _copy_traces_final(traces_final) if preserve_amplitude else None

    h, w = int(mask_shape_2d[0]), int(mask_shape_2d[1])
    flat_inf, _, h_inf, w_inf = _prepare_infer_flat(infer_mask3d)
    if (h, w) != (h_inf, w_inf):
        raise ValueError(
            f"mask_shape_2d {mask_shape_2d} does not match infer_mask3d spatial dimensions ({h_inf}, {w_inf})"
        )
    n_pix = h_inf * w_inf

    vid = np.asarray(input_img, dtype=np.float64)
    if vid.ndim != 3:
        raise ValueError("input_img must have shape (T, H, W)")
    video_all_finite = bool(np.all(np.isfinite(vid)))

    pairs = _overlap_pairs(masks_final, min_overlap_pixels=min_overlap_pixels)
    pair_mask_indices = sorted({j for pair in pairs for j in pair})
    mask_flats = {j: _mask_flat(masks_final, j) for j in pair_mask_indices}
    frac_cache: dict[int, np.ndarray] = {}

    for j0, j1 in pairs:
        try:
            t0_out, t1_out, _ = _modulate_overlap_pair(
                masks_final,
                traces_final,
                j0,
                j1,
                flat_inf,
                n_pix,
                h,
                w,
                vid,
                sigma_frames=sigma_frames,
                strength=strength,
                ov_cap=ov_cap,
                mask_flats=mask_flats,
                frac_cache=frac_cache,
                video_all_finite=video_all_finite,
            )
        except ValueError:
            continue
        traces_final = _set_trace_rows(
            traces_final,
            [(j0, t0_out), (j1, t1_out)],
        )
        overlap_optimized[j0] = True
        overlap_optimized[j1] = True

    if preserve_amplitude and traces_ref is not None:
        row_updates: List[Tuple[int, np.ndarray]] = []
        for j in np.where(overlap_optimized)[0]:
            j = int(j)
            t_new = _trace_row(traces_final, j)
            t_ref = _trace_row(traces_ref, j)
            t_aligned = _align_trace_amplitude(
                t_new,
                t_ref,
                active_mask=None,
                method=amplitude_method,
            )
            row_updates.append((j, t_aligned))
        traces_final = _set_trace_rows(traces_final, row_updates)

    sources = np.logical_or(sources, overlap_optimized)

    masks_final = (masks_final.tocsr() > 0).astype(np.int64)
    return masks_final, traces_final, sources

import os
import cv2
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from enum import Enum

from numpy.linalg import norm
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    center_of_mass,
    gaussian_filter,
    label,
    maximum_filter,
    uniform_filter,
)
from scipy.ndimage import zoom as ndi_zoom

# Eight-neighborhood connectivity, including diagonals.
_CC_STRUCT8 = np.ones((3, 3), dtype=int)


def _diag(verbose, level, message):
    """Print diagnostics only when ``verbose >= level``; do not use UserWarning."""
    if int(verbose) >= int(level):
        print(message)


__all__ = [
    "DualModalityUnmixer",
    "run_fact_guided_demix",
    "fact_guided_demixing",
]

# ---------------------------------------------------------------------------
# Port of suite2p detection/sparsedetect logic, adapted from suite2p source while
# preserving the default sparse_mode flow. ``nmasks`` is undefined upstream, so it
# is fixed at 0 here to match the else branch.
# ---------------------------------------------------------------------------


def _gaussian_temporal_high_pass(mov, width):
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter(mov[:, j, :], [width, 0])
    return mov


def _block_mean_temporal_high_pass(mov, width):
    mov = mov.copy()
    for i in range(0, mov.shape[0], width):
        mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov


def _temporal_high_pass_filter(mov, width):
    width = int(width)
    return _gaussian_temporal_high_pass(mov, width) if width < 10 else _block_mean_temporal_high_pass(
        mov, width)


def _temporal_difference_rms(mov, batch_size):
    nbins, Ly, Lx = mov.shape
    batch_size = min(int(batch_size), nbins)
    sdmov = np.zeros((Ly, Lx), np.float32)
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix + batch_size, :, :], axis=0)**2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov


def _spatial_downsample_by_two(mov, taper_edge=True):
    n_frames, Ly, Lx = mov.shape
    movd = np.zeros((n_frames, int(np.ceil(Ly / 2)), Lx), np.float32)
    movd[:, :Ly // 2, :] = np.mean([mov[:, 0:-1:2, :], mov[:, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:, -1, :] = mov[:, -1, :] / 2 if taper_edge else mov[:, -1, :]
    mov2 = np.zeros(
        (n_frames, int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), np.float32)
    mov2[:, :, :Lx // 2] = np.mean([movd[:, :, 0:-1:2], movd[:, :, 1::2]], axis=0)
    if Lx % 2 == 1:
        mov2[:, :, -1] = movd[:, :, -1] / 2 if taper_edge else movd[:, :, -1]
    return mov2


def _thresholded_energy_map(mov, intensity_threshold):
    nbinned, Lyp, Lxp = mov.shape
    Vt = np.zeros((Lyp, Lxp), np.float32)
    thr = float(intensity_threshold)
    for t in range(nbinned):
        mt = mov[t]
        Vt += mt**2 * (mt > thr)
    return Vt**.5


def _subtract_local_spatial_background(mov, filter_size):
    nbinned, Ly, Lx = mov.shape
    c1 = uniform_filter(np.ones((Ly, Lx)), size=filter_size, mode="constant")
    movt = np.zeros_like(mov)
    for frame, framet in zip(mov, movt):
        framet[:] = frame - (uniform_filter(frame, size=filter_size, mode="constant") /
                             c1)
    return movt


def _square_filter_movie(mov, filter_size):
    movt = np.zeros_like(mov, dtype=np.float32)
    for frame, framet in zip(mov, movt):
        framet[:] = filter_size * uniform_filter(frame, size=filter_size,
                                                   mode="constant")
    return movt


def _extend_roi_cardinal(ypix, xpix, Ly, Lx, niter=1):
    for _ in range(niter):
        yx = ((ypix, ypix, ypix, ypix - 1, ypix + 1), (xpix, xpix + 1, xpix - 1, xpix,
                                                       xpix))
        yx = np.array(yx)
        yx = yx.reshape((2, -1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix


def _grow_roi_from_active_frames(ypix, xpix, mov, Lyc, Lxc, active_frames):
    npix = 0
    it = 0
    sgn = 1.0
    while npix < 10000:
        npix = ypix.size
        ypix, xpix = _extend_roi_cardinal(ypix, xpix, Lyc, Lxc, 1)
        usub = mov[np.ix_(active_frames, ypix * Lxc + xpix)]
        lam = np.mean(usub, axis=0)
        ix = lam > max(0, lam.max() / 5.0)
        if ix.sum() == 0:
            break
        ypix, xpix, lam = ypix[ix], xpix[ix], lam[ix]
        if it == 0:
            sgn = 1.0
        if np.sign(sgn * (ix.sum() - npix)) <= 0:
            break
        npix = ypix.size
        it += 1
    lam = lam / np.sum(lam**2)**.5
    return ypix, xpix, lam


def _initialize_square_roi(yi, xi, lx, Ly, Lx):
    lhf = int((lx - 1) / 2)
    ipix = np.tile(np.arange(-lhf, -lhf + lx, dtype=np.int32), reps=(lx, 1))
    x0 = xi + ipix
    y0 = yi + ipix.T
    mask = np.ones_like(ipix, dtype=np.float32)
    ix = np.all((y0 >= 0, y0 < Ly, x0 >= 0, x0 < Lx), axis=0)
    x0 = x0[ix]
    y0 = y0[ix]
    mask = mask[ix]
    mask = mask / norm(mask)
    return y0.flatten(), x0.flatten(), mask.flatten()


def _evaluate_two_component_split(mpix0, lam, Th2):
    mpix = mpix0.copy()
    xproj = mpix @ lam
    gf0 = xproj > Th2
    mpix[gf0, :] -= np.outer(xproj[gf0], lam)
    vexp0 = np.sum(mpix0**2) - np.sum(mpix**2)
    k = np.argmax(np.sum(mpix * np.float32(mpix > 0), axis=1))
    mu = [lam * np.float32(mpix[k] < 0), lam * np.float32(mpix[k] > 0)]
    mpix = mpix0.copy()
    goodframe = []
    xproj = []
    for mu0 in mu:
        mu0[:] /= norm(mu0) + 1e-6
        xp = mpix @ mu0
        mpix[gf0, :] -= np.outer(xp[gf0], mu0)
        goodframe.append(gf0)
        xproj.append(xp[gf0])
    flag = [False, False]
    V = np.zeros(2)
    for _ in range(3):
        for k in range(2):
            if flag[k]:
                continue
            mpix[goodframe[k], :] += np.outer(xproj[k], mu[k])
            xp = mpix @ mu[k]
            goodframe[k] = xp > Th2
            V[k] = np.sum(xp**2)
            if np.sum(goodframe[k]) == 0:
                flag[k] = True
                V[k] = -1
                continue
            xproj[k] = xp[goodframe[k]]
            mu[k] = np.mean(mpix[goodframe[k], :] * xproj[k][:, np.newaxis], axis=0)
            mu[k][mu[k] < 0] = 0
            mu[k] /= (1e-6 + np.sum(mu[k]**2)**.5)
            mpix[goodframe[k], :] -= np.outer(xproj[k], mu[k])
    k = np.argmax(V)
    vexp = np.sum(mpix0**2) - np.sum(mpix**2)
    vrat = vexp / (vexp0 + 1e-10)
    return vrat, (mu[k], xproj[k], goodframe[k])


def _extend_weighted_mask(ypix, xpix, lam, Ly, Lx):
    nel = len(xpix)
    yx = ((ypix, ypix, ypix, ypix - 1, ypix - 1, ypix - 1, ypix + 1, ypix + 1,
           ypix + 1), (xpix, xpix + 1, xpix - 1, xpix, xpix + 1, xpix - 1, xpix,
                       xpix + 1, xpix - 1))
    yx = np.array(yx)
    yx = yx.reshape((2, -1))
    yu, ind = np.unique(yx, axis=1, return_inverse=True)
    LAM = np.zeros(yu.shape[1])
    for j in range(len(ind)):
        LAM[ind[j]] += lam[j % nel] / 3
    ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
    ypix1, xpix1 = yu[:, ix]
    lam1 = LAM[ix]
    return ypix1, xpix1, lam1


def _build_multiscale_weighted_masks(ypix0, xpix0, lam0, Lyp, Lxp):
    xs = [xpix0]
    ys = [ypix0]
    lms = [lam0]
    for j in range(1, len(Lyp)):
        ipix, ind = np.unique(
            np.int32(xs[j - 1] / 2) + np.int32(ys[j - 1] / 2) * Lxp[j],
            return_inverse=True)
        LAM = np.zeros(len(ipix))
        for i in range(len(xs[j - 1])):
            LAM[ind[i]] += lms[j - 1][i] / 2
        lms.append(LAM)
        ys.append(np.int32(ipix / Lxp[j]))
        xs.append(np.int32(ipix % Lxp[j]))
    for j in range(len(Lyp)):
        ys[j], xs[j], lms[j] = _extend_weighted_mask(ys[j], xs[j], lms[j], Lyp[j], Lxp[j])
    return ys, xs, lms


class _SpatialScaleMode(Enum):
    Forced = "FORCED"
    Estimated = "estimated"


def _estimate_spatial_scale(I):
    I0 = I.max(axis=0)
    imap = np.argmax(I, axis=0).flatten()
    ipk = np.abs(I0 - maximum_filter(I0, size=(11, 11))).flatten() < 1e-4
    isort = np.argsort(I0.flatten()[ipk])[::-1]
    if isort.size == 0:
        return np.int32(0)
    sub = imap[ipk][isort[:50]]
    if sub.size == 0:
        return np.int32(0)
    vals, counts = np.unique(sub, return_counts=True)
    return np.int32(vals[np.argmax(counts)])


def _resolve_spatial_scale(I, spatial_scale, verbose=0):
    if spatial_scale > 0:
        return max(1, min(4, int(spatial_scale))), _SpatialScaleMode.Forced
    scale = int(_estimate_spatial_scale(I))
    if scale > 0:
        return scale, _SpatialScaleMode.Estimated
    _diag(
        verbose,
        2,
        "Spatial scale estimation failed. Setting spatial scale to 1 in order to continue.",
    )
    return 1, _SpatialScaleMode.Forced


def _temporal_bin_movie(mov_tb, bin_size, badframes=None):
    """Single-batch streaming temporal binning equivalent to suite2p.detection.detect.bin_movie."""
    n_frames = mov_tb.shape[0]
    good_frames = ~badframes if badframes is not None else np.ones(n_frames, dtype=bool)
    batch_size = min(int(good_frames.sum()), 500)
    Ly, Lx = mov_tb.shape[1], mov_tb.shape[2]
    bin_size = max(1, min(int(bin_size), n_frames))
    num_binned_frames = max(1, n_frames // bin_size)
    mov = np.zeros((num_binned_frames, Ly, Lx), np.float32)
    curr_bin_number = 0
    for k in np.arange(0, n_frames, batch_size):
        data = mov_tb[k:min(k + batch_size, n_frames)].astype(np.float32)
        good_indices = good_frames[k:min(k + batch_size, n_frames)]
        if good_indices.mean() > 0.5:
            data = data[good_indices]
        if data.shape[0] == 0:
            continue
        bs = int(bin_size)
        if data.shape[0] > bs:
            n_d = data.shape[0]
            data = data[:(n_d // bs) * bs]
            data = data.reshape(-1, bs, Ly, Lx).mean(axis=1)
        else:
            data = data.mean(axis=0)[np.newaxis, :, :]
        if mov.shape[0] > curr_bin_number:
            n_bins = data.shape[0]
            end = min(curr_bin_number + n_bins, mov.shape[0])
            n_take = end - curr_bin_number
            if n_take > 0:
                mov[curr_bin_number:end] = data[:n_take]
            curr_bin_number = end
    return mov[:curr_bin_number]


def _resize_prior_to_shape(prior_hw, out_h, out_w):
    """Bilinearly resize a spatial prior map to align it with the V1 map."""
    if prior_hw.shape == (out_h, out_w):
        return prior_hw.astype(np.float32)
    zh = out_h / float(prior_hw.shape[0])
    zw = out_w / float(prior_hw.shape[1])
    z = ndi_zoom(prior_hw.astype(np.float32), (zh, zw), order=1)
    if z.shape[0] != out_h or z.shape[1] != out_w:
        z = cv2.resize(
            z, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    p = z - float(z.min())
    m = float(p.max()) + 1e-10
    return (p / m).astype(np.float32)


def _find_multiscale_seed_maxima(V1, gxy, Lyp, Lxp, Lyc, Lxc, seed_valid_mask):
    """
    Select maxima over V1 at each scale, matching suite2p. When ``seed_valid_mask``
    is a Boolean ``(Ly, Lx)`` array, only V1 responses at permitted locations enter
    argmax, constraining peak and growth seeds to the eroded parent mask.
    """
    nscales = len(V1)
    v0max = np.zeros(nscales, dtype=np.float32)
    imax_lin = np.zeros(nscales, dtype=np.int64)
    for j in range(nscales):
        Vj = V1[j]
        if seed_valid_mask is not None:
            gy = np.clip(gxy[j][1].astype(np.int32), 0, Lyc - 1)
            gx = np.clip(gxy[j][0].astype(np.int32), 0, Lxc - 1)
            vmsk = Vj * seed_valid_mask[gy, gx].astype(np.float32)
        else:
            vmsk = Vj
        flat = vmsk.ravel()
        imax_lin[j] = int(np.argmax(flat))
        v0max[j] = float(flat[imax_lin[j]])
    return v0max, imax_lin


def _detect_sparse_rois(
    mov,
    *,
    high_pass,
    neuropil_high_pass,
    batch_size,
    spatial_scale,
    threshold_scaling,
    max_iterations,
    percentile=0.0,
    prior_peak_boost=0.0,
    prior_map_hw=None,
    seed_valid_mask=None,
    verbose=0,
):
    """
    Main loop equivalent to suite2p sparsedetect.sparsery.
    When ``prior_peak_boost > 0`` and ``prior_map_hw`` is not None, non-negative
    prior weighting enhances V1 maps before iteration; this is not part of the
    upstream suite2p implementation. ``prior_peak_boost == 0`` leaves V1 unchanged.
    When ``seed_valid_mask`` is a Boolean ``(Ly, Lx)`` map, each iteration considers
    V1 responses only where it is True (all other positions are zero), preventing
    seeds from starting in background near the parent-mask boundary.
    """
    mov = np.asarray(mov, dtype=np.float32).copy()
    mean_img = mov.mean(axis=0)
    mov = _temporal_high_pass_filter(mov, int(high_pass))
    max_proj = mov.max(axis=0)
    sdmov = _temporal_difference_rms(mov, batch_size=batch_size)
    mov = _subtract_local_spatial_background(mov=mov / sdmov, filter_size=int(neuropil_high_pass))

    _, Lyc, Lxc = mov.shape
    LL = np.meshgrid(np.arange(Lxc), np.arange(Lyc))
    gxy = [np.array(LL).astype(np.float32)]
    dmov = mov
    movu = []
    Lyp, Lxp = np.zeros(5, np.int32), np.zeros(5, np.int32)
    for j in range(5):
        movu0 = _square_filter_movie(dmov, 3)
        dmov = 2 * _spatial_downsample_by_two(dmov)
        gxy0 = _spatial_downsample_by_two(gxy[j], False)
        gxy.append(gxy0)
        _, Lyp[j], Lxp[j] = movu0.shape
        movu.append(movu0)

    I = np.zeros((len(gxy), gxy[0].shape[1], gxy[0].shape[2]), np.float32)
    ly0, lx0 = gxy[0].shape[1], gxy[0].shape[2]
    for movu0, gxy0, I0 in zip(movu, gxy, I):
        kx = min(3, gxy0.shape[1] - 1)
        ky = min(3, gxy0.shape[2] - 1)
        z = movu0.max(axis=0)
        if (
            kx >= 1
            and ky >= 1
            and gxy0.shape[1] > kx
            and gxy0.shape[2] > ky
        ):
            gmodel = RectBivariateSpline(
                gxy0[1, :, 0], gxy0[0, 0, :], z, kx=kx, ky=ky)
            I0[:] = gmodel(gxy[0][1, :, 0], gxy[0][0, 0, :])
        else:
            I0[:] = ndi_zoom(
                z.astype(np.float64),
                (ly0 / z.shape[0], lx0 / z.shape[1]),
                order=1,
            ).astype(np.float32)
    v_corr = I.max(axis=0)

    scale, estimate_mode = _resolve_spatial_scale(
        I=I, spatial_scale=int(spatial_scale), verbose=verbose
    )
    spatscale_pix = 3 * 2**scale
    Th2 = float(threshold_scaling) * 5 * max(1, scale)
    vmultiplier = max(1, mov.shape[0] / 1200.0)
    if verbose >= 1:
        print(
            "NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f"
            % (estimate_mode.value, spatscale_pix, vmultiplier, vmultiplier * Th2))

    v_map = [_thresholded_energy_map(movu0, Th2) for movu0 in movu]
    movu = [movu0.reshape(movu0.shape[0], -1) for movu0 in movu]

    V1 = deepcopy(v_map)
    if prior_peak_boost > 0.0 and prior_map_hw is not None:
        w = float(prior_peak_boost)
        for j in range(5):
            pj = _resize_prior_to_shape(prior_map_hw, int(Lyp[j]), int(Lxp[j]))
            V1[j] = V1[j] * (1.0 + w * pj)

    mov = np.reshape(mov, (-1, Lyc * Lxc))
    lxs = 3 * 2**np.arange(5)
    nscales = len(lxs)
    nmasks = 0

    v_max = np.zeros(max_iterations)
    ihop = np.zeros(max_iterations)
    v_split = np.zeros(max_iterations)
    stats = []
    for tj in range(max_iterations):
        v0max, imax_lin = _find_multiscale_seed_maxima(
            V1, gxy, Lyp, Lxp, Lyc, Lxc, seed_valid_mask)
        imap = int(np.argmax(v0max))
        imax = int(imax_lin[imap])
        yi, xi = np.unravel_index(imax, (Lyp[imap], Lxp[imap]))
        yi, xi = float(gxy[imap][1, yi, xi]), float(gxy[imap][0, yi, xi])
        med = [int(yi), int(xi)]

        v_max[tj] = float(v0max.max())
        if v_max[tj] < vmultiplier * Th2:
            break
        ls = int(lxs[imap])
        ihop[tj] = imap

        yi, xi = int(yi), int(xi)
        ypix0, xpix0, lam0 = _initialize_square_roi(yi, xi, ls, Lyc, Lxc)

        tproj = (mov[:, ypix0 * Lxc + xpix0] * lam0[0]).sum(axis=-1)
        if percentile > 0:
            threshold = min(Th2, float(np.percentile(tproj, percentile)))
        else:
            threshold = Th2
        active_frames = np.nonzero(tproj > threshold)[0]

        for _ in range(3):
            ypix0, xpix0, lam0 = _grow_roi_from_active_frames(
                ypix0, xpix0, mov, Lyc, Lxc, active_frames)
            tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
            active_frames = np.nonzero(tproj > threshold)[0]
            if len(active_frames) < 1:
                if tj < nmasks:
                    continue
                break
        if len(active_frames) < 1:
            if tj < nmasks:
                continue
            break

        v_split[tj], ipack = _evaluate_two_component_split(mov[:, ypix0 * Lxc + xpix0], lam0, threshold)
        if v_split[tj] > 1.25:
            lam0, xp, active_frames = ipack
            tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
            tproj[active_frames] = xp
            ix = lam0 > lam0.max() / 5.0
            xpix0 = xpix0[ix]
            ypix0 = ypix0[ix]
            lam0 = lam0[ix]
            ymed = np.median(ypix0)
            xmed = np.median(xpix0)
            imin = np.argmin((xpix0 - xmed)**2 + (ypix0 - ymed)**2)
            med = [ypix0[imin], xpix0[imin]]

        mov[np.ix_(active_frames, ypix0 * Lxc + xpix0)] -= tproj[active_frames][:, np.newaxis] * lam0
        ys, xs, lms = _build_multiscale_weighted_masks(ypix0, xpix0, lam0, Lyp, Lxp)
        thr = float(threshold)
        for j in range(nscales):
            movu[j][np.ix_(active_frames, xs[j] + Lxp[j] * ys[j])] -= np.outer(
                tproj[active_frames], lms[j])
            Mx = movu[j][:, xs[j] + Lxp[j] * ys[j]]
            V1[j][ys[j], xs[j]] = (Mx**2 * np.float32(Mx > thr)).sum(axis=0)**.5

        stats.append({
            "ypix": ypix0.astype(int),
            "xpix": xpix0.astype(int),
            "lam": lam0 * sdmov[ypix0, xpix0],
            "med": med,
            "footprint": ihop[tj],
        })

        if verbose >= 1 and tj % 1000 == 0:
            print("%d ROIs, score=%2.2f" % (tj, v_max[tj]))

    new_ops = {
        "mean_img": mean_img,
        "max_proj": max_proj,
        "Vmax": v_max,
        "ihop": ihop,
        "Vsplit": v_split,
        "Vcorr": v_corr,
        "spatscale_pix": spatscale_pix,
    }
    return new_ops, stats


def _pairwise_com_distance_and_iou_matrices(masks_csr, H, W):
    """
    Compute pairwise center-of-mass Euclidean distances (pixels) and the IoU matrix
    for multiple flattened CSR binary masks produced by demixing.

    Returns
    ----
    dist_mx, iou_mx, com_yx : (n,n), (n,n), (n,2)
        Each ``com_yx`` row is ``(y, x)``. Empty masks have a ``nan`` center of mass
        and corresponding ``nan`` distances.
    """
    n = int(masks_csr.shape[0])
    if n == 0:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        )
    flat_bin = []
    com_yx = np.full((n, 2), np.nan, dtype=np.float64)
    for i in range(n):
        m2 = masks_csr.getrow(i).toarray().reshape(H, W) > 0
        flat_bin.append(m2)
        if np.any(m2):
            cy, cx = center_of_mass(m2.astype(np.float64))
            com_yx[i, 0] = cy
            com_yx[i, 1] = cx

    dist_mx = np.zeros((n, n), dtype=np.float64)
    iou_mx = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = flat_bin[i], flat_bin[j]
            inter = np.count_nonzero(np.logical_and(a, b))
            uni = np.count_nonzero(np.logical_or(a, b))
            ij = float(inter) / float(uni) if uni > 0 else 0.0
            iou_mx[i, j] = ij
            iou_mx[j, i] = ij
            if np.any(np.isnan(com_yx[i])) or np.any(np.isnan(com_yx[j])):
                d_ij = np.nan
            else:
                d_ij = float(np.linalg.norm(com_yx[i] - com_yx[j]))
            dist_mx[i, j] = d_ij
            dist_mx[j, i] = d_ij
    return dist_mx, iou_mx, com_yx


def _split_cc_masks_ge_min_area(mask_bin, min_area, structure=_CC_STRUCT8):
    """
    Label a binary image with 8-neighborhood connectivity (or the supplied
    ``structure``), remove connected components smaller than ``min_area``, and
    return one binary image for every remaining component.
    """
    labeled, num = label(mask_bin, structure=structure)
    keep = np.zeros_like(mask_bin, dtype=bool)
    for lab in range(1, num + 1):
        sel = labeled == lab
        if np.sum(sel) >= min_area:
            keep[sel] = True
    if not np.any(keep):
        return []
    labeled2, num2 = label(keep, structure=structure)
    out = []
    for lab in range(1, num2 + 1):
        cc = (labeled2 == lab).astype(np.int64)
        if np.sum(cc) >= min_area:
            out.append(cc)
    return out


def _fill_holes_binary_mask(mask_bin):
    """Fill holes fully enclosed by foreground in a binary sub-mask."""
    m = np.asarray(mask_bin).astype(bool)
    if not np.any(m):
        return np.zeros_like(mask_bin, dtype=np.int64)
    return binary_fill_holes(m).astype(np.int64)


def _safe_fit_ellipse_params(contour):
    """Return finite OpenCV ellipse parameters, or ``None`` for an invalid fit."""
    cnt = np.asarray(contour)
    if len(cnt) < 5:
        return None
    try:
        (cx, cy), (majax, minax), ang = cv2.fitEllipse(cnt)
    except cv2.error:
        return None

    params = np.asarray([cx, cy, majax, minax, ang], dtype=np.float64)
    if not np.all(np.isfinite(params)):
        return None
    if float(majax) <= 0.0 or float(minax) <= 0.0:
        return None

    # cv2.ellipse receives integer center/axis values through the Python binding.
    # Reject numerically unstable fits before they can overflow that interface.
    cv_int_max = float(np.iinfo(np.int32).max)
    if max(abs(float(cx)), abs(float(cy)), float(majax), float(minax)) > cv_int_max:
        return None
    return float(cx), float(cy), float(majax), float(minax), float(ang)


def _ellipse_patch_fit_contour(mask_bool_hw, *, edgecolor="cyan", linewidth=0.7):
    """
    Fit an ellipse by least squares with OpenCV ``fitEllipse`` to the largest outer
    contour of binary foreground and return a thin-stroked matplotlib Ellipse.
    Return None when no contour exists or fewer than five points are available.
    """
    from matplotlib.patches import Ellipse

    m = np.asarray(mask_bool_hw, dtype=np.uint8)
    if m.ndim != 2 or not np.any(m):
        return None
    m = (m > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    fitted = _safe_fit_ellipse_params(cnt)
    if fitted is None:
        return None
    cx, cy, majax, minax, ang = fitted
    return Ellipse(
        (cx, cy),
        majax,
        minax,
        angle=ang,
        fill=False,
        edgecolor=edgecolor,
        linewidth=float(linewidth),
        zorder=6,
    )


def _filled_ellipse_mask_from_binary(mask_bool_hw):
    """
    Consistent with ``fitEllipse``: fit the largest outer contour and use
    ``cv2.ellipse`` to create a filled ``(H, W)`` binary image. Return None when
    fitting is impossible.
    """
    m = np.asarray(mask_bool_hw, dtype=np.uint8)
    if m.ndim != 2 or not np.any(m):
        return None
    h, w = m.shape
    m = (m > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    fitted = _safe_fit_ellipse_params(cnt)
    if fitted is None:
        return None
    cx, cy, majax, minax, ang = fitted
    blank = np.zeros((h, w), dtype=np.uint8)
    try:
        center = (int(round(cx)), int(round(cy)))
        half_w = max(1, int(round(majax * 0.5)))
        half_h = max(1, int(round(minax * 0.5)))
        cv_int_max = int(np.iinfo(np.int32).max)
        if (
            abs(center[0]) > cv_int_max
            or abs(center[1]) > cv_int_max
            or half_w > cv_int_max
            or half_h > cv_int_max
        ):
            return None
        cv2.ellipse(
            blank,
            center,
            (half_w, half_h),
            ang,
            0.0,
            360.0,
            255,
            thickness=-1,
        )
    except (cv2.error, OverflowError, TypeError, ValueError):
        return None
    if not np.any(blank):
        return None
    return blank.astype(bool)


def _convex_hull_mask_from_binary(mask_bool_hw):
    """
    Compute and fill the convex hull of the largest outer contour, returning an
    ``(H, W)`` int64 binary image. Return None if a convex hull cannot be formed.
    """
    m = np.asarray(mask_bool_hw, dtype=np.uint8)
    if m.ndim != 2 or not np.any(m):
        return None
    h, w = m.shape
    m = (m > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return None
    hull = cv2.convexHull(cnt)
    blank = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blank, [hull], contourIdx=-1, color=255, thickness=-1)
    # Store 0/1 in sparse masks (not 255); evaluation and area filters use mask.dot().
    return (blank > 0).astype(np.int64)


def _save_debug_unmix_mask_figure(
    mask_hw,
    seed_yx_global,
    out_path,
    title=None,
    parent_mask_hw=None,
    parent_gray=0.55,
    ellipse_edgecolor="cyan",
    ellipse_linewidth=0.7,
):
    """Store a binary mask as RGB and highlight a seed in global ``(y, x)`` coordinates.

    The seed may correspond to a single component. If ``parent_mask_hw`` is an
    original large binary mask matching ``mask_hw``, render its True region gray as
    background, overlay the current mask in green, and draw a thin OpenCV
    ``fitEllipse`` boundary for the current sub-mask.
    """
    import matplotlib.pyplot as plt

    mask_hw = np.asarray(mask_hw, dtype=float)
    h, w = mask_hw.shape
    fg = np.clip(mask_hw, 0.0, 1.0) > 0.5
    vis = np.zeros((h, w, 3), dtype=np.float32)
    if parent_mask_hw is not None:
        pm = np.asarray(parent_mask_hw, dtype=bool)
        if pm.shape != (h, w):
            raise ValueError(
                f"parent_mask_hw shape {pm.shape} != mask_hw {(h, w)}"
            )
        vis[pm, 0] = parent_gray
        vis[pm, 1] = parent_gray
        vis[pm, 2] = parent_gray
    else:
        vis[..., 1] = np.clip(mask_hw, 0.0, 1.0)
    vis[fg, 0] = 0.0
    vis[fg, 1] = 1.0
    vis[fg, 2] = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(max(4, w / 80), max(4, h / 80)))
    ax.imshow(vis, interpolation="nearest")
    ax.set_aspect("equal")
    ell = _ellipse_patch_fit_contour(
        fg,
        edgecolor=ellipse_edgecolor,
        linewidth=ellipse_linewidth,
    )
    if ell is not None:
        ax.add_patch(ell)
    if seed_yx_global is not None and len(seed_yx_global) > 0:
        sy = np.asarray(seed_yx_global[:, 0], dtype=float)
        sx = np.asarray(seed_yx_global[:, 1], dtype=float)
        ax.scatter(
            sx,
            sy,
            c="red",
            s=18,
            marker="x",
            linewidths=0.55,
            label="seed",
            zorder=10,
        )
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title or "", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _prior_m_per_pixel_agg(M, agg, verbose=0):
    """Reduce ``M`` over time to one scalar per foreground pixel for ``prior_map_hw``."""
    M = np.asarray(M, dtype=np.float64)
    a = str(agg).lower().strip()
    if a in ("max", "peak"):
        return M.max(axis=1).astype(np.float32)
    if a in ("p95", "percentile95", "perc95"):
        return np.percentile(M, 95, axis=1).astype(np.float32)
    if a not in ("mean", "avg", "average", ""):
        _diag(
            verbose,
            2,
            "FACT_Guided_Demix_Sp: unknown prior_spatial_agg=%r; using mean." % (agg,),
        )
    return M.mean(axis=1).astype(np.float32)


def _mask_weighted_svd_denoise(V, M, gamma, n_components, random_state=42):
    """
    Consistent with the legacy DualModalityUnmixer: apply truncated SVD to
    ``V * (1 + gamma * M)`` and reconstruct it, using framewise information in
    ``M`` by weighting each pixel's T-dimensional time series before reduction.
    """
    from sklearn.decomposition import TruncatedSVD

    V = np.asarray(V, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    n_fg, T = V.shape
    n_comp = int(min(max(1, n_components), n_fg, T))
    Vw = V * (1.0 + float(gamma) * M)
    svd = TruncatedSVD(n_components=n_comp, random_state=int(random_state))
    U = svd.fit_transform(Vw.astype(np.float64))
    VT = svd.components_
    return (U @ VT).astype(np.float32)


def _extract_weighted_roi_trace(mov_tb, stat):
    """suite2p extraction: ``F = mov[:, ypix, xpix] @ (lam / sum(lam))`` for ``mov`` shaped ``(T, Ly, Lx)``."""
    yp = np.asarray(stat["ypix"], dtype=np.int64)
    xp = np.asarray(stat["xpix"], dtype=np.int64)
    lam = np.asarray(stat["lam"], dtype=np.float64)
    s = lam.sum() + 1e-12
    lam_n = lam / s
    return (mov_tb[:, yp, xp] @ lam_n).astype(np.float32)


def _fg_weights_from_stat(stat, Ly, Lx, coords):
    """Scatter ``stat['lam']`` across the full frame, then sample foreground pixels at ``coords`` with shape ``(n_fg,)``."""
    full = np.zeros((Ly, Lx), dtype=np.float64)
    full[np.asarray(stat["ypix"], dtype=int),
         np.asarray(stat["xpix"], dtype=int)] = np.asarray(stat["lam"], dtype=np.float64)
    yy = coords[:, 0].astype(int)
    xx = coords[:, 1].astype(int)
    return full[yy, xx]


def _blend_trace_with_prior_temporal(trace, prior_trace, blend):
    """
    Linearly blend an original fluorescence trace with a model-prior temporal trace.
    First normalize ``prior_trace`` to the mean and standard deviation of ``trace``,
    then blend using ``blend``.
    """
    b = float(np.clip(blend, 0.0, 1.0))
    t = np.asarray(trace, dtype=np.float64).ravel()
    p = np.asarray(prior_trace, dtype=np.float64).ravel()
    n = int(min(t.size, p.size))
    if n <= 0:
        return np.asarray(trace, dtype=np.float32).ravel()
    t0 = t[:n]
    if b <= 0.0:
        return t0.astype(np.float32, copy=False)

    p0 = p[:n]
    k_tail = max(1, int(round(n * 0.15)))
    p0 = p0 - float(np.mean(p0[-k_tail:]))
    s_p = float(np.std(p0))
    mu_t = float(np.mean(t0))
    s_t = float(np.std(t0))
    if s_p <= 1e-12:
        p_scaled = np.full_like(t0, mu_t, dtype=np.float64)
    else:
        p_scaled = ((p0 - float(np.mean(p0))) / (s_p + 1e-12)) * (s_t + 1e-12) + mu_t
    return ((1.0 - b) * t0 + b * p_scaled).astype(np.float32)


class DualModalityUnmixer:
    """
    Local ROI decomposition equivalent to suite2p ``sparsery`` in ``sparse_mode``,
    followed by suite2p-style weighted-sum extraction of F.

    Optionally construct a spatial prior map from model inference ``M`` and apply
    non-negative amplification to multiscale V1 peak maps before the main loop
    (``prior_peak_boost``). With zero boost and ``max_rois is None``, this section
    follows upstream suite2p apart from local-bounding-box and floating-point detail.

    When ``prior_svd_gamma > 0``, reconstruct framewise ``V * (1 + gamma * M)`` by
    truncated SVD and use the reconstruction as the detection movie for ``sparsery``.
    F remains extracted from original ``V`` using ``lam`` weights, so model predictions
    are not mixed directly into fluorescence traces.
    """

    def __init__(
        self,
        V,
        M,
        coords,
        img_shape,
        *,
        verbose=0,
        temporal_high_pass_width=100,
        spatial_background_filter_size=25,
        temporal_difference_batch_size=500,
        spatial_scale=0,
        detection_threshold_scaling=1.0,
        detection_iteration_blocks=20,
        max_binned_frames=5000,
        temporal_bin_seconds=1.0,
        frame_rate=15.0,
        active_frame_percentile=0.0,
        prior_peak_boost=1.0,
        prior_spatial_sharpness=0.0,
        prior_spatial_agg="mean",
        prior_svd_gamma=1.0,
        prior_svd_n_components=50,
        prior_svd_blend=1.0,
        prior_svd_random_state=42,
        max_rois=None,
        restrict_seeds_to_eroded_mask=True,
        seed_erosion_kernel_size=6,
    ):
        self.V = np.asarray(V, dtype=np.float32)
        self.M = np.asarray(M, dtype=np.float32)
        self.coords = np.asarray(coords, dtype=np.int64)
        self.img_shape = tuple(int(x) for x in img_shape)
        self.n_fg, self.T = self.V.shape
        Ly, Lx = self.img_shape
        self.verbose = int(verbose)
        self.temporal_high_pass_width = int(temporal_high_pass_width)
        self.spatial_background_filter_size = int(spatial_background_filter_size)
        self.temporal_difference_batch_size = int(temporal_difference_batch_size)
        self.spatial_scale = int(spatial_scale)
        self.detection_threshold_scaling = float(detection_threshold_scaling)
        self.detection_iteration_blocks = int(detection_iteration_blocks)
        self.max_binned_frames = int(max_binned_frames)
        self.temporal_bin_seconds = float(temporal_bin_seconds)
        self.frame_rate = float(frame_rate)
        self.active_frame_percentile = float(active_frame_percentile)
        self.prior_peak_boost = float(prior_peak_boost)
        self.prior_spatial_sharpness = float(prior_spatial_sharpness)
        self.prior_spatial_agg = str(prior_spatial_agg)
        self.prior_svd_gamma = float(prior_svd_gamma)
        self.prior_svd_n_components = int(prior_svd_n_components)
        self.prior_svd_blend = float(prior_svd_blend)
        self.prior_svd_random_state = int(prior_svd_random_state)
        self.max_rois = max_rois
        self.restrict_seeds_to_eroded_mask = bool(restrict_seeds_to_eroded_mask)
        self.seed_erosion_kernel_size = int(seed_erosion_kernel_size)

        V_det = self.V
        if self.prior_svd_gamma > 0.0:
            V_svd = _mask_weighted_svd_denoise(
                self.V,
                self.M,
                self.prior_svd_gamma,
                self.prior_svd_n_components,
                random_state=self.prior_svd_random_state,
            )
            b = float(np.clip(self.prior_svd_blend, 0.0, 1.0))
            V_det = ((1.0 - b) * self.V + b * V_svd).astype(np.float32)
            self._vprint(
                1,
                "DualModalityUnmixer: mask-weighted SVD for detection movie "
                f"(gamma={self.prior_svd_gamma:g}, n_comp={min(self.prior_svd_n_components, self.n_fg, self.T)}, blend={b:g}).",
            )

        self.mov_raw = np.zeros((self.T, Ly, Lx), dtype=np.float32)
        self.mov_raw_detect = np.zeros((self.T, Ly, Lx), dtype=np.float32)
        for i in range(self.n_fg):
            self.mov_raw[:, self.coords[i, 0], self.coords[i, 1]] = self.V[i, :]
            self.mov_raw_detect[:, self.coords[i, 0], self.coords[i, 1]] = V_det[i, :]

        self.prior_map_hw = None
        if self.prior_peak_boost > 0.0:
            pm = np.zeros((Ly, Lx), dtype=np.float32)
            pm[self.coords[:, 0], self.coords[:, 1]] = _prior_m_per_pixel_agg(
                self.M, self.prior_spatial_agg, verbose=self.verbose
            )
            if pm.max() > pm.min() + 1e-10:
                pm = (pm - pm.min()) / (pm.max() - pm.min())
            else:
                pm = np.zeros_like(pm, dtype=np.float32)
            if self.prior_spatial_sharpness < 1.0:
                sig = max(0.0, (1.0 - self.prior_spatial_sharpness) * 3.0)
                if sig > 1e-3:
                    pm = gaussian_filter(pm, sigma=float(sig))
                    if pm.max() > pm.min() + 1e-10:
                        pm = (pm - pm.min()) / (pm.max() - pm.min())
                    else:
                        pm = np.zeros_like(pm, dtype=np.float32)
            self.prior_map_hw = pm.astype(np.float32)

        self.seed_coords = np.zeros((0, 2), dtype=np.float64)
        self.stats = []
        self._detection_diagnostics = {}

    def _vprint(self, level, *args, **kwargs):
        if self.verbose >= int(level):
            print(*args, **kwargs)

    def run_pipeline(self):
        """Temporal binning → suite2p sparsery → lam-normalized F at full temporal resolution."""
        Ly, Lx = self.img_shape
        T = self.T
        self._vprint(1, "DualModalityUnmixer: sparse ROI detection")
        bin_size = max(
            1,
            int(T // max(1, self.max_binned_frames)),
            int(round(self.temporal_bin_seconds * self.frame_rate)),
        )
        mov_binned = _temporal_bin_movie(self.mov_raw_detect, bin_size, badframes=None)
        if mov_binned.size == 0:
            _diag(self.verbose, 2, "DualModalityUnmixer: empty binned movie; no ROIs.")
            return None, None

        max_it = 250 * max(1, self.detection_iteration_blocks)
        prior_hw = self.prior_map_hw if self.prior_peak_boost > 0.0 else None
        seed_valid_mask = None
        if self.restrict_seeds_to_eroded_mask:
            full = np.zeros((Ly, Lx), dtype=bool)
            cy = np.clip(self.coords[:, 0], 0, Ly - 1)
            cx = np.clip(self.coords[:, 1], 0, Lx - 1)
            full[cy, cx] = True
            ksz = int(self.seed_erosion_kernel_size)
            if ksz < 1:
                ksz = 3
            if ksz == 1:
                er = full
            else:
                if ksz % 2 == 0:
                    ksz += 1
                struct = np.ones((ksz, ksz), dtype=bool)
                er = binary_erosion(full, structure=struct, iterations=1)
            if not np.any(er):
                _diag(
                    self.verbose,
                    2,
                    "DualModalityUnmixer: restrict_seeds_to_eroded_mask=True but erosion "
                    "removed all foreground; falling back to the un-eroded parent mask for seeds.",
                )
                er = full
            seed_valid_mask = er
            self._vprint(
                1,
                "DualModalityUnmixer: peak and seed start positions are restricted to the eroded parent mask "
                f"(kernel={ksz}x{ksz}, fg_after={int(er.sum())}/{int(full.sum())}).",
            )
        self._detection_diagnostics, stats = _detect_sparse_rois(
            mov_binned,
            high_pass=self.temporal_high_pass_width,
            neuropil_high_pass=self.spatial_background_filter_size,
            batch_size=self.temporal_difference_batch_size,
            spatial_scale=self.spatial_scale,
            threshold_scaling=self.detection_threshold_scaling,
            max_iterations=max_it,
            percentile=self.active_frame_percentile,
            prior_peak_boost=self.prior_peak_boost,
            prior_map_hw=prior_hw,
            seed_valid_mask=seed_valid_mask,
            verbose=self.verbose,
        )
        if self.max_rois is not None:
            stats = stats[: int(self.max_rois)]
        self.stats = stats
        n_roi = len(stats)
        if n_roi == 0:
            _diag(self.verbose, 2, "DualModalityUnmixer: sparse ROI detection returned no ROIs.")
            return None, None

        self.seed_coords = np.array([st["med"] for st in stats], dtype=np.float64)
        traces = np.stack(
            [_extract_weighted_roi_trace(self.mov_raw, st) for st in stats], axis=0)
        spatial = np.stack(
            [_fg_weights_from_stat(st, Ly, Lx, self.coords) for st in stats], axis=1)
        self._vprint(1, "--> sparse ROI detection complete (%d ROIs)." % n_roi)
        return spatial, traces


def run_fact_guided_demix(
    V,
    M,
    coords,
    img_shape,
    gamma=3.0,
    alpha=1.0,
    beta=1.0,
    radius=5.0,
    num_pc=50,
    bg_threshold=0.3,
    max_iter=5,
    max_k=None,
    verbose=0,
    temporal_high_pass_width=100,
    spatial_background_filter_size=25,
    temporal_difference_batch_size=500,
    spatial_scale=0,
    detection_threshold_scaling=1.0,
    detection_iteration_blocks=20,
    max_binned_frames=5000,
    temporal_bin_seconds=1.0,
    frame_rate=15.0,
    active_frame_percentile=0.0,
    prior_peak_boost=1.0,
    prior_spatial_sharpness=0.0,
    prior_spatial_agg="mean",
    prior_svd_gamma=1.0,
    prior_svd_n_components=None,
    prior_svd_blend=1.0,
    prior_svd_random_state=42,
    max_rois=None,
    restrict_seeds_to_eroded_mask=True,
    seed_erosion_kernel_size=6,
):
    """
    Decompose foreground from one large mask bounding box using suite2p-compatible
    ``sparsery``, then calculate F for each ROI from the original full-temporal-
    resolution video using suite2p-style ``lam``-normalized weights.

    Legacy parameter names ``gamma, alpha, beta, radius, num_pc, bg_threshold,`` and
    ``max_iter`` remain compatible but no longer participate in this pipeline. To use
    the spatial prior from model ``M``, set ``prior_peak_boost > 0`` and optionally
    configure ``prior_spatial_sharpness`` and ``prior_spatial_agg``. To use framewise
    ``M``-assisted detection through legacy mask-weighted SVD, set ``prior_svd_gamma > 0``.
    In that case, sparsery receives a blend of SVD reconstruction and ``V``, while F
    is still extracted from original ``V``. ``prior_svd_n_components=None`` uses
    ``num_pc``. ``max_k`` maps to ``max_rois``; it truncates ROIs and is not a suite2p
    default. For exact upstream behavior, use ``max_k=None`` and ``max_rois=None``.
    For ``restrict_seeds_to_eroded_mask`` and ``seed_erosion_kernel_size``, see the
    same-named fields in ``fact_guided_demixing``.
    """
    max_rois_use = max_rois if max_rois is not None else max_k
    svd_ncomp = (
        int(prior_svd_n_components)
        if prior_svd_n_components is not None
        else int(num_pc)
    )
    unmixer = DualModalityUnmixer(
        V,
        M,
        coords,
        img_shape,
        verbose=verbose,
        temporal_high_pass_width=temporal_high_pass_width,
        spatial_background_filter_size=spatial_background_filter_size,
        temporal_difference_batch_size=temporal_difference_batch_size,
        spatial_scale=spatial_scale,
        detection_threshold_scaling=detection_threshold_scaling,
        detection_iteration_blocks=detection_iteration_blocks,
        max_binned_frames=max_binned_frames,
        temporal_bin_seconds=temporal_bin_seconds,
        frame_rate=frame_rate,
        active_frame_percentile=active_frame_percentile,
        prior_peak_boost=prior_peak_boost,
        prior_spatial_sharpness=prior_spatial_sharpness,
        prior_spatial_agg=prior_spatial_agg,
        prior_svd_gamma=prior_svd_gamma,
        prior_svd_n_components=svd_ncomp,
        prior_svd_blend=prior_svd_blend,
        prior_svd_random_state=prior_svd_random_state,
        max_rois=max_rois_use,
        restrict_seeds_to_eroded_mask=restrict_seeds_to_eroded_mask,
        seed_erosion_kernel_size=seed_erosion_kernel_size,
    )
    spatial, traces = unmixer.run_pipeline()
    seeds = np.asarray(unmixer.seed_coords, dtype=np.float64)
    return spatial, traces, seeds


def fact_guided_demixing(
    masks_final,
    infer_mask3d,
    input_img,
    mask_shape_2d,
    max_area,
    thresh_refine,
    min_area,
    gamma=4.0,
    alpha=1.0,
    beta=1.5,
    num_pc=50,
    radius=5.0,
    bg_threshold=0.3,
    max_iter=5,
    max_k=None,
    min_fg_pixels=None,
    debug_mode=0,
    debug_out_dir=None,
    verbose=0,
    return_full_traces=True,
    return_sources=True,
    temporal_high_pass_width=100,
    spatial_background_filter_size=25,
    temporal_difference_batch_size=500,
    spatial_scale=0,
    detection_threshold_scaling=1.0,
    detection_iteration_blocks=20,
    max_binned_frames=5000,
    temporal_bin_seconds=1.0,
    frame_rate=15.0,
    active_frame_percentile=0.0,
    prior_peak_boost=1.0,
    prior_spatial_sharpness=0.0,
    prior_spatial_agg="mean",
    prior_svd_gamma=1.0,
    prior_svd_n_components=None,
    prior_svd_blend=1.0,
    prior_svd_random_state=42,
    max_rois=None,
    demix_binarize_only=True,
    prior_mask_blend=0.0,
    prior_mask_power=1.0,
    prior_trace_blend=0.0,
    prior_trace_agg="mean",
    restrict_seeds_to_eroded_mask=True,
    seed_erosion_kernel_size=6,
    mask_shape_regularize="convex",
    morph_close_radius=3,
):
    """
    Apply suite2p-compatible sparse decomposition inside the bounding box of each
    large mask and calculate F at the original frame rate using ``lam`` weights.
    Optionally use the spatial prior from model ``infer_mask3d`` to boost peak maps
    through ``prior_peak_boost``, or use framewise mask-weighted SVD from ``M`` through
    ``prior_svd_gamma`` to provide a reconstructed detection movie to ``sparsery``.
    F remains extracted from original ``input_img`` foreground.

    Parameters
    ----
    masks_final : scipy.sparse.csr_matrix, shape (N_mask, H*W)
        Final post-processed masks stored in flattened form.
    infer_mask3d : np.ndarray, shape (T, H, W)
        Three-dimensional mask sequence produced by inference.
    input_img : np.ndarray, shape (T, H, W)
        Original input video.
    mask_shape_2d : tuple(int, int)
        Spatial dimensions ``(H, W)`` of the 2-D masks.
    max_area : int
        Area threshold for large masks that should be extracted and demixed again.
    thresh_refine : float
        Binarize suite2p spatial weights ``lam`` for each demixed ROI. Divide the
        bounding-box ``lam`` by its peak before comparison with ``thresh_refine``;
        thus ``0.3`` retains pixels at least 30% of the peak. This differs from the
        absolute threshold used by other full-image binary masks and prevents small
        absolute ``lam`` values from producing empty ROIs and no sub-mask output.
    min_area : int
        After binarization, discard 8-neighborhood connected components with an area
        smaller than this number of pixels. Unused when ``demix_binarize_only=True``.
    demix_binarize_only : bool
        When True, after demixing only binarize by ``thresh_refine`` and output one
        full binary mask per ROI. Do not split connected components, filter by
        ``min_area``, merge high-overlap masks, revert based on enclosing-ellipse
        centroids, or discard a sub-mask identical to the full parent mask. This
        makes it easier to view binary regions corresponding one-to-one with suite2p
        ROIs. The default False retains the original post-processing.
    max_k : int or None
        Maximum number of ROIs to retain, passed to ``max_rois``. ``None`` means no
        truncation, as in suite2p. When ``max_rois`` is also supplied, it takes priority.
    gamma, alpha, beta, radius, num_pc, bg_threshold, max_iter
        Retained for the legacy API; not used by the current suite2p pipeline.
    temporal_high_pass_width, spatial_background_filter_size, temporal_difference_batch_size, spatial_scale,
    detection_threshold_scaling, detection_iteration_blocks, max_binned_frames, temporal_bin_seconds, frame_rate,
    active_frame_percentile
        Correspond to related suite2p ``default_ops`` / ``detection_wrapper`` options.
        Following the upstream implementation, ``detection_iteration_blocks`` is
        multiplied by 250 to form the internal iteration limit.
    prior_peak_boost : float
        Degree of model spatial-prior participation: multiply multiscale ``V1`` by
        ``(1 + prior_peak_boost * prior_map)``. Set to ``0`` for suite2p-equivalent
        peak maps without prior amplification.
    prior_spatial_sharpness : float
        Spatial-prior smoothing: ``1.0`` leaves the prior unsmoothed; values below
        ``1`` apply Gaussian smoothing, with smaller values producing more smoothing.
    prior_spatial_agg : str
        Aggregation used to reduce ``infer_mask3d`` over time to a scalar 2-D prior
        for each foreground pixel: ``mean`` (default), ``max`` (temporal peak), or
        ``p95`` (95th percentile). Used only when ``prior_peak_boost > 0``.
    prior_svd_gamma : float
        When greater than zero, apply truncated SVD reconstruction to
        ``V * (1 + prior_svd_gamma * M)`` and blend it with original ``V`` using
        ``prior_svd_blend`` as the ``sparsery`` input movie. ``0`` disables this and
        matches pure suite2p detection input. Fluorescence traces still come from V.
    prior_svd_n_components : int or None
        Number of SVD components; use ``num_pc`` when None.
    prior_svd_blend : float
        Weight of SVD reconstruction in the detection movie, in ``[0, 1]``; ``1``
        fully uses the reconstruction-and-V blend.
    prior_svd_random_state : int
        Random seed for ``TruncatedSVD``.
    max_rois : int or None
        Maximum ROIs retained after decomposition, truncated in ``sparsery`` discovery
        order; None disables truncation.
    prior_mask_blend : float
        Blend ratio in ``[0, 1]`` between ``lam`` and the model spatial-prior map when
        binarizing demixed spatial maps. ``0`` uses only ``lam`` (original behavior);
        ``1`` uses only the prior map.
    prior_mask_power : float
        Power-law compression or enhancement of the blended spatial-score map: values
        above 1 emphasize high responses, while values below 1 smooth them.
    prior_trace_blend : float
        Blend ratio in ``[0, 1]`` between a post-demixing trace and the model temporal
        prior; ``0`` retains the original trace.
    prior_trace_agg : str
        Pixel aggregation for ``M`` when creating each ROI temporal prior:
        ``mean|max|p95``.
    restrict_seeds_to_eroded_mask : bool
        When True, each suite2p-style ``sparsery`` iteration selects multiscale peaks
        (the seed start points for ROI growth) only from the foreground subset of the
        original parent mask after ``binary_erosion`` with a square
        ``seed_erosion_kernel_size`` element (3×3 by default). This reduces seeds at
        background pixels near large-mask edges. False preserves peak selection over
        the entire parent mask.
    seed_erosion_kernel_size : int
        Used only when ``restrict_seeds_to_eroded_mask=True``. This is the square
        structuring-element side length in pixels and must be at least 1. ``1`` means
        no erosion, allowing seeds over the full parent mask. Even values are increased
        by 1 automatically to preserve symmetry.
    mask_shape_regularize : str
        Sub-mask shape-regularization mode: ``none|close|convex|ellipse``. ``none``
        retains original behavior; ``close`` applies elliptical-kernel closing to fill
        small gaps and suppress minor concavities; ``convex`` replaces the mask with
        its outer-contour convex hull; and ``ellipse`` replaces it with a filled fit.
    morph_close_radius : int
        Used only when ``mask_shape_regularize='close'``. This is the elliptical
        closing-kernel radius in pixels, giving a ``(2*r+1, 2*r+1)`` kernel; minimum 1.
    min_fg_pixels : int or None
        Skip demixing and retain the original mask when it has fewer foreground pixels
        than this value; None uses ``max(min_area, 3)``.
    debug_mode : int
        When 1, save each final binary mask as PNG after every large-mask demixing
        operation and overlay the seed for its current demixed component as a thin red
        cross. Default 0 writes no images.
    debug_out_dir : str or None
        Directory for debug images. When ``debug_mode=1`` and this is None, use
        ``fact_guided_unmix_debug`` under the current working directory.
    verbose : int
        Verbosity control: ``0`` prints nothing; ``1`` reports key pipeline details;
        ``2`` additionally reports detailed matrix statistics.

    Returns
    ----
    masks_merged : scipy.sparse.csr_matrix
        Flattened sparse matrix combining non-extracted masks and demixed masks.
        Large-mask rows are removed from the combined result; a demixed result that
        remains identical to its input large mask is not written back.
    pure_traces : np.ndarray
        By default (``return_full_traces=False``), the collection of clean traces from
        ``run_fact_guided_demix``, shaped ``(N_new, T)``. When
        ``return_full_traces=True``, return all traces aligned one-to-one with
        ``masks_merged`` rows, shaped ``(N_out, T)``; rows without direct demixing
        results use NaN placeholders.
    sources : np.ndarray, optional, shape (N_out,) or (N_new,)
        Returned only when ``return_sources=True``. True means the row's trace came
        from a demixed sub-mask; False means it has no demixed trace and is a
        placeholder or must be filled externally later.
    """
    def _return_pack(masks_out, traces_out, sources_out):
        if return_sources:
            return masks_out, traces_out, sources_out
        return masks_out, traces_out

    if masks_final.shape[0] == 0:
        empty_trace = np.zeros((0, infer_mask3d.shape[0]), dtype=np.float32)
        empty_sources = np.zeros((0,), dtype=bool)
        return _return_pack(masks_final, empty_trace, empty_sources)

    verbose = int(verbose)

    def vprint(level, *args, **kwargs):
        if verbose >= int(level):
            print(*args, **kwargs)

    H, W = mask_shape_2d
    T = infer_mask3d.shape[0]
    if infer_mask3d.shape != input_img.shape:
        raise ValueError("infer_mask3d and input_img must have the same shape (T, H, W).")
    shape_mode = str(mask_shape_regularize).strip().lower()
    valid_shape_modes = {"none", "close", "convex", "ellipse"}
    if shape_mode not in valid_shape_modes:
        raise ValueError(
            "mask_shape_regularize must be one of {'none', 'close', 'convex', 'ellipse'}."
        )
    morph_close_radius = max(1, int(morph_close_radius))

    if debug_mode == 1:
        out_dir = debug_out_dir if debug_out_dir is not None else os.path.join(
            os.getcwd(), "fact_guided_unmix_debug"
        )
        os.makedirs(out_dir, exist_ok=True)
        _debug_fig_i = 0
        vprint(
            1,
            f"[refine_large_masks_with_fact_guided_unmix] debug_mode=1, debug image directory: {out_dir}"
        )

    # Select large masks for extraction by binary area.
    masks_binary = masks_final.astype(bool).astype(np.uint8)
    area_arr = np.asarray(masks_binary.sum(axis=1)).ravel()
    large_idx = np.where(area_arr > max_area)[0]
    vprint(
        1,
        f"[refine_large_masks_with_fact_guided_unmix] masks with area above threshold {max_area}: {large_idx.size}"
    )

    if large_idx.size == 0:
        if return_full_traces:
            full_traces = np.full((masks_final.shape[0], T), np.nan, dtype=np.float32)
            full_sources = np.zeros((masks_final.shape[0],), dtype=bool)
            return _return_pack(masks_final, full_traces, full_sources)
        empty_trace = np.zeros((0, T), dtype=np.float32)
        empty_sources = np.zeros((0,), dtype=bool)
        return _return_pack(masks_final, empty_trace, empty_sources)

    keep_mask = np.ones(masks_final.shape[0], dtype=bool)
    keep_mask[large_idx] = False
    masks_kept = masks_final[keep_mask]

    refined_rows = []
    traces_all = []
    refined_trace_rows = []
    refined_source_rows = []

    def append_refined_row(row_csr, tr=None, source=False):
        refined_rows.append(row_csr)
        if tr is None:
            refined_trace_rows.append(np.full((T,), np.nan, dtype=np.float32))
        else:
            refined_trace_rows.append(np.asarray(tr, dtype=np.float32))
        refined_source_rows.append(bool(source))

    for idx in large_idx:
        mask_2d = masks_final.getrow(idx).toarray().reshape(H, W) > 0
        if not np.any(mask_2d):
            continue

        ys, xs = np.where(mask_2d)
        if ys.size == 0:
            continue

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        h_box = y1 - y0 + 1
        w_box = x1 - x0 + 1

        # Collect foreground pixels from the original large mask only; do not dilate.
        # The prior implementation created two zeroed ``(T, H_box, W_box)`` arrays,
        # filled foreground, transposed them, and gathered the same foreground indices.
        # Direct collection yields the identical float32 foreground matrix while
        # avoiding bounding-box-scale temporary memory.
        mask_crop = mask_2d[y0:y1 + 1, x0:x1 + 1]
        fg = mask_crop.ravel().astype(bool)
        flat_idx = np.flatnonzero(fg)
        n_fg = flat_idx.size
        min_fg = min_fg_pixels if min_fg_pixels is not None else max(int(min_area), 3)
        if n_fg < min_fg:
            append_refined_row(masks_final.getrow(idx).astype(np.int64), tr=None, source=False)
            continue

        infer_crop = infer_mask3d[:, y0:y1 + 1, x0:x1 + 1]
        input_crop = input_img[:, y0:y1 + 1, x0:x1 + 1]
        M_sub = np.asarray(infer_crop[:, mask_crop], dtype=np.float32).T.copy(order="C")
        V_sub = np.asarray(input_crop[:, mask_crop], dtype=np.float32).T.copy(order="C")
        H_box, W_box = h_box, w_box
        coords_sub = np.column_stack(np.nonzero(mask_crop))
        prior_spatial_sub = _prior_m_per_pixel_agg(
            M_sub, prior_spatial_agg, verbose=verbose
        ).astype(np.float64)
        if prior_spatial_sub.size != n_fg:
            prior_spatial_sub = np.zeros((n_fg,), dtype=np.float64)
        ps_min = float(np.min(prior_spatial_sub)) if prior_spatial_sub.size > 0 else 0.0
        ps_max = float(np.max(prior_spatial_sub)) if prior_spatial_sub.size > 0 else 0.0
        if ps_max > ps_min + 1e-12:
            prior_spatial_sub = (prior_spatial_sub - ps_min) / (ps_max - ps_min)
        else:
            prior_spatial_sub = np.zeros_like(prior_spatial_sub, dtype=np.float64)
        prior_spatial_full_flat = np.zeros(H_box * W_box, dtype=np.float64)
        prior_spatial_full_flat[flat_idx] = prior_spatial_sub

        overlapping_masks, pure_traces, seed_coords_local = run_fact_guided_demix(
            V=V_sub,
            M=M_sub,
            coords=coords_sub,
            img_shape=(H_box, W_box),
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            radius=radius,
            num_pc=num_pc,
            bg_threshold=bg_threshold,
            max_iter=max_iter,
            max_k=max_k,
            verbose=verbose,
            temporal_high_pass_width=temporal_high_pass_width,
            spatial_background_filter_size=spatial_background_filter_size,
            temporal_difference_batch_size=temporal_difference_batch_size,
            spatial_scale=spatial_scale,
            detection_threshold_scaling=detection_threshold_scaling,
            detection_iteration_blocks=detection_iteration_blocks,
            max_binned_frames=max_binned_frames,
            temporal_bin_seconds=temporal_bin_seconds,
            frame_rate=frame_rate,
            active_frame_percentile=active_frame_percentile,
            prior_peak_boost=prior_peak_boost,
            prior_spatial_sharpness=prior_spatial_sharpness,
            prior_spatial_agg=prior_spatial_agg,
            prior_svd_gamma=prior_svd_gamma,
            prior_svd_n_components=prior_svd_n_components,
            prior_svd_blend=prior_svd_blend,
            prior_svd_random_state=prior_svd_random_state,
            max_rois=max_rois,
            restrict_seeds_to_eroded_mask=restrict_seeds_to_eroded_mask,
            seed_erosion_kernel_size=seed_erosion_kernel_size,
        )

        # Retain the original large mask when no usable demixing result exists.
        if overlapping_masks is None or pure_traces is None:
            append_refined_row(masks_final.getrow(idx).astype(np.int64), tr=None, source=False)
            continue

        if pure_traces.ndim == 1:
            pure_traces = pure_traces[np.newaxis, :]

        # With foreground pixels only, run_fact_guided_demix returns spatial_masks
        # shaped ``(n_fg, K)``.
        if overlapping_masks.shape[0] == n_fg:
            mask_mat = overlapping_masks.T
        elif overlapping_masks.shape[1] == n_fg:
            mask_mat = overlapping_masks
        else:
            append_refined_row(masks_final.getrow(idx).astype(np.int64), tr=None, source=False)
            continue

        trace_chunk = []
        chunk_mask_rows = []  # Synchronized with trace_chunk; sub-mask rows from this large mask only.
        cc_seen_total = 0  # Components after binarization/splitting; distinguishes discarded identical masks from no valid components.
        for ki in range(mask_mat.shape[0]):
            if seed_coords_local is not None and ki < seed_coords_local.shape[0]:
                seed_yx_global_ki = np.array(
                    [
                        [
                            float(seed_coords_local[ki, 0]) + float(y0),
                            float(seed_coords_local[ki, 1]) + float(x0),
                        ]
                    ],
                    dtype=np.float64,
                )
            else:
                seed_yx_global_ki = None

            vec = np.asarray(mask_mat[ki]).ravel()
            full_flat = np.zeros(H_box * W_box, dtype=np.float64)
            full_flat[flat_idx] = vec
            mask_local_2d = full_flat.reshape(H_box, W_box)
            # A suite2p ``lam`` value at one pixel is often far below 1 because the
            # weight denominator is approximately ROI ``npix``. Absolute comparison
            # with the full-image binary-mask ``thresh_refine`` would empty the ROI,
            # producing no sub-mask or debug image. Normalize by this ROI's peak
            # weight before comparing with ``thresh_refine``.
            peak = float(np.max(mask_local_2d))
            if peak > 1e-12:
                lam_norm = (mask_local_2d / peak).astype(np.float64)
                b_mask = float(np.clip(prior_mask_blend, 0.0, 1.0))
                if b_mask > 1e-12:
                    prior_local_2d = prior_spatial_full_flat.reshape(H_box, W_box)
                    score_local = (1.0 - b_mask) * lam_norm + b_mask * prior_local_2d
                else:
                    score_local = lam_norm
                pwr = max(1e-6, float(prior_mask_power))
                if abs(pwr - 1.0) > 1e-6:
                    score_local = np.power(np.clip(score_local, 0.0, 1.0), pwr)
                mask_local_bin = (score_local >= float(thresh_refine)).astype(np.int64)
            else:
                mask_local_bin = np.zeros((H_box, W_box), dtype=np.int64)
            if demix_binarize_only:
                cc_list = [mask_local_bin] if np.any(mask_local_bin) else []
            else:
                cc_list = _split_cc_masks_ge_min_area(
                    mask_local_bin,
                    min_area=int(min_area),
                    structure=_CC_STRUCT8,
                )
            tr = np.asarray(pure_traces[ki], dtype=np.float32)
            b_trace = float(np.clip(prior_trace_blend, 0.0, 1.0))
            if b_trace > 1e-12:
                w = np.clip(vec.astype(np.float64), 0.0, None)
                sw = float(np.sum(w))
                if sw <= 1e-12:
                    w = mask_local_bin.reshape(-1)[flat_idx].astype(np.float64)
                    sw = float(np.sum(w))
                if sw > 1e-12:
                    prior_t = (w @ M_sub) / sw
                else:
                    prior_t = _prior_m_per_pixel_agg(
                        M_sub.T,
                        prior_trace_agg,
                        verbose=verbose,
                    ).astype(np.float64)
                    if prior_t.size != tr.size:
                        prior_t = np.mean(M_sub, axis=0).astype(np.float64)
                tr = _blend_trace_with_prior_temporal(tr, prior_t, b_trace)
            for cc in cc_list:
                cc_seen_total += 1
                cc = _fill_holes_binary_mask(cc)
                if shape_mode == "close":
                    ksz = 2 * morph_close_radius + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                    cc = cv2.morphologyEx(
                        cc.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                    ).astype(np.int64)
                elif shape_mode == "convex":
                    cc_convex = _convex_hull_mask_from_binary(cc)
                    if cc_convex is not None:
                        cc = cc_convex
                elif shape_mode == "ellipse":
                    cc_ellipse = _filled_ellipse_mask_from_binary(cc.astype(bool))
                    if cc_ellipse is not None:
                        cc = cc_ellipse.astype(np.int64)
                mask_full = np.zeros((H, W), dtype=np.int64)
                mask_full[y0:y1 + 1, x0:x1 + 1] = cc
                # Discard a demixed sub-mask identical to the input large mask over
                # the full image to avoid retaining the original mask twice.
                if (not demix_binarize_only) and np.array_equal(
                        mask_full.astype(bool), mask_2d):
                    vprint(
                        2,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)} ki={ki}: binary sub-mask equals the parent mask over the full image; skipped.",
                    )
                    continue
                row_csr = sp.csr_matrix(mask_full.reshape(1, H * W))
                append_refined_row(row_csr, tr=tr, source=True)
                chunk_mask_rows.append(row_csr)
                trace_chunk.append(tr)
                if debug_mode == 1:
                    _debug_fig_i += 1
                    fn = os.path.join(
                        out_dir,
                        f"unmix_maskidx{int(idx)}_fig{_debug_fig_i:04d}.png",
                    )
                    _save_debug_unmix_mask_figure(
                        mask_full,
                        seed_yx_global_ki,
                        fn,
                        title=f"row={idx} fig={_debug_fig_i} ki={ki}",
                        parent_mask_hw=mask_2d,
                    )

        if not trace_chunk:
            # Restore the original large row when no sub-mask is written. Previously,
            # if ``cc_seen_total > 0`` but all components were discarded as identical
            # to the parent, neither sub-masks nor the large row were retained, making
            # the large row disappear from the merged result and corrupting mask counts.
            if (
                cc_seen_total > 0
                and pure_traces is not None
                and int(pure_traces.shape[0]) > 0
            ):
                vprint(
                    1,
                    "[refine_large_masks_with_fact_guided_unmix] "
                    f"idx={int(idx)}: demixing produced {int(pure_traces.shape[0])} ROIs, but no binary sub-mask "
                    "was written (common causes: a sub-mask equals the parent over the full image while "
                    "demix_binarize_only is disabled, or every split connected component is smaller than min_area); "
                    "restoring the original large mask.",
                )
            elif (
                cc_seen_total == 0
                and pure_traces is not None
                and int(pure_traces.shape[0]) > 0
            ):
                vprint(
                    1,
                    "[refine_large_masks_with_fact_guided_unmix] "
                    f"idx={int(idx)}: demixing produced {int(pure_traces.shape[0])} ROIs, but every ROI's "
                    "binarized lam is empty (if this persists, check whether thresh_refine is near 1 or lam is all zero); "
                    "restoring the original large mask.",
                )
            append_refined_row(masks_final.getrow(idx).astype(np.int64), tr=None, source=False)
        else:
            stacked_tc = np.stack(trace_chunk, axis=0)
            nm = stacked_tc.shape[0]
            masks_chunk = sp.vstack(chunk_mask_rows, format="csr")

            # Rule: if a sub-mask's intersection with another sub-mask exceeds 0.95
            # of its own area, merge it into the other sub-mask and retain the other
            # sub-mask's trace.
            if (not demix_binarize_only) and nm >= 2:
                masks_bool = [
                    masks_chunk.getrow(i).toarray().reshape(H, W).astype(bool)
                    for i in range(nm)
                ]
                active = np.ones(nm, dtype=bool)
                merge_events = []
                for i in range(nm):
                    if not active[i]:
                        continue
                    area_i = int(np.count_nonzero(masks_bool[i]))
                    if area_i == 0:
                        continue
                    for j in range(nm):
                        if i == j or (not active[j]):
                            continue
                        inter_ij = int(np.count_nonzero(masks_bool[i] & masks_bool[j]))
                        ratio_i_in_j = inter_ij / float(area_i)
                        if ratio_i_in_j > 0.95:
                            masks_bool[j] = masks_bool[j] | masks_bool[i]
                            active[i] = False
                            merge_events.append((i, j, ratio_i_in_j))
                            break

                if merge_events:
                    for _ in range(nm):
                        refined_rows.pop()
                        refined_trace_rows.pop()
                        refined_source_rows.pop()

                    new_chunk_mask_rows = []
                    new_trace_chunk = []
                    for i in range(nm):
                        if not active[i]:
                            continue
                        mb = _fill_holes_binary_mask(masks_bool[i])
                        masks_bool[i] = mb.astype(bool)
                        row_csr = sp.csr_matrix(mb.reshape(1, H * W))
                        new_chunk_mask_rows.append(row_csr)
                        new_trace_chunk.append(np.asarray(trace_chunk[i], dtype=np.float32))
                        append_refined_row(row_csr, tr=np.asarray(trace_chunk[i], dtype=np.float32), source=True)

                    chunk_mask_rows = new_chunk_mask_rows
                    trace_chunk = new_trace_chunk
                    stacked_tc = np.stack(trace_chunk, axis=0)
                    nm = stacked_tc.shape[0]
                    masks_chunk = sp.vstack(chunk_mask_rows, format="csr")

                    vprint(
                        1,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)}: high-overlap merging triggered {len(merge_events)} time(s):"
                    )
                    for src_i, dst_j, rr in merge_events:
                        vprint(
                            1,
                            "[refine_large_masks_with_fact_guided_unmix] "
                            f"idx={int(idx)}: sub-mask {src_i} merged into {dst_j} "
                            f"(inter/src={rr:.4f}); retaining trace {dst_j}."
                        )

            # With exactly two sub-masks, treat splitting as invalid and restore the
            # original large mask when the filled enclosing-ellipse centroid distance is below 2 px.
            revert_two_close_ellipse_com = False
            if (not demix_binarize_only) and nm == 2:
                m0 = masks_chunk.getrow(0).toarray().reshape(H, W) > 0
                m1 = masks_chunk.getrow(1).toarray().reshape(H, W) > 0
                e0 = _filled_ellipse_mask_from_binary(m0)
                e1 = _filled_ellipse_mask_from_binary(m1)
                if e0 is not None and e1 is not None:
                    c0 = np.asarray(
                        center_of_mass(e0.astype(np.float64)), dtype=np.float64
                    )
                    c1 = np.asarray(
                        center_of_mass(e1.astype(np.float64)), dtype=np.float64
                    )
                    if np.linalg.norm(c0 - c1) < 2.0:
                        revert_two_close_ellipse_com = True

            if revert_two_close_ellipse_com:
                for _ in range(nm):
                    refined_rows.pop()
                    refined_trace_rows.pop()
                    refined_source_rows.pop()
                append_refined_row(masks_final.getrow(idx).astype(np.int64), tr=None, source=False)
                vprint(
                    1,
                    "[refine_large_masks_with_fact_guided_unmix] "
                    f"idx={int(idx)}: filled enclosing-ellipse COM distance for two sub-masks is below 2 pixels; "
                    "reverting the split and restoring the original large mask (no pure_traces written for this group)."
                )
            else:
                traces_all.append(stacked_tc)
                # Everything below is diagnostic-only.  GUI and reproduction
                # runs use verbose=0, so avoid both the unnecessary matrix work
                # and any numerical risk from fitting display-only ellipses.
                if verbose < 1:
                    continue
                dist_m, iou_m, com_yx = _pairwise_com_distance_and_iou_matrices(
                    masks_chunk, H, W
                )
                vprint(
                    1,
                    "[refine_large_masks_with_fact_guided_unmix] "
                    f"COM (y, x) for the {nm} sub-masks demixed from original large mask idx={int(idx)}:"
                )
                vprint(
                    2,
                    np.array2string(
                        com_yx,
                        precision=3,
                        suppress_small=False,
                        max_line_width=120,
                    )
                )
                if nm >= 2:
                    vprint(
                        1,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)} pairwise sub-mask COM Euclidean distances (pixels):"
                    )
                    vprint(
                        2,
                        np.array2string(
                            dist_m,
                            precision=4,
                            suppress_small=False,
                            max_line_width=120,
                        )
                    )
                    vprint(
                        1,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)} pairwise sub-mask IoU:"
                    )
                    vprint(
                        2,
                        np.array2string(
                            iou_m,
                            precision=4,
                            suppress_small=True,
                            max_line_width=120,
                        )
                    )

                if nm >= 2:
                    corr_tc = np.corrcoef(stacked_tc)
                    vprint(
                        2,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)} pairwise Pearson-correlation matrix for sub-mask traces "
                        "(row/column order matches sub-masks and pure_traces rows):\n"
                        + np.array2string(
                            corr_tc,
                            precision=4,
                            suppress_small=True,
                            max_line_width=120,
                        )
                    )
                else:
                    vprint(
                        1,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)}: only one sub-mask / one trace; no pairwise multi-trace correlation."
                    )

                ell_rows = []
                ell_all_ok = True
                for i in range(nm):
                    m2 = masks_chunk.getrow(i).toarray().reshape(H, W) > 0
                    em = _filled_ellipse_mask_from_binary(m2)
                    if em is None:
                        ell_all_ok = False
                        break
                    ell_rows.append(
                        sp.csr_matrix(em.astype(np.int64).reshape(1, H * W))
                    )
                if ell_all_ok and len(ell_rows) == nm:
                    masks_ell = sp.vstack(ell_rows, format="csr")
                    dist_e, iou_e, com_e = _pairwise_com_distance_and_iou_matrices(
                        masks_ell, H, W
                    )
                    vprint(
                        1,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)} filled enclosing-ellipse COM (y, x) for each sub-mask, "
                        "in sub-mask order:"
                    )
                    vprint(
                        2,
                        np.array2string(
                            com_e,
                            precision=3,
                            suppress_small=False,
                            max_line_width=120,
                        )
                    )
                    if nm >= 2:
                        vprint(
                            1,
                            "[refine_large_masks_with_fact_guided_unmix] "
                            f"idx={int(idx)} pairwise filled enclosing-ellipse COM Euclidean distances (pixels):"
                        )
                        vprint(
                            2,
                            np.array2string(
                                dist_e,
                                precision=4,
                                suppress_small=False,
                                max_line_width=120,
                            )
                        )
                        vprint(
                            1,
                            "[refine_large_masks_with_fact_guided_unmix] "
                            f"idx={int(idx)} pairwise filled enclosing-ellipse IoU:"
                        )
                        vprint(
                            2,
                            np.array2string(
                                iou_e,
                                precision=4,
                                suppress_small=True,
                                max_line_width=120,
                            )
                        )
                else:
                    vprint(
                        1,
                        "[refine_large_masks_with_fact_guided_unmix] "
                        f"idx={int(idx)}: some sub-masks could not produce filled enclosing ellipses; skipping ellipse IoU/COM output."
                    )

    if len(refined_rows) > 0:
        masks_refined = sp.vstack(refined_rows, format="csr")
        traces_refined = np.vstack(refined_trace_rows).astype(np.float32)
        sources_refined = np.asarray(refined_source_rows, dtype=bool)
    else:
        masks_refined = sp.csr_matrix((0, H * W), dtype=np.int64)
        traces_refined = np.full((0, T), np.nan, dtype=np.float32)
        sources_refined = np.zeros((0,), dtype=bool)

    if masks_kept.shape[0] > 0 and masks_refined.shape[0] > 0:
        masks_merged = sp.vstack([masks_kept, masks_refined], format="csr").astype(np.int64)
    elif masks_refined.shape[0] > 0:
        masks_merged = masks_refined.astype(np.int64)
    else:
        masks_merged = masks_kept.astype(np.int64)

    # Ensure all rows are 0/1 (kept rows may be 1; refined rows may still be non-binary).
    masks_merged = (masks_merged > 0).astype(np.int64)

    traces_kept = np.full((masks_kept.shape[0], T), np.nan, dtype=np.float32)
    sources_kept = np.zeros((masks_kept.shape[0],), dtype=bool)
    if traces_kept.shape[0] > 0 and traces_refined.shape[0] > 0:
        traces_full_out = np.vstack([traces_kept, traces_refined]).astype(np.float32)
        sources_full_out = np.concatenate([sources_kept, sources_refined]).astype(bool)
    elif traces_refined.shape[0] > 0:
        traces_full_out = traces_refined.astype(np.float32)
        sources_full_out = sources_refined.astype(bool)
    else:
        traces_full_out = traces_kept.astype(np.float32)
        sources_full_out = sources_kept.astype(bool)

    if len(traces_all) > 0:
        pure_traces_out = np.vstack(traces_all).astype(np.float32)
    else:
        pure_traces_out = np.zeros((0, T), dtype=np.float32)

    if return_full_traces:
        return _return_pack(masks_merged, traces_full_out, sources_full_out)
    demix_sources = np.ones((pure_traces_out.shape[0],), dtype=bool)
    return _return_pack(masks_merged, pure_traces_out, demix_sources)

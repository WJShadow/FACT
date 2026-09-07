from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
    pytorch_after,
)

tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"

__all__ = [
    "sliding_window_inference",
    "sliding_window_inference_with_progress",
    "sliding_window_inference_hdf5",
]


def sliding_window_inference(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    
    buffered = buffer_steps is not None and buffer_steps > 0
    num_spatial_dims = len(inputs.shape) - 2
    if buffered:
        if buffer_dim < -num_spatial_dims or buffer_dim > num_spatial_dims:
            raise ValueError(f"buffer_dim must be in [{-num_spatial_dims}, {num_spatial_dims}], got {buffer_dim}.")
        if buffer_dim < 0:
            buffer_dim += num_spatial_dims
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=not buffered)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    if not buffered:
        non_blocking = False
        windows_range = range(0, total_slices, sw_batch_size)
    else:
        slices, n_per_batch, b_slices, windows_range = _create_buffered_slices(
            slices, batch_size, sw_batch_size, buffer_dim, buffer_steps
        )
        non_blocking, _ss = torch.cuda.is_available(), -1
        for x in b_slices[:n_per_batch]:
            if x[1] < _ss:  # detect overlapping slices
                non_blocking = False
                break
            _ss = x[2]

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map_ = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map_ = compute_importance_map(
                valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
            )
            if len(importance_map_.shape) == num_spatial_dims and not process_fn:
                importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in tqdm(windows_range) if progress else windows_range:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, b_slices[b_s][0] if buffered else total_slices))
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
        seg_prob_out = predictor(win_data, *args, **kwargs)  # batched patch

        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        if process_fn:
            seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
        else:
            w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        if buffered:
            c_start, c_end = b_slices[b_s][1:]
            if not sw_device_buffer:
                k = seg_tuple[0].shape[1]  # len(seg_tuple) > 1 is currently ignored
                sp_size = list(image_size)
                sp_size[buffer_dim] = c_end - c_start
                sw_device_buffer = [torch.zeros(size=[1, k, *sp_size], dtype=compute_dtype, device=sw_device)]
            for p, s in zip(seg_tuple[0], unravel_slice):
                offset = s[buffer_dim + 2].start - c_start
                s[buffer_dim + 2] = slice(offset, offset + roi_size[buffer_dim])
                s[0] = slice(0, 1)
                sw_device_buffer[0][s] += p * w_t
            b_i += len(unravel_slice)
            if b_i < b_slices[b_s][0]:
                continue
        else:
            sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if not buffered and seg_shape != roi_size:
                z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor: Callable = torch.empty if non_blocking else torch.zeros  # type: ignore
                output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                w_t_ = w_t.to(device)
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_
            if buffered:
                o_slice = [slice(None)] * len(inputs.shape)
                o_slice[buffer_dim + 2] = slice(c_start, c_end)
                img_b = b_s // n_per_batch  # image batch index
                o_slice[0] = slice(img_b, img_b + 1)
                if non_blocking:
                    output_image_list[0][o_slice].copy_(sw_device_buffer[0], non_blocking=non_blocking)
                else:
                    output_image_list[0][o_slice] += sw_device_buffer[0].to(device=device)
            else:
                sw_device_buffer[ss] *= w_t
                sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
                _compute_coords(unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])
        sw_device_buffer = []
        if buffered:
            b_s += 1

    if non_blocking:
        torch.cuda.current_stream().synchronize()

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        for ss, output_i in enumerate(output_image_list):
            zoom_scale = [_shape_d / _roi_size_d for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)]
            final_slicing: list[slice] = []
            for sp in range(num_spatial_dims):
                si = num_spatial_dims - sp - 1
                slice_dim = slice(
                    int(round(pad_size[sp * 2] * zoom_scale[si])),
                    int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                )
                final_slicing.insert(0, slice_dim)
            output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

    return final_output  # type: ignore


def sliding_window_inference_with_progress(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    *args: Any,
    progress_callback: Callable[[float, int, int], None] | None = None,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """Call the original implementation while reporting completed windows.

    The original ``sliding_window_inference`` above is intentionally left
    unchanged. Progress is observed by wrapping the predictor, which is called
    once for every patch batch assembled by the original implementation.
    """

    num_spatial_dims = len(inputs.shape) - 2
    overlap_tuple = ensure_tuple_rep(overlap, num_spatial_dims)
    image_size_input = tuple(int(value) for value in inputs.shape[2:])
    roi_size_tuple = fall_back_tuple(roi_size, image_size_input)
    image_size = tuple(
        max(image_size_input[index], int(roi_size_tuple[index]))
        for index in range(num_spatial_dims)
    )
    scan_interval = _get_scan_interval(
        image_size, roi_size_tuple, num_spatial_dims, overlap_tuple
    )
    window_count = len(
        dense_patch_slices(image_size, roi_size_tuple, scan_interval)
    ) * int(inputs.shape[0])
    completed = 0

    if progress_callback is not None:
        progress_callback(0.0, 0, window_count)

    def reporting_predictor(win_data: torch.Tensor, *predictor_args: Any, **predictor_kwargs: Any):
        nonlocal completed
        output = predictor(win_data, *predictor_args, **predictor_kwargs)
        completed = min(window_count, completed + int(win_data.shape[0]))
        if progress_callback is not None:
            fraction = 1.0 if window_count <= 0 else completed / window_count
            progress_callback(float(fraction), completed, window_count)
        return output

    return sliding_window_inference(
        inputs,
        roi_size,
        sw_batch_size,
        reporting_predictor,
        overlap,
        mode,
        sigma_scale,
        padding_mode,
        cval,
        sw_device,
        device,
        progress,
        roi_weight_map,
        process_fn,
        buffer_steps,
        buffer_dim,
        *args,
        **kwargs,
    )


def sliding_window_inference_hdf5(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    *args: Any,
    scratch_path: str,
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress_callback: Callable[[float, int, int], None] | None = None,
    process_fn: Callable | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run sliding-window inference with the accumulators backed by HDF5.

    This is intentionally a single-tensor implementation for FACT's two-class
    predictor.  Window coordinates, padding, importance maps, predictor order,
    and float32 arithmetic mirror :func:`sliding_window_inference`; only the
    output and overlap count arrays are persisted in the supplied scratch file.
    """

    import h5py

    if process_fn is not None:
        raise ValueError("HDF5 inference does not support process_fn for FACT predictors.")
    if isinstance(inputs, MetaTensor):
        inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("HDF5 inference requires a torch.Tensor input.")
    if inputs.ndim < 3:
        raise ValueError("HDF5 inference requires batch, channel, and spatial dimensions.")

    num_spatial_dims = len(inputs.shape) - 2
    overlap_tuple = ensure_tuple_rep(overlap, num_spatial_dims)
    for value in overlap_tuple:
        if value < 0 or value >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap_tuple}.")
    compute_dtype = inputs.dtype
    batch_size, _, *image_size_input = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device
    roi_size_tuple = fall_back_tuple(roi_size, image_size_input)
    image_size = tuple(
        max(int(image_size_input[index]), int(roi_size_tuple[index]))
        for index in range(num_spatial_dims)
    )

    pad_size: list[int] = []
    for index in range(len(inputs.shape) - 1, 1, -1):
        diff = max(int(roi_size_tuple[index - 2]) - int(inputs.shape[index]), 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )

    scan_interval = _get_scan_interval(
        image_size, roi_size_tuple, num_spatial_dims, overlap_tuple
    )
    slices = dense_patch_slices(image_size, roi_size_tuple, scan_interval, return_slice=True)
    num_win = len(slices)
    total_slices = num_win * int(batch_size)
    windows_range = range(0, total_slices, max(1, int(sw_batch_size)))
    valid_patch_size = get_valid_patch_size(image_size, roi_size_tuple)
    try:
        valid_p_size = ensure_tuple(valid_patch_size)
        importance_map = compute_importance_map(
            valid_p_size,
            mode=mode,
            sigma_scale=sigma_scale,
            device=sw_device,
            dtype=compute_dtype,
        )
        if len(importance_map.shape) == num_spatial_dims:
            importance_map = importance_map[None, None]
    except Exception as exc:
        raise RuntimeError(
            f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
            "Seems to be OOM. Please try a smaller patch size or mode='constant'."
        ) from exc
    importance_map = convert_data_type(
        importance_map, torch.Tensor, device=sw_device, dtype=compute_dtype
    )[0]

    completed = 0
    if progress_callback is not None:
        progress_callback(0.0, 0, total_slices)

    with h5py.File(str(scratch_path), "w") as h5_file:
        output_ds = None
        count_ds = None
        output_shape: tuple[int, ...] | None = None
        for slice_start in windows_range:
            slice_range = range(
                slice_start,
                min(slice_start + max(1, int(sw_batch_size)), total_slices),
            )
            unravel_slice = [
                [slice(index // num_win, index // num_win + 1), slice(None)]
                + list(slices[index % num_win])
                for index in slice_range
            ]
            if len(unravel_slice) > 1:
                win_data = torch.cat(
                    [inputs[window_slice] for window_slice in unravel_slice]
                ).to(sw_device)
            else:
                win_data = inputs[unravel_slice[0]].to(sw_device)
            seg_prob_out = predictor(win_data, *args, **kwargs)
            if not isinstance(seg_prob_out, torch.Tensor):
                raise TypeError("HDF5 FACT inference requires a tensor predictor output.")
            seg = seg_prob_out
            weights = importance_map
            if len(weights.shape) == num_spatial_dims:
                weights = weights[None, None]
            weights = weights.to(dtype=compute_dtype, device=sw_device)
            seg_shape = tuple(int(value) for value in seg.shape[2:])
            z_scale = None
            if seg_shape != tuple(int(value) for value in roi_size_tuple):
                z_scale = [
                    out_size / float(in_size)
                    for out_size, in_size in zip(seg_shape, roi_size_tuple)
                ]
                weights = F.interpolate(weights, seg_shape, mode=_nearest_mode)
            if output_ds is None:
                output_shape = (int(batch_size), int(seg.shape[1]), *image_size)
                chunk_spatial = tuple(max(1, min(32, int(size))) for size in output_shape[2:])
                output_ds = h5_file.create_dataset(
                    "accumulator",
                    shape=output_shape,
                    dtype="float32",
                    chunks=(1, 1, *chunk_spatial),
                    compression=None,
                    shuffle=False,
                )
                count_ds = h5_file.create_dataset(
                    "count",
                    shape=(1, 1, *output_shape[2:]),
                    dtype="float32",
                    chunks=(1, 1, *chunk_spatial),
                    compression=None,
                    shuffle=False,
                )
                # The count map is independent of batch order and is built
                # once, exactly as in the in-memory MONAI implementation.
                weights_np = weights.detach().to(device="cpu").numpy().astype(np.float32, copy=False)
                for spatial_slice in slices:
                    target = (slice(None), slice(None), *spatial_slice)
                    current = np.asarray(count_ds[target], dtype=np.float32)
                    np.add(current, weights_np, out=current, casting="unsafe")
                    count_ds[target] = current

            assert output_ds is not None and count_ds is not None and output_shape is not None
            weights_np = weights.detach().to(device="cpu").numpy().astype(np.float32, copy=False)
            for patch, window_slice in zip(seg, unravel_slice):
                patch_np = (patch * weights).detach().to(device="cpu").numpy().astype(np.float32, copy=False)
                idx = list(window_slice)
                if z_scale is not None:
                    spatial = []
                    for axis, original in enumerate(window_slice[2:]):
                        spatial.append(
                            slice(
                                int(original.start * z_scale[axis]),
                                int(original.stop * z_scale[axis]),
                            )
                        )
                    idx[2:] = spatial
                target = tuple(idx)
                current = np.asarray(output_ds[target], dtype=np.float32)
                np.add(current, patch_np, out=current, casting="unsafe")
                output_ds[target] = current
            completed = min(total_slices, completed + len(unravel_slice))
            if progress_callback is not None:
                progress_callback(
                    1.0 if total_slices <= 0 else completed / total_slices,
                    completed,
                    total_slices,
                )

        if output_ds is None or count_ds is None or output_shape is None:
            raise RuntimeError("FACT HDF5 inference produced no predictor output.")
        final_output = np.empty(output_shape, dtype=np.float32)
        first_spatial = int(output_shape[2])
        step = max(1, min(32, first_spatial))
        for start in range(0, first_spatial, step):
            stop = min(first_spatial, start + step)
            region = (slice(None), slice(None), slice(start, stop), *([slice(None)] * (num_spatial_dims - 1)))
            numerator = np.asarray(output_ds[region], dtype=np.float32)
            denominator = np.asarray(count_ds[(slice(None), slice(None), slice(start, stop), *([slice(None)] * (num_spatial_dims - 1)))], dtype=np.float32)
            np.divide(numerator, denominator, out=numerator, where=denominator != 0)
            final_output[region] = numerator

    if any(pad_size):
        zoom_scale = [
            output_size / float(roi_size_tuple[index])
            for index, output_size in enumerate(final_output.shape[2:])
        ]
        final_slicing: list[slice] = []
        for spatial_index in range(num_spatial_dims):
            reverse_index = num_spatial_dims - spatial_index - 1
            final_slicing.insert(
                0,
                slice(
                    int(round(pad_size[spatial_index * 2] * zoom_scale[reverse_index])),
                    int(round(
                        (pad_size[spatial_index * 2] + image_size_input[reverse_index])
                        * zoom_scale[reverse_index]
                    )),
                ),
            )
        final_output = final_output[(slice(None), slice(None), *final_slicing)]
    return torch.from_numpy(np.ascontiguousarray(final_output, dtype=np.float32)).to(device)


def _create_buffered_slices(slices, batch_size, sw_batch_size, buffer_dim, buffer_steps):
    """rearrange slices for buffering"""
    slices_np = np.asarray(slices)
    slices_np = slices_np[np.argsort(slices_np[:, buffer_dim, 0], kind="mergesort")]
    slices = [tuple(slice(c[0], c[1]) for c in i) for i in slices_np]
    slices_np = slices_np[:, buffer_dim]

    _, _, _b_lens = np.unique(slices_np[:, 0], return_counts=True, return_index=True)
    b_ends = np.cumsum(_b_lens).tolist()  # possible buffer flush boundaries
    x = [0, *b_ends][:: min(len(b_ends), int(buffer_steps))]
    if x[-1] < b_ends[-1]:
        x.append(b_ends[-1])
    n_per_batch = len(x) - 1
    windows_range = [
        range(b * x[-1] + x[i], b * x[-1] + x[i + 1], sw_batch_size)
        for b in range(batch_size)
        for i in range(n_per_batch)
    ]
    b_slices = []
    for _s, _r in enumerate(windows_range):
        s_s = slices_np[windows_range[_s - 1].stop % len(slices) if _s > 0 else 0, 0]
        s_e = slices_np[(_r.stop - 1) % len(slices), 1]
        b_slices.append((_r.stop, s_s, s_e))  # buffer index, slice start, slice end
    windows_range = itertools.chain(*windows_range)  # type: ignore
    return slices, n_per_batch, b_slices, windows_range


def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[idx_zm] += p


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)

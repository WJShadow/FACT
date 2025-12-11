from typing import Any
import numpy as np
import torch
import cupy as cp
from monai.utils import optional_import
cp_ndarray, _ = optional_import("cupy", name="ndarray")
UNSUPPORTED_TYPES = {np.dtype("uint16"): np.int32, np.dtype("uint32"): np.int64, np.dtype("uint64"): np.int64}
import monai
from monai.config.type_definitions import DtypeLike, NdarrayTensor


def convert_to_cupy(data: Any, dtype: np.dtype | None = None, wrap_sequence: bool = False, safe: bool = False) -> Any:
    """
    Utility to convert the input data to a cupy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to cupy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, cupy array, list, dictionary, int, float, bool, str, etc.
            Tensor, numpy array, cupy array, float, int, bool are converted to cupy arrays,
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to Cupy array, tt must be an argument of `numpy.dtype`,
            for more details: https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if safe:
        data = safe_dtype_range(data, dtype)
    # direct calls
    if isinstance(data, torch.Tensor) and data.device.type == "cuda":
        # This is needed because of https://github.com/cupy/cupy/issues/7874#issuecomment-1727511030
        if data.dtype == torch.bool:
            data = data.detach().to(torch.uint8)
            if dtype is None:
                dtype = bool  # type: ignore
        data = cp.asarray(data, dtype) 
    elif isinstance(data, (cp_ndarray, np.ndarray, torch.Tensor, float, int, bool)):
        data = cp.asarray(data, dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_cupy(i, dtype) for i in data]
        return cp.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_cupy(i, dtype) for i in data)
        return cp.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_cupy(v, dtype) for k, v in data.items()}
    # make it contiguous
    if not isinstance(data, cp.ndarray):
        raise ValueError(f"The input data type [{type(data)}] cannot be converted into cupy arrays!")

    if data.ndim > 0:
        data = cp.ascontiguousarray(data)
    return data



def convert_to_tensor(
    data: Any,
    dtype: DtypeLike | torch.dtype = None,
    device: None | str | torch.device = None,
    wrap_sequence: bool = False,
    track_meta: bool = False,
    safe: bool = False,
) -> Any:
    """
    Utility to convert the input data to a PyTorch Tensor, if `track_meta` is True, the output will be a `MetaTensor`,
    otherwise, the output will be a regular torch Tensor.
    If passing a dictionary, list or tuple, recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.
        track_meta: whether to track the meta information, if `True`, will convert to `MetaTensor`.
            default to `False`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[tensor(0), tensor(244)]`.
            If `True`, then `[256, -12]` -> `[tensor(255), tensor(0)]`.

    """

    def _convert_tensor(tensor: Any, **kwargs: Any) -> Any:
        if not isinstance(tensor, torch.Tensor):
            # certain numpy types are not supported as being directly convertible to Pytorch tensors
            if isinstance(tensor, np.ndarray) and tensor.dtype in UNSUPPORTED_TYPES:
                tensor = tensor.astype(UNSUPPORTED_TYPES[tensor.dtype])

            # if input data is not Tensor, convert it to Tensor first
            tensor = torch.as_tensor(tensor, **kwargs)
        if track_meta and not isinstance(tensor, monai.data.MetaTensor):
            return monai.data.MetaTensor(tensor)
        if not track_meta and isinstance(tensor, monai.data.MetaTensor):
            return tensor.as_tensor()
        return tensor

    if safe:
        data = safe_dtype_range(data, dtype)
    dtype = get_equivalent_dtype(dtype, torch.Tensor)
    if isinstance(data, torch.Tensor):
        return _convert_tensor(data).to(dtype=dtype, device=device, memory_format=torch.contiguous_format)
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)
            return _convert_tensor(data, dtype=dtype, device=device)
    elif (has_cp and isinstance(data, cp_ndarray)) or isinstance(data, (float, int, bool)):
        return _convert_tensor(data, dtype=dtype, device=device)
    elif isinstance(data, list):
        list_ret = [convert_to_tensor(i, dtype=dtype, device=device, track_meta=track_meta) for i in data]
        return _convert_tensor(list_ret, dtype=dtype, device=device) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_tensor(i, dtype=dtype, device=device, track_meta=track_meta) for i in data)
        return _convert_tensor(tuple_ret, dtype=dtype, device=device) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device, track_meta=track_meta) for k, v in data.items()}

    return data


def safe_dtype_range(data: Any, dtype: DtypeLike | torch.dtype = None) -> Any:
    """
    Utility to safely convert the input data to target dtype.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert to target dtype and keep the original type.
            for dictionary, list or tuple, convert every item.
        dtype: target data type to convert.
    """

    def _safe_dtype_range(data, dtype):
        output_dtype = dtype if dtype is not None else data.dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        if data.ndim == 0:
            data_bound = (data, data)
        else:
            if isinstance(data, torch.Tensor):
                data_bound = (torch.min(data), torch.max(data))
            else:
                data_bound = (np.min(data), np.max(data))
        if (data_bound[1] > dtype_bound_value[1]) or (data_bound[0] < dtype_bound_value[0]):
            if isinstance(data, torch.Tensor):
                return torch.clamp(data, dtype_bound_value[0], dtype_bound_value[1])
            elif isinstance(data, np.ndarray):
                return np.clip(data, dtype_bound_value[0], dtype_bound_value[1])
            elif has_cp and isinstance(data, cp_ndarray):
                return cp.clip(data, dtype_bound_value[0], dtype_bound_value[1])
        else:
            return data

    if has_cp and isinstance(data, cp_ndarray):
        return cp.asarray(_safe_dtype_range(data, dtype))
    elif isinstance(data, np.ndarray):
        return np.asarray(_safe_dtype_range(data, dtype))
    elif isinstance(data, torch.Tensor):
        return _safe_dtype_range(data, dtype)
    elif isinstance(data, (float, int, bool)) and dtype is None:
        return data
    elif isinstance(data, (float, int, bool)) and dtype is not None:
        output_dtype = dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        data = dtype_bound_value[1] if data > dtype_bound_value[1] else data
        data = dtype_bound_value[0] if data < dtype_bound_value[0] else data
        return data

    elif isinstance(data, list):
        return [safe_dtype_range(i, dtype=dtype) for i in data]
    elif isinstance(data, tuple):
        return tuple(safe_dtype_range(i, dtype=dtype) for i in data)
    elif isinstance(data, dict):
        return {k: safe_dtype_range(v, dtype=dtype) for k, v in data.items()}
    return data

def get_dtype_bound_value(dtype: DtypeLike | torch.dtype) -> tuple[float, float]:
    """
    Get dtype bound value
    Args:
        dtype: dtype to get bound value
    Returns:
        (bound_min_value, bound_max_value)
    """
    if dtype in UNSUPPORTED_TYPES:
        is_floating_point = False
    else:
        is_floating_point = get_equivalent_dtype(dtype, torch.Tensor).is_floating_point
    dtype = get_equivalent_dtype(dtype, np.array)
    if is_floating_point:
        return (np.finfo(dtype).min, np.finfo(dtype).max)  # type: ignore
    else:
        return (np.iinfo(dtype).min, np.iinfo(dtype).max)

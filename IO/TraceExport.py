"""Export fluorescence measurements and their source metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


_COMPONENT_KEYS = {
    "roi": "traces_cell",
    "background": "traces_bg",
    "final": "traces_final",
}


def export_trace_measurements(
    measurements: Mapping[str, Any],
    path: str | Path,
    *,
    components: Iterable[str] = ("roi", "background", "final"),
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write selected (N, T) traces to NPY, MAT or HDF5 without rescaling.

    NPY writes one suffixed array per component and a shared JSON file.
    MAT and HDF5 store arrays and metadata_json in a single file.
    Return the paths of all written files.
    """
    destination = Path(path).expanduser()
    suffix = destination.suffix.lower()
    if suffix not in {".npy", ".mat", ".h5"}:
        raise ValueError("Output extension must be .npy, .mat or .h5.")
    if isinstance(components, str):
        components = (components,)
    selected = tuple(dict.fromkeys(components))
    if not selected or any(name not in _COMPONENT_KEYS for name in selected):
        raise ValueError("Select one or more of: roi, background, final.")

    metadata = dict(measurements["metadata"])
    rows = int(measurements["masks_final"].shape[0])
    frames, height, width = metadata["video_shape"]
    if measurements["masks_final"].shape != (rows, height * width):
        raise ValueError("Masks and measurement metadata must have matching shapes.")
    if metadata["row_indices"] != list(range(rows)):
        raise ValueError("Measurement rows must match the supplied final mask rows.")
    arrays = {}
    for component in selected:
        key = _COMPONENT_KEYS[component]
        array = np.asarray(measurements[key])
        if array.shape != (rows, frames) or array.dtype.kind not in "biuf":
            raise ValueError(f"{key} must be a real array with shape ({rows}, {frames}).")
        arrays[key] = array
    metadata["selected_components"] = list(selected)
    metadata["array_dtypes"] = {key: str(array.dtype) for key, array in arrays.items()}
    metadata_json = json.dumps(metadata, indent=2, allow_nan=False)

    if suffix == ".npy":
        paths = tuple(destination.with_name(f"{destination.stem}_{name}.npy") for name in selected)
        metadata_path = destination.with_name(f"{destination.stem}_metadata.json")
        outputs = (*paths, metadata_path)
    else:
        outputs = (destination,)
    if not overwrite:
        for output in outputs:
            if output.exists():
                raise FileExistsError(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".npy":
        for component, output in zip(selected, paths):
            with output.open("wb" if overwrite else "xb") as handle:
                np.save(handle, arrays[_COMPONENT_KEYS[component]], allow_pickle=False)
        with metadata_path.open("w" if overwrite else "x", encoding="utf-8") as handle:
            handle.write(metadata_json + "\n")
    elif suffix == ".mat":
        from scipy.io import savemat

        with destination.open("wb" if overwrite else "xb") as handle:
            savemat(handle, {**arrays, "metadata_json": metadata_json},
                    do_compression=True, long_field_names=True)
    else:
        import h5py

        with h5py.File(destination, "w" if overwrite else "x") as handle:
            handle.attrs["metadata_json"] = metadata_json
            for key, array in arrays.items():
                options = {"compression": "gzip", "shuffle": True} if array.size else {}
                handle.create_dataset(key, data=array, **options)
    return outputs


__all__ = ["export_trace_measurements"]

"""Validate TIFF measurement units, alignment and export fidelity."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

from IO.TraceExport import export_trace_measurements
from PostSlice.PostProcess_Pipeline import Post_FACT_guided_backgroundremoving
from PostSlice.TraceMeasurement import measure_raw_tiff_traces


def sample_data():
    frames, height, width = 20, 32, 32
    planes = np.zeros((4, height, width), dtype=bool)
    planes[0, 7:11, 7:11] = True
    planes[1, 8:12, 9:13] = True
    planes[2, 21:25, 22:26] = True
    background = 1000.0 + np.arange(frames) * 4.0
    video = np.broadcast_to(background[:, None, None], (frames, height, width)).copy()
    for index in range(3):
        signal = 100 + 30 * np.sin(np.arange(frames) * 0.7 + index)
        video[:, planes[index]] += signal[:, None]
    inference = np.zeros(video.shape, dtype=np.uint8)
    inference[:10, planes[0]] = 1
    inference[10:, planes[1]] = 1
    inference[:, planes[2]] = 1
    return sp.csr_matrix(planes.reshape(4, -1)), video, inference, planes, background


class TraceMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.masks, self.video, self.inference, self.planes, self.background = sample_data()

    def test_pixel_means_and_fixed_subtraction_retain_original_units(self):
        result = measure_raw_tiff_traces(
            self.masks, self.video, run_overlap_demix=False,
            use_nnls=False, alpha_default=1.4, alpha_max=0.1,
        )
        for index in range(3):
            expected = self.video[:, self.planes[index]].mean(axis=1)
            np.testing.assert_allclose(result["traces_cell"][index], expected, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(result["traces_bg"][index], self.background)
        expected_final = (result["traces_cell"] - 1.4 * result["traces_bg"]).astype(np.float32)
        np.testing.assert_array_equal(result["traces_final"], expected_final)
        np.testing.assert_array_equal(result["alpha_nnls"], np.full(4, 1.4))
        np.testing.assert_array_equal(result["traces_cell"][3], np.zeros(20))
        self.assertEqual(result["metadata"]["row_indices"], [0, 1, 2, 3])
        self.assertEqual(result["metadata"]["units"], "raw_intensity")
        self.assertEqual(result["metadata"]["coefficient_status"], ["fixed"] * 4)

    def test_automatic_coefficient_cap_and_empty_ring_fallback(self):
        result = measure_raw_tiff_traces(
            self.masks, self.video, run_overlap_demix=False, alpha_max=0.25,
        )
        np.testing.assert_array_equal(result["alpha_nnls"], np.full(4, 0.25))
        self.assertEqual(result["metadata"]["coefficient_status"][-1], "fallback")
        whole_image = sp.csr_matrix(np.ones((1, 32 * 32)))
        empty_ring = measure_raw_tiff_traces(
            whole_image, self.video, run_overlap_demix=False, alpha_max=0.25,
        )
        np.testing.assert_array_equal(empty_ring["traces_bg"], np.zeros((1, 20)))
        self.assertEqual(empty_ring["metadata"]["empty_background"], [True])

    def test_overlap_matches_core_and_preserves_inputs_and_roi_stage(self):
        video_before = self.video.copy()
        masks_before = self.masks.copy()
        inference_before = self.inference.copy()
        uncorrected = measure_raw_tiff_traces(self.masks, self.video, run_overlap_demix=False)
        result = measure_raw_tiff_traces(self.masks, self.video, infer_mask3d=self.inference)
        expected = Post_FACT_guided_backgroundremoving(
            self.masks.copy(), uncorrected["traces_cell"].copy(), np.zeros(4, dtype=bool),
            infer_mask3d=self.inference, input_img=self.video, mask_shape_2d=(32, 32),
        )
        for key in ("traces_cell", "traces_bg", "traces_final", "alpha_nnls", "sources"):
            np.testing.assert_array_equal(result[key], expected[key])
        np.testing.assert_array_equal(result["traces_cell"], uncorrected["traces_cell"])
        self.assertFalse(np.array_equal(result["traces_final"][:2], uncorrected["traces_final"][:2]))
        self.assertEqual(result["metadata"]["overlap_applied"], [True, True, False, False])
        np.testing.assert_array_equal(self.video, video_before)
        np.testing.assert_array_equal(self.inference, inference_before)
        np.testing.assert_array_equal(self.masks.toarray(), masks_before.toarray())
        np.testing.assert_array_equal(result["masks_final"].toarray(), masks_before.toarray())

    def test_measurement_does_not_reuse_guided_trace_overrides(self):
        guided = np.full((4, 20), 0.2)
        normalized = self.video / self.video.max()
        pipeline = Post_FACT_guided_backgroundremoving(
            self.masks.copy(), guided, np.ones(4, dtype=bool),
            infer_mask3d=self.inference, input_img=normalized, mask_shape_2d=(32, 32),
        )
        before = {key: value.copy() for key, value in pipeline.items() if hasattr(value, "copy")}
        raw = measure_raw_tiff_traces(pipeline["masks_final"], self.video, infer_mask3d=self.inference)
        self.assertGreater(raw["traces_cell"][0].min(), 1000)
        self.assertEqual(raw["metadata"]["extraction_method"], "mask_mean")
        np.testing.assert_array_equal(raw["override_used"], np.zeros(4, dtype=bool))
        for key, value in before.items():
            if sp.issparse(value):
                np.testing.assert_array_equal(pipeline[key].toarray(), value.toarray())
            else:
                np.testing.assert_array_equal(pipeline[key], value)

    def test_dense_and_sparse_masks_match(self):
        dense = self.masks.toarray()
        first = measure_raw_tiff_traces(dense, self.video, run_overlap_demix=False)
        second = measure_raw_tiff_traces(self.masks, self.video, run_overlap_demix=False)
        for key in ("traces_cell", "traces_bg", "traces_final"):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(dense, self.masks.toarray())

    def test_empty_masks_preserve_frame_count(self):
        for overlap in (False, True):
            result = measure_raw_tiff_traces(
                sp.csr_matrix((0, 32 * 32)), self.video,
                infer_mask3d=self.inference, run_overlap_demix=overlap,
            )
            for key in ("traces_cell", "traces_bg", "traces_final"):
                self.assertEqual(result[key].shape, (0, 20))
            self.assertEqual(result["metadata"]["row_indices"], [])

    def test_rejects_unaligned_or_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            measure_raw_tiff_traces(self.masks, self.video)
        with self.assertRaises(ValueError):
            measure_raw_tiff_traces(self.masks, self.video, infer_mask3d=self.inference[:-1])
        with self.assertRaises(ValueError):
            measure_raw_tiff_traces(self.masks[:, :-1], self.video, run_overlap_demix=False)
        bad_video = self.video.copy()
        bad_video[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            measure_raw_tiff_traces(self.masks, bad_video, run_overlap_demix=False)


class TraceExportTests(unittest.TestCase):
    def setUp(self):
        masks, video, inference, _, _ = sample_data()
        self.measurements = measure_raw_tiff_traces(masks, video, infer_mask3d=inference)
        self.directory = tempfile.TemporaryDirectory(prefix="fact-trace-export-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def assert_array_fidelity(self, loaded):
        for key in ("traces_cell", "traces_bg", "traces_final"):
            expected = self.measurements[key]
            np.testing.assert_array_equal(loaded[key], expected)
            self.assertEqual(loaded[key].dtype, expected.dtype)

    def test_numpy_roundtrip_with_metadata(self):
        metadata_before = json.dumps(self.measurements["metadata"], sort_keys=True)
        paths = export_trace_measurements(self.measurements, self.root / "raw.npy")
        self.assertEqual(len(paths), 4)
        loaded = {key: np.load(self.root / f"raw_{component}.npy", allow_pickle=False)
                  for component, key in (("roi", "traces_cell"), ("background", "traces_bg"),
                                         ("final", "traces_final"))}
        self.assert_array_fidelity(loaded)
        metadata = json.loads(paths[-1].read_text(encoding="utf-8"))
        self.assertEqual(metadata["row_indices"], [0, 1, 2, 3])
        self.assertEqual(metadata["source"], "raw")
        self.assertEqual(metadata["selected_components"], ["roi", "background", "final"])
        self.assertEqual(json.dumps(self.measurements["metadata"], sort_keys=True), metadata_before)

    def test_matlab_roundtrip_with_metadata(self):
        paths = export_trace_measurements(self.measurements, self.root / "raw.mat")
        loaded = loadmat(paths[0])
        self.assert_array_fidelity(loaded)
        metadata = json.loads(str(loaded["metadata_json"][0]))
        self.assertEqual(metadata["units"], "raw_intensity")
        self.assertEqual(metadata["coefficients"], self.measurements["alpha_nnls"].tolist())

    def test_hdf5_roundtrip_with_metadata(self):
        import h5py

        paths = export_trace_measurements(self.measurements, self.root / "raw.h5")
        with h5py.File(paths[0], "r") as handle:
            self.assert_array_fidelity({key: handle[key][...] for key in handle})
            metadata = json.loads(handle.attrs["metadata_json"])
        self.assertEqual(metadata["extraction_method"], "mask_mean")
        self.assertEqual(metadata["overlap_applied"], [True, True, False, False])

    def test_roi_selection_and_existing_file_protection(self):
        paths = export_trace_measurements(self.measurements, self.root / "raw.npy", components="roi")
        self.assertEqual({p.name for p in paths}, {"raw_roi.npy", "raw_metadata.json"})
        original = paths[0].read_bytes()
        with self.assertRaises(FileExistsError):
            export_trace_measurements(self.measurements, self.root / "raw.npy")
        self.assertEqual(paths[0].read_bytes(), original)
        self.assertFalse((self.root / "raw_background.npy").exists())
        export_trace_measurements(self.measurements, self.root / "raw.npy", components="roi", overwrite=True)
        self.assertEqual(paths[0].read_bytes(), original)

    def test_empty_measurements_export_in_all_formats(self):
        import h5py

        self.measurements = measure_raw_tiff_traces(
            sp.csr_matrix((0, 16)), np.ones((3, 4, 4)), run_overlap_demix=False,
        )
        for suffix in ("npy", "mat", "h5"):
            paths = export_trace_measurements(self.measurements, self.root / f"empty.{suffix}")
            if suffix == "npy":
                loaded = {key: np.load(self.root / f"empty_{component}.npy")
                          for component, key in (("roi", "traces_cell"), ("background", "traces_bg"),
                                                 ("final", "traces_final"))}
            elif suffix == "mat":
                loaded = loadmat(paths[0])
            else:
                with h5py.File(paths[0], "r") as handle:
                    loaded = {key: handle[key][...] for key in handle}
            self.assert_array_fidelity(loaded)

    def test_invalid_selection_and_alignment_write_nothing(self):
        for components in ((), ("unknown",)):
            with self.assertRaises(ValueError):
                export_trace_measurements(self.measurements, self.root / "bad.npy", components=components)
        damaged = dict(self.measurements, traces_cell=np.zeros((3, 20)))
        with self.assertRaises(ValueError):
            export_trace_measurements(damaged, self.root / "bad.npy")
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

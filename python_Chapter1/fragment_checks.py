"""GPU correctness checks embedded directly into the fragment-shader notebook."""
import unittest

import numpy as np

from fragment_fdtd import FragmentTM, TMConfig, TMReference


class FragmentTests(unittest.TestCase):
    context = None

    @classmethod
    def setUpClass(cls):
        if cls.context is None:
            raise unittest.SkipTest("An NVIDIA EGL context is required")

    def assert_fields_close(self, gpu, reference):
        e, h = gpu.read_fields()
        self.assertEqual(e.dtype, np.dtype("float32"))
        self.assertEqual(h.dtype, np.dtype("float32"))
        for actual, expected in ((e, reference.e), (h, reference.h)):
            for channel in range(actual.shape[-1]):
                scale = max(float(np.max(np.abs(expected[..., channel]))), 1e-8)
                relative_max_error = float(np.max(np.abs(actual[..., channel] - expected[..., channel]))) / scale
                self.assertLess(relative_max_error, 5e-5)
        self.assertTrue(np.all(e[[0, -1]] == 0))
        self.assertTrue(np.all(e[:, [0, -1]] == 0))

    def test_random_stencil_both_boundaries(self):
        rng = np.random.default_rng(19)
        for boundary in ("pec", "pml"):
            with self.subTest(boundary=boundary):
                config = TMConfig(32, boundary)
                reference = TMReference(config)
                reference.e[:] = rng.normal(size=reference.e.shape).astype(np.float32)
                reference.e[..., 3] = 0
                if boundary == "pec":
                    reference.e[..., 1:3] = 0
                reference.e[[0, -1]] = 0
                reference.e[:, [0, -1]] = 0
                reference.h[:] = rng.normal(scale=.01, size=reference.h.shape).astype(np.float32)
                reference.h[:, -1, 0] = reference.h[-1, :, 1] = 0
                gpu = FragmentTM(self.context, config)
                try:
                    gpu.e[0].write(reference.e.tobytes())
                    gpu.h[0].write(reference.h.tobytes())
                    gpu.step()
                    reference.step()
                    self.assert_fields_close(gpu, reference)
                finally:
                    gpu.release()

    def test_complete_pulse_trajectories(self):
        for boundary in ("pec", "pml"):
            with self.subTest(boundary=boundary):
                config = TMConfig(50, boundary)
                reference = TMReference(config)
                gpu = FragmentTM(self.context, config)
                try:
                    for step in range(config.default_steps):
                        reference.step()
                        gpu.step()
                        if (step + 1) % 37 == 0:
                            self.assert_fields_close(gpu, reference)
                    self.assert_fields_close(gpu, reference)
                finally:
                    gpu.release()

    def test_symmetry_and_absorption(self):
        fields = {}
        for boundary in ("pec", "pml"):
            gpu = FragmentTM(self.context, TMConfig(50, boundary))
            try:
                for _ in range(210):
                    gpu.step()
                e, h = gpu.read_fields()
                self.assertTrue(np.isfinite(e).all() and np.isfinite(h).all())
                fields[boundary] = e[..., 0]
            finally:
                gpu.release()
        np.testing.assert_allclose(fields["pec"], fields["pec"].T, atol=2e-4)
        np.testing.assert_allclose(fields["pec"], fields["pec"][::-1], atol=2e-4)
        core = np.s_[8:-8, 8:-8]
        self.assertLess(np.linalg.norm(fields["pml"][core]), .1 * np.linalg.norm(fields["pec"][core]))

    def test_reset_reproducibility(self):
        gpu = FragmentTM(self.context, TMConfig(32))
        try:
            for _ in range(20):
                gpu.step()
            expected = gpu.read_fields()
            gpu.reset()
            self.assertEqual(gpu.n, 0)
            self.assertTrue(all(np.all(array == 0) for array in gpu.read_fields()))
            for _ in range(20):
                gpu.step()
            for actual, initial in zip(gpu.read_fields(), expected):
                np.testing.assert_array_equal(actual, initial)
        finally:
            gpu.release()

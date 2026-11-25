import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cell_point import CellPointData
from track_particles import CellPointFlow, sample_surface


class CellPointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        points = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [-1, 0, 0]])
        np.savetxt(self.root / "points.dat", points)
        np.savetxt(self.root / "tets.dat", [[3, 0, 2, 1], [4, 0, 1, 2]], fmt="%d")
        (self.root / "patches").mkdir()
        np.savetxt(self.root / "patches/inlet.dat", [[3, 1, 2]], fmt="%d")
        for t, values in [(0, points*2 + 1), (1, np.ones_like(points)*7)]:
            (self.root / str(t)).mkdir()
            np.savetxt(self.root / str(t) / "U.dat", values)
        self.flow = CellPointFlow.__new__(CellPointFlow)
        self.flow.times = np.array([0., 1.])
        self.flow.sampler = CellPointData(self.root, self.flow.times)

    def test_near_face_shared_face_and_outside(self):
        xyz = np.array([[-1e-6, .2, .2], [1e-6, .2, .2], [0., .2, .2], [2., .2, .2]])
        ids, weights = self.flow.sampler.locate(xyz)
        np.testing.assert_array_equal(ids, [1, 0, 0, -1])
        np.testing.assert_array_equal(self.flow.sampler.locate(xyz[::-1])[0][::-1], ids)
        np.testing.assert_allclose(weights[:3].sum(axis=1), 1.)
        for t, expected in [(0., xyz*2 + 1), (1., np.full_like(xyz, 7.))]:
            velocity, valid = self.flow.sample(t, xyz)
            np.testing.assert_array_equal(valid, ids >= 0)
            np.testing.assert_allclose(velocity[valid], expected[valid])
            np.testing.assert_array_equal(velocity[~valid], 0.)

    def test_transient_release_reuses_stencil(self):
        xyz = np.array([[.1, .2, .3], [-.1, .2, .3], [.1, .3, .1]])
        t = np.array([.25, .5, .75])
        sampler = self.flow.sampler
        with patch.object(sampler, "locate", wraps=sampler.locate) as locate:
            velocity, valid = self.flow.sample_release(t, xyz)
            self.assertEqual(locate.call_count, 1)
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(velocity, (1-t[:, None])*(xyz*2+1) + t[:, None]*7)

    def test_native_inlet_and_missing_velocity(self):
        sampler = self.flow.sampler
        points = sample_surface(sampler.patch_surface("inlet"), 32, np.random.default_rng(27))
        np.testing.assert_allclose(points.sum(axis=1), 1.)
        np.testing.assert_array_equal(self.flow.sample(0., points)[1], True)
        with self.assertRaisesRegex(ValueError, "Missing cellPoint export"):
            CellPointData(self.root, [2.])

    def test_inverted_tet_alone_cannot_establish_membership(self):
        # Reverse the positive-x tet. Its geometric hull still covers the
        # first probe point, but its outward half-spaces do not contain it.
        np.savetxt(self.root / "tets.dat", [[3, 0, 1, 2], [4, 0, 1, 2]], fmt="%d")
        sampler = CellPointData(self.root, [0.])
        ids, _ = sampler.locate(np.array([[.1, .2, .2], [-.1, .2, .2]]))
        np.testing.assert_array_equal(ids, [-1, 1])


if __name__ == "__main__":
    unittest.main()

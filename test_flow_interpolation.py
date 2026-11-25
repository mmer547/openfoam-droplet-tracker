import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from track_particles import Flow


class TransientInterpolationTest(unittest.TestCase):
    def test_shared_probe_matches_two_independent_spatial_probes(self):
        points = vtk.vtkPoints()
        for point in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
            points.InsertNextPoint(point)
        grids = []
        for t in (0, 1, 2):
            grid = vtk.vtkUnstructuredGrid()
            grid.SetPoints(points)
            grid.InsertNextCell(vtk.VTK_TETRA, 4, (0, 1, 2, 3))
            grid.InsertNextCell(vtk.VTK_TETRA, 4, (1, 2, 3, 4))
            values = (1+t)*np.arange(15, dtype=np.float32).reshape(5, 3) + t*t
            velocity = numpy_to_vtk(values, deep=True)
            velocity.SetName("U")
            grid.GetPointData().AddArray(velocity)
            grids.append(grid)

        flow = Flow.__new__(Flow)
        flow.times = np.array([0., 1., 2.])
        flow.field = grids.__getitem__
        flow.interval_index = None
        xyz = np.array([[.2, .2, .1], [.7, .6, .7], [2., 1., 1.]])
        # Unsorted release times span two intervals; the last point is outside.
        times = np.array([1.25, .125, 1.75])
        actual, ok = flow.sample_release(times, xyz)
        expected = []
        for point, t in zip(xyz, times):
            i = int(t)
            left, left_ok = flow.sample(float(i), point[None, :])
            right, right_ok = flow.sample(float(i+1), point[None, :])
            expected.append((left + (t-i)*(right-left))[0])
            self.assertEqual(left_ok[0], right_ok[0])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(ok, [True, True, False])

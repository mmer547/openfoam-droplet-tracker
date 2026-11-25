import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from track_particles import Boundaries, Flow, triangles


def flow_for(*fields):
    flow = Flow.__new__(Flow)
    flow.times = np.arange(len(fields), dtype=float)
    flow.field = Mock(side_effect=fields.__getitem__)
    flow.sampler = None
    return flow


def boxes():
    image = vtk.vtkImageData()
    image.SetDimensions(3, 2, 2)
    convert = vtk.vtkAppendFilter()
    convert.AddInputData(image)
    convert.Update()
    grid = convert.GetOutput()
    values = numpy_to_vtk(vtk_to_numpy(grid.GetPoints().GetData()).copy())
    values.SetName("U")
    grid.GetPointData().SetVectors(values)
    return grid


class SamplingTests(unittest.TestCase):
    def test_cyclone_particle_142_and_warped_cells(self):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(str(Path(__file__).parent / "data/cyclone-cells.vtk"))
        reader.Update()
        flow = flow_for(reader.GetOutput())
        # IDs 1, 10 and 142 at 2.375 s, followed by a former outside endpoint.
        points = np.array([
            [0.23606594201117337, -0.2669432599968436, -0.39384903289337414],
            [-0.11450145175674284, 0.3736877654451745, -0.3095868502557231],
            [-0.10383513616240979, -0.4131101213659698, 0.41498745983130725],
            [0.24271168, 0.44973323, 0.63122562],
        ])
        velocity, valid = flow.sample(0., points)
        np.testing.assert_array_equal(valid, True)
        self.assertEqual(velocity.dtype, np.float32)
        self.assertEqual([flow.sampler.locate(p) for p in points], [0, 1, 2, 3])
        np.testing.assert_allclose(velocity[2], [7.326304, -1.851917, -2.005562], atol=1e-6)
        for order in ([3, 2, 1, 0], [2], [0, 1], [2, 3]):
            batch, mask = flow.sample(0., points[order])
            np.testing.assert_array_equal(batch, velocity[order])
            np.testing.assert_array_equal(mask, valid[order])
        grid = reader.GetOutput()
        vtk_to_numpy(grid.GetPointData().GetArray("U"))[:] = vtk_to_numpy(grid.GetPoints().GetData())
        np.testing.assert_allclose(flow.sample(0., points[-1:])[0], points[-1:], atol=3e-8)

    def test_shared_faces_edges_inward_release_and_exterior(self):
        flow = flow_for(boxes())
        points = np.array([[0, .4, .6], [1e-6, .4, .6], [1, .3, .2],
                           [1, 0, 0], [1.1, .3, .2], [-1e-6, .4, .6]])
        velocity, valid = flow.sample_release(np.zeros(len(points)), points)
        np.testing.assert_array_equal(valid, [True]*5 + [False])
        np.testing.assert_allclose(velocity[valid], points[valid], atol=1e-14)
        self.assertEqual([flow.sampler.locate(p) for p in points], [0, 0, 0, 0, 1, -1])

    def test_spatial_and_temporal_interpolation_at_release(self):
        first = boxes()
        second = vtk.vtkUnstructuredGrid()
        second.DeepCopy(first)
        vtk_to_numpy(second.GetPointData().GetArray("U"))[:] += 4
        flow = flow_for(first, second)
        points = np.array([[.2, .1, .5], [1.6, .3, .7], [1, .4, .2]])
        velocity, valid = flow.sample(.25, points)
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(velocity, points + 1)
        times = np.array([0., .25, 1.])
        velocity, valid = flow.sample_release(times, points)
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(velocity, points + 4*times[:, None])

    def test_concave_polyhedron_rejects_gap_inside_bounds(self):
        # Extrude an L-shaped polygon. The missing corner is inside its bounding box.
        ring = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        points = vtk.vtkPoints()
        for z in (0, 1):
            for x, y in ring:
                points.InsertNextPoint(x, y, z)
        faces = vtk.vtkCellArray()
        faces.InsertNextCell(6, list(reversed(range(6))))
        faces.InsertNextCell(6, list(range(6, 12)))
        for i in range(6):
            j = (i+1) % 6
            faces.InsertNextCell(4, [i, j, j+6, i+6])
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.InsertNextCell(vtk.VTK_POLYHEDRON, 12, list(range(12)), faces)
        values = numpy_to_vtk(np.ones((12, 3)))
        values.SetName("U")
        grid.GetPointData().SetVectors(values)
        coords = np.array([[.5, 1.5, .5], [1.5, .5, .5], [1.5, 1.5, .5]])
        velocity, valid = flow_for(grid).sample(0., coords)
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_allclose(velocity[valid], 1.)
        surface = vtk.vtkPolyData()
        surface.SetPoints(points)
        surface.SetPolys(faces)
        boundary = Boundaries({"wall": triangles(surface)}, {"wall"}, set())
        self.assertEqual(boundary.count_enclosed(coords), 2)


if __name__ == "__main__":
    unittest.main()

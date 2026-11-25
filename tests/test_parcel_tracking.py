import tempfile
import unittest
from pathlib import Path

import numpy as np

from cell_point import CellPointData
from parcel_tracking import ParcelTracker


def two_boxes(path):
    """Two unit cubes joined at x=1, each split from its centre to its faces."""
    points = np.array([(x,y,z) for x in range(3) for y in range(2) for z in range(2)]
                      + [(.5,.5,.5), (1.5,.5,.5)])
    patches = {"inlet": [], "outlet": [], "walls": []}
    tets = []
    for cell in range(2):
        centre = 12 + cell
        for axis in range(3):
            for upper in (0, 1):
                corners = []
                for a, b in ((0,0), (1,0), (1,1), (0,1)):
                    xyz = [cell, 0, 0]
                    xyz[axis] += upper
                    xyz[(axis+1)%3] += a
                    xyz[(axis+2)%3] += b
                    corners.append(4*xyz[0] + 2*xyz[1] + xyz[2])
                for i in (1, 2):
                    face = [corners[0], corners[i], corners[i+1]]
                    if not upper:
                        face.reverse()
                    tets.append([centre, *face])
                    if axis:
                        patches["walls"].append(face)
                    elif cell+upper != 1:
                        patches["outlet" if upper else "inlet"].append(face)
    (path/"0").mkdir()
    (path/"patches").mkdir()
    np.savetxt(path/"points.dat", points)
    np.savetxt(path/"tets.dat", tets, fmt="%d")
    np.savetxt(path/"volumes.dat", [1., 1.])
    for name, faces in patches.items():
        np.savetxt(path/"patches"/(name+".dat"), faces, fmt="%d")
    np.savetxt(path/"0/U.dat", np.tile([4., 0., 0.], (len(points), 1)))
    return CellPointData(path, [0.])


class ParcelTrackingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = two_boxes(Path(tmp.name))
        self.tracker = ParcelTracker(self.data, ["inlet", "outlet", "walls"], 2,
                                     1e-4, 1000., 1., 1e-5, [0.,0.,0.], 1.)

    def test_inward_release_crosses_multiple_cells_and_exits_once(self):
        ids = np.arange(2)
        self.tracker.release(ids, np.array([[0., .23, .56], [.5, .5, .5]]),
                             self.data, [4., 0., 0.])
        patch, fraction = self.tracker.advance(ids, .6, 0.)
        np.testing.assert_array_equal(patch, [1, 1])
        np.testing.assert_allclose(fraction*.6, [.5, .375])
        np.testing.assert_allclose(self.tracker.position(ids), [[2., .23, .56], [2., .5, .5]])

    def test_move_then_update_and_oblique_wall(self):
        ids = np.arange(2)
        self.tracker.release(ids, np.array([[.2,.3,.4], [.4,.999,.2]]), self.data, [1.,1.,0.])
        patch, fraction = self.tracker.advance(ids, .01, 0.)
        np.testing.assert_array_equal(patch, [-1, 2])
        np.testing.assert_allclose(self.tracker.position(ids), [[.21,.31,.4], [.401,1.,.2]])
        np.testing.assert_allclose(fraction, [1., .1])
        beta = 18.e-5/(1000*1e-8)*(1+(10*np.sqrt(10))**(2/3)/6)
        expected = [4-3*np.exp(-.01*beta), np.exp(-.01*beta), 0.]
        np.testing.assert_allclose(self.tracker.u[0], expected)

    def test_partial_first_step_on_shared_face(self):
        ids = np.array([0])
        self.tracker.release(ids, np.array([[1., .3, .5]]), self.data, [4.,0.,0.])
        patch, fraction = self.tracker.advance(ids, .2, .5)
        np.testing.assert_array_equal(patch, [-1])
        np.testing.assert_allclose(fraction, [1.])
        np.testing.assert_allclose(self.tracker.position(ids), [[1.4, .3, .5]])

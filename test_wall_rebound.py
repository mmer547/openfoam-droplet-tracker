from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np
import vtk

from track_particles import Boundaries, advance_particles, reflect_velocity, triangles


class UniformFlow:
    def sample_release(self, times, positions):
        return np.zeros_like(positions), np.ones(len(times), dtype=bool)


def plane(x, normal):
    source = vtk.vtkPlaneSource()
    source.SetOrigin(x, -1, -normal)
    source.SetPoint1(x, 1, -normal)
    source.SetPoint2(x, -1, normal)
    source.Update()
    return triangles(source.GetOutput())


class WallInteractionTests(unittest.TestCase):
    def step(self, rebounds, tau=1e20, duration=1.):
        boundaries = Boundaries({"left": plane(0, -1), "right": plane(1, 1)}, set(), {"left"})
        args = SimpleNamespace(diameter=None, tau=tau, rebounds=rebounds, flow_csv=True)
        events = []
        def record(t, xyz, ids, status, index):
            events.append((status, t[0], xyz[0].copy()))
        counts, crossings = Counter(), {"left": []}
        xyz, ids, speed = advance_particles(
            args, UniformFlow(), boundaries, np.array([17]), np.array([[.25, 0., 0.]]),
            np.zeros((1, 3)), np.array([[4., .2, 0.]]), np.array([0.]), duration,
            np.zeros(3), record, counts, crossings)
        return events, xyz, speed, crossings

    def test_multiple_impacts_in_one_step(self):
        events, xyz, speed, _ = self.step({"right": (1, 0), "left": (1, 0)})
        self.assertEqual([e[0] for e in events], ["rebounded"]*4)
        np.testing.assert_allclose([e[1] for e in events], [.1875, .4375, .6875, .9375])
        np.testing.assert_allclose(xyz, [[.25, .2, 0]], atol=1e-14)
        np.testing.assert_allclose(speed, [[4, .2, 0]], atol=1e-14)

    def test_damped_rebound_then_escape(self):
        events, xyz, _, crossings = self.step({"right": (.5, .2)})
        self.assertEqual([e[0] for e in events], ["rebounded", "escaped"])
        np.testing.assert_allclose(events[-1][2], [0, .1175, 0], atol=1e-14)
        np.testing.assert_allclose(crossings["left"], [[.6875]])
        self.assertEqual(len(xyz), 0)

    def test_drag_uses_speed_at_impact(self):
        events, _, speed, _ = self.step({"right": (.5, .2)}, tau=.5, duration=.4)
        self.assertEqual(len(events), 1)
        # Linear drag in still gas commutes with the linear rebound map.
        np.testing.assert_allclose(speed, np.array([[-2., .16, 0]]) * np.exp(-.4/.5))

    def test_tangential_damping_and_inward_motion(self):
        v = reflect_velocity(np.array([[-3., 2., 1.]]), np.array([[1., 0., 0.]]), 1, 1)
        np.testing.assert_array_equal(v, [[-3., 0., 0.]])

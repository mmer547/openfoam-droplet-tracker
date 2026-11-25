"""Read an exportCellPoint snapshot series from a static OpenFOAM case."""
from functools import lru_cache

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


class CellPointData:
    tolerance = 1e-15  # OpenFOAM v2606 double-precision cellPointWeight::tol

    def __init__(self, directory, times):
        self.directory = directory
        folders = {float(path.parent.name): path.parent for path in directory.glob("*/U.dat")
                   if path.parent.name != "patches"}
        missing = [t for t in times if t not in folders]
        if missing:
            raise ValueError(f"Missing cellPoint export for times {missing} in {directory}")
        self.files = [folders[t] / "U.dat" for t in times]
        points = np.loadtxt(directory / "points.dat", ndmin=2)
        tets = np.loadtxt(directory / "tets.dat", dtype=np.int64, ndmin=2)
        if (points.shape[1] != 3 or not np.isfinite(points).all() or not len(tets)
                or tets.shape[1] != 4 or tets.min() < 0 or tets.max() >= len(points)):
            raise ValueError("Invalid cellPoint geometry or tetrahedron connectivity")
        self.node_count = len(points)
        vertices = points[tets]
        matrix = (vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)
        det = np.linalg.det(matrix)
        # The first vertex is the OpenFOAM cell centre. Sum its signed tet volumes.
        volumes = np.bincount(tets[:, 0], weights=det/6)
        usable = np.abs(det) > self.tolerance * np.abs(volumes[tets[:, 0]])
        if not usable.any():
            raise ValueError("No non-degenerate cellPoint tetrahedra")
        self.tets = tets[usable]
        self.inverted = det[usable] < 0
        folded = np.unique(self.tets[self.inverted, 0])
        self.check_overlap = np.isin(self.tets[:, 0], folded) if len(folded) else None
        self.origins = points[self.tets[:, 0]]
        self.inverse = np.linalg.inv(matrix[usable])
        coords = vtk.vtkPoints()
        coords.SetData(numpy_to_vtk(points, deep=True))
        cells = vtk.vtkCellArray()
        offsets = np.arange(0, self.tets.size+1, 4, dtype=np.int64)
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(self.tets.ravel(), deep=True))
        mesh = vtk.vtkUnstructuredGrid()
        mesh.SetPoints(coords)
        mesh.SetCells(vtk.VTK_TETRA, cells)
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(mesh)
        self.locator.BuildLocator()
        self.padding = 64 * np.finfo(float).eps * max(mesh.GetLength(), np.abs(points).max())
        self.values = lru_cache(maxsize=2)(self._read_values)

    def _read_values(self, index):
        values = np.loadtxt(self.files[index], ndmin=2)
        if values.shape != (self.node_count, 3) or not np.isfinite(values).all():
            raise ValueError(f"Invalid cellPoint velocity field in {self.files[index]}")
        return values

    def patch_surface(self, name):
        # Seed directly on the exported double-precision boundary. A VTK reader
        # patch rounded to float32 may already lie outside the native tets.
        path = self.directory / "patches" / (name + ".dat")
        patch = vtk.vtkPolyData()
        if not path.stat().st_size:
            return patch
        faces = np.loadtxt(path, dtype=np.int64, ndmin=2)
        if faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= self.node_count:
            raise ValueError(f"Invalid cellPoint injection patch {name}")
        point_ids, local = np.unique(faces.ravel(), return_inverse=True)
        xyz = vtk_to_numpy(self.locator.GetDataSet().GetPoints().GetData())
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(xyz[point_ids], deep=True))
        cells = vtk.vtkCellArray()
        offsets = np.arange(0, faces.size+1, 3, dtype=np.int64)
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(local.astype(np.int64, copy=False), deep=True))
        patch.SetPoints(points)
        patch.SetPolys(cells)
        return patch

    def _weights(self, ids, xyz):
        face = np.einsum("nij,nj->ni", self.inverse[ids], xyz - self.origins[ids])
        return np.column_stack((1-face.sum(axis=1), face))

    def locate(self, xyz):
        ids = np.fromiter((self.locator.FindCell(p) for p in xyz), dtype=np.intp, count=len(xyz))
        weights = np.zeros((len(xyz), 4))
        found = ids >= 0
        weights[found] = self._weights(ids[found], xyz[found])
        candidates = vtk.vtkIdList()
        # VTK can accept a point just outside a tet. Resolve it (and shared faces)
        # with the exported connectivity and double-precision barycentric weights.
        review = ~found | (weights.min(axis=1) <= self.tolerance)
        if self.check_overlap is not None:
            review[found] |= self.check_overlap[ids[found]]
        for row in np.flatnonzero(review):
            point = xyz[row]
            box = [v for x in point for v in (x-self.padding, x+self.padding)]
            self.locator.FindCellsWithinBounds(box, candidates)
            near = np.array(sorted(candidates.GetId(j) for j in range(candidates.GetNumberOfIds())),
                            dtype=np.intp)
            w = self._weights(near, point)
            contains = (w >= -self.tolerance).all(axis=1)
            inside = np.flatnonzero(contains & ~self.inverted[near])
            if not len(inside):
                # tet::inside uses SMALL in mesh length units. A face weight
                # divided by its gradient magnitude is the signed face distance.
                grad = self.inverse[near]
                norms = np.column_stack((np.linalg.norm(grad.sum(axis=1), axis=1),
                                         np.linalg.norm(grad, axis=2)))
                inside = np.flatnonzero((w >= -self.tolerance * norms).all(axis=1)
                                        & ~self.inverted[near])
            if len(inside):
                first = inside[0]
                # In a non-star-shaped cell, use a positive tet for membership,
                # then follow cellPointWeight's exported order for interpolation.
                if self.check_overlap is not None and self.check_overlap[near[first]]:
                    centre = self.tets[near[first], 0]
                    eligible = np.flatnonzero(contains & (self.tets[near, 0] == centre))
                    if len(eligible):
                        first = eligible[0]
                ids[row], weights[row] = near[first], w[first]
            else:
                ids[row], weights[row] = -1, 0.
        return ids, weights

    def stencil(self, xyz):
        ids, weights = self.locate(xyz)
        valid = ids >= 0
        return (valid, np.repeat(np.flatnonzero(valid), 4),
                self.tets[ids[valid]].ravel(), weights[valid].ravel())

"""Offline particle tracking on a static, reconstructed OpenFOAM mesh (SI units)."""
import argparse
import csv
import gzip
import math
import re
import os
from contextlib import contextmanager
from functools import lru_cache
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


DEFAULT_CELL_CACHE_SIZE = 1024


class CellSampler:
    """Locate each point independently on the closed faces of a static mesh."""

    def __init__(self, grid, cache_size=DEFAULT_CELL_CACHE_SIZE):
        self.grid = vtk.vtkUnstructuredGrid()
        self.grid.CopyStructure(grid)
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(self.grid)
        self.locator.BuildLocator()
        self.tol = 64 * np.finfo(float).eps * max(grid.GetLength(), max(map(abs, grid.GetBounds())))
        self.candidates = vtk.vtkIdList()
        # Keep local geometry for active cells without retaining a locator per mesh cell.
        self.cell = lru_cache(maxsize=cache_size)(self._build_cell)

    def _build_cell(self, cid):
        source = self.grid.GetCell(cid)
        cell = source.NewInstance()
        cell.DeepCopy(source)
        ids = np.array([cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())])
        local = {pid: j for j, pid in enumerate(ids)}
        faces = vtk.vtkCellArray()
        reversed_faces = vtk.vtkUnsignedCharArray()
        reversed_faces.SetName("reversed")
        for j in range(cell.GetNumberOfFaces()):
            face = cell.GetFace(j)
            ring = [face.GetPointId(k) for k in range(face.GetNumberOfPoints())]
            if face.GetCellType() == vtk.VTK_PIXEL:
                ring[2], ring[3] = ring[3], ring[2]
            start = ring.index(min(ring))
            ring = ring[start:] + ring[:start]
            # Both neighbours triangulate a shared polygon in the same order.
            reverse = ring[-1] < ring[1]
            reversed_faces.InsertNextValue(int(reverse))
            if reverse:
                ring = ring[:1] + ring[:0:-1]
            faces.InsertNextCell(len(ring), [local[pid] for pid in ring])
        surface = vtk.vtkPolyData()
        surface.SetPoints(cell.GetPoints())
        surface.SetPolys(faces)
        surface.GetCellData().AddArray(reversed_faces)
        shell = triangles(surface)
        for i in np.flatnonzero(vtk_to_numpy(shell.GetCellData().GetArray("reversed"))):
            shell.GetPolys().ReverseCellAtId(i)
        connectivity = vtk_to_numpy(shell.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        points = vtk_to_numpy(cell.GetPoints().GetData()).astype(float)
        a, b, c = (points - points[0])[connectivity].transpose(1, 0, 2)
        # A negative signed volume means this cell's faces wind inward.
        if np.einsum("ij,ij->", a, np.cross(b, c)) < 0:
            for i in range(len(connectivity)):
                shell.GetPolys().ReverseCellAtId(i)
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(shell)
        return cell, ids, distance

    def locate(self, point):
        cid = self.locator.FindCell(point)
        if (cid >= 0 and self.locator.InsideCellBounds(point, cid)
                and self.cell(cid)[2].EvaluateFunction(point) < -self.tol):
            return cid
        # A warped cell can fail VTK's parametric test. Check all nearby shells.
        # Sorting also gives points on shared faces and edges a stable owner.
        box = [v for x in point for v in (x - self.tol, x + self.tol)]
        self.locator.FindCellsWithinBounds(box, self.candidates)
        for cid in sorted(self.candidates.GetId(i) for i in range(self.candidates.GetNumberOfIds())):
            if (self.locator.InsideCellBounds(point, cid, self.tol)
                    and self.cell(cid)[2].EvaluateFunction(point) <= self.tol):
                return cid
        return -1

    def stencil(self, xyz):
        rows, nodes, coeffs = [], [], []
        valid = np.zeros(len(xyz), dtype=bool)
        for i, point in enumerate(xyz):
            cid = self.locate(point)
            if cid < 0:
                continue
            cell, ids, _ = self.cell(cid)
            weights = [0.] * len(ids)
            result = cell.EvaluatePosition(point, [0.]*3, vtk.reference(0), [0.]*3,
                                           vtk.reference(0.), weights)
            # A warped cell can supply valid shape weights with parametric status 0.
            if result < 0:
                continue
            valid[i] = True
            rows.extend([i] * len(ids))
            nodes.extend(ids)
            coeffs.extend(weights)
        return valid, np.asarray(rows, dtype=int), np.asarray(nodes, dtype=int), np.asarray(coeffs)


class Flow:
    def __init__(self, marker, cell_cache_size=DEFAULT_CELL_CACHE_SIZE):
        self.cell_cache_size = cell_cache_size
        self.reader = vtk.vtkOpenFOAMReader()
        self.reader.SetFileName(str(marker))
        self.reader.SkipZeroTimeOff()
        self.reader.CreateCellToPointOn()
        self.reader.UpdateInformation()
        self.reader.DisableAllPatchArrays()
        self.reader.SetPatchArrayStatus("internalMesh", 1)
        self.reader.DisableAllCellArrays()
        self.reader.SetCellArrayStatus("U", 1)
        values = self.reader.GetTimeValues()
        if values is None or not values.GetNumberOfValues():
            raise ValueError("No saved times in case")
        self.times = vtk_to_numpy(values).copy()
        self.cache = {}
        self.sampler = None

    def boundary_surfaces(self):
        self.reader.EnableAllPatchArrays()
        try:
            self.reader.UpdateTimeStep(float(self.times[0]))
            blocks = self.reader.GetOutput().NewIterator()
            blocks.InitTraversal()
            patches = {}
            while not blocks.IsDoneWithTraversal():
                block = blocks.GetCurrentDataObject()
                if isinstance(block, vtk.vtkPolyData):
                    name = blocks.GetCurrentMetaData().Get(vtk.vtkCompositeDataSet.NAME())
                    patches[name] = triangles(block)
                blocks.GoToNextItem()
            return patches
        finally:
            self.reader.DisableAllPatchArrays()
            self.reader.SetPatchArrayStatus("internalMesh", 1)

    def field(self, index):
        if index not in self.cache:
            self.reader.UpdateTimeStep(float(self.times[index]))
            grid = self.reader.GetOutput().GetBlock(0)
            if not isinstance(grid, vtk.vtkUnstructuredGrid) or not grid.GetPointData().HasArray("U"):
                raise ValueError(f"Missing internalMesh or U at {self.times[index]}")
            copy = vtk.vtkUnstructuredGrid()
            copy.DeepCopy(grid)
            copy.GetPointData().SetActiveVectors("U")
            if len(self.cache) >= 2:
                del self.cache[min(self.cache)]
            self.cache[index] = copy
        return self.cache[index]

    def values(self, index):
        return vtk_to_numpy(self.field(index).GetPointData().GetArray("U"))

    def _sample_between(self, lower, upper, weight, xyz):
        if self.sampler is None:
            self.sampler = CellSampler(self.field(lower), self.cell_cache_size)
        valid, rows, nodes, coeffs = self.sampler.stencil(xyz)

        def probe(i):
            values = self.values(i)
            velocity = np.zeros((len(xyz), 3))
            np.add.at(velocity, rows, coeffs[:, None] * values[nodes])
            # Match the field precision produced by VTK's point-data interpolation.
            return velocity.astype(values.dtype, copy=False)
        left = probe(lower)
        if lower == upper:
            return left, valid
        return left + weight*(probe(upper)-left), valid

    def sample(self, t, xyz):
        j = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times)-1))
        if not j or t == self.times[j]:
            return self._sample_between(j, j, 0., xyz)
        weight = (t-self.times[j-1]) / (self.times[j]-self.times[j-1])
        return self._sample_between(j-1, j, weight, xyz)

    def sample_release(self, times, xyz):
        """Sample a batch at individual release times using the bracketing fields."""
        if times[0] == times[-1] or len(self.times) == 1:
            return self.sample(times[0], xyz)
        right = np.clip(np.searchsorted(self.times, times), 1, len(self.times)-1)
        velocity = np.empty_like(xyz)
        valid = np.empty(len(times), dtype=bool)
        for j in np.unique(right):
            selected = right == j
            weight = (times[selected] - self.times[j-1]) / (self.times[j] - self.times[j-1])
            velocity[selected], valid[selected] = self._sample_between(j-1, j, weight[:, None], xyz[selected])
        return velocity, valid


class CellPointFlow(Flow):
    def __init__(self, marker):
        super().__init__(marker)
        from cell_point import CellPointData
        self.sampler = CellPointData(Path(marker).parent / "postProcessing/trackerCellPoint", self.times)

    def values(self, index):
        return self.sampler.values(index)

    def boundary_surfaces(self):
        return {name: self.sampler.patch_surface(name) for name in super().boundary_surfaces()}


def triangles(surface):
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(surface)
    triangulate.PassLinesOff()
    triangulate.PassVertsOff()
    triangulate.Update()
    return triangulate.GetOutput()


def sample_surface(mesh, count, rng):
    if not mesh.GetNumberOfPolys():
        raise ValueError("Patch has no polygon faces")
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float, copy=False)
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    a, b, c = points[faces].transpose(1, 0, 2)
    ab, ac = b-a, c-a
    weights = np.linalg.norm(np.cross(ab, ac), axis=1)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Patch must have finite, positive surface area")
    face = rng.choice(len(faces), size=count, p=weights/total)
    # Reflect the upper half of the unit square into the reference triangle.
    uv = rng.random((count, 2))
    upper = uv.sum(axis=1) > 1
    uv[upper] = 1 - uv[upper]
    return a[face] + uv[:, :1]*ab[face] + uv[:, 1:]*ac[face]


STATUS = {"active": 0, "deposited": 1, "escaped": 2, "boundary": 3, "outside": 4}


def wall_names(case):
    # The boundary dictionary is textual even for binary polyMesh fields.
    path = case.parent / "constant/polyMesh/boundary"
    if path.exists():
        data = path.read_bytes()
    elif path.with_suffix(".gz").exists():
        with gzip.open(path.with_suffix(".gz")) as f:
            data = f.read()
    else:
        return set()
    body = re.sub(r"/\*.*?\*/|//[^\n]*", " ", data.decode("latin1"), flags=re.S)
    return {name.strip('"') for name, fields in
            re.findall(r'([^\s(){};]+)\s*\{([^{}]*)\}', body)
            if re.search(r'(?:^|;)\s*type\s+wall\s*;', fields)}


class BoundaryGrid:
    """Conservative broad phase for batched short segments on a static mesh."""

    def __init__(self, a, b, c):
        lower = np.minimum(np.minimum(a, b), c)
        upper = np.maximum(np.maximum(a, b), c)
        self.origin = lower.min(axis=0)
        extent = upper.max(axis=0) - self.origin
        self.size = 64
        self.scale = np.divide(self.size, extent, out=np.zeros(3), where=extent > 0)
        lo = self.bins(lower)
        hi = self.bins(upper) + 1
        # Range additions mark every bin overlapped by each triangle's bounding box.
        n = self.size + 1
        counts = np.zeros((n, n, n), dtype=np.int64)
        for x in (0, 1):
            for y in (0, 1):
                for z in (0, 1):
                    bounds = (lo, hi)
                    corner = (bounds[x][:, 0], bounds[y][:, 1], bounds[z][:, 2])
                    np.add.at(counts, corner, (-1)**(x+y+z))
        self.occupied = counts.cumsum(0).cumsum(1).cumsum(2)[:-1, :-1, :-1] > 0

    def bins(self, xyz):
        return np.clip((xyz - self.origin) * self.scale, 0, self.size-1).astype(np.intp)

    def candidates(self, start, end):
        first = self.bins(start)
        last = self.bins(end)
        # A segment whose endpoints share an empty bin cannot intersect a face.
        return np.flatnonzero(self.occupied[tuple(first.T)] | np.any(first != last, axis=1))


class Boundaries:
    def __init__(self, patches, walls, outlets):
        self.names = list(patches)
        self.status = ["escaped" if name in outlets else
                       "deposited" if name in walls else "boundary" for name in self.names]
        append = vtk.vtkAppendPolyData()
        patch_ids = []
        for index, mesh in enumerate(patches.values()):
            if mesh.GetNumberOfPolys():
                append.AddInputData(mesh)
                patch_ids.append(np.full(mesh.GetNumberOfPolys(), index))
        if not patch_ids:
            raise ValueError("No boundary faces in mesh")
        append.Update()
        mesh = append.GetOutput()
        self.patch_ids = np.concatenate(patch_ids)
        xyz = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float, copy=False)
        faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        a, b, c = xyz[faces].transpose(1, 0, 2)
        self.grid = BoundaryGrid(a, b, c)
        self.normals = np.cross(b-a, c-a)  # OpenFOAM boundary faces point outwards.
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(mesh)
        self.locator.BuildLocator()

    def hits(self, start, end):
        patch = np.full(len(start), -1)
        fraction = np.full(len(start), np.inf)
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        cells = vtk.vtkIdList()
        cell = vtk.vtkGenericCell()
        for i in self.grid.candidates(start, end):
            p, q = start[i], end[i]
            if not self.locator.IntersectWithLine(p, q, 0., points, cells, cell):
                continue
            direction = q-p
            length2 = direction @ direction
            for j in range(cells.GetNumberOfIds()):
                face = cells.GetId(j)
                if self.normals[face] @ direction <= 0:
                    continue
                u = np.clip((points.GetPoint(j)-p) @ direction / length2, 0., 1.)
                if u < fraction[i]:
                    fraction[i] = u
                    patch[i] = self.patch_ids[face]
        return patch, fraction

    def count_enclosed(self, xyz):
        """Independently classify failed samples against the complete boundary."""
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(self.locator.GetDataSet())
        clean.Update()
        surface = clean.GetOutput()
        if not vtk.vtkSelectEnclosedPoints.IsSurfaceClosed(surface):
            return None
        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetTolerance(1e-8)
        enclosed.Initialize(surface)
        return sum(enclosed.IsInsideSurface(point) for point in xyz)


@contextmanager
def trajectory_output(path, names):
    if path is None:
        yield None
        return
    suffix = path.suffix.lower()
    if suffix not in (".vtk", ".vtp"):
        with path.open("x", newline="") as f:
            out = csv.writer(f)
            out.writerow(["id", "time", "x", "y", "z", "status", "patch"])
            def emit(time, coords, labels, status, patch=-1):
                times = np.broadcast_to(time, (len(labels),))
                name = names[patch] if patch >= 0 else ""
                out.writerows((int(i), t, *p, status, name) for i, t, p in zip(labels, times, coords))
            yield emit
        return

    if path.exists():
        raise FileExistsError(path)
    batches = []

    def record(time, coords, labels, status, patch=-1):
        if len(labels):
            batches.append((coords, labels, np.full(len(labels), time, dtype=float),
                            np.full(len(labels), STATUS[status], dtype=np.uint8),
                            np.full(len(labels), patch, dtype=np.int32)))

    yield record
    xyz, ids, times, status, patch_ids = (np.concatenate(parts) for parts in zip(*batches))
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz))
    tracks = vtk.vtkPolyData()
    tracks.SetPoints(points)
    # Stable grouping keeps each particle's samples in chronological order.
    order = np.argsort(ids, kind="stable").astype(np.int64)
    offsets = np.r_[0, np.cumsum(np.bincount(ids), dtype=np.int64)]
    lines = vtk.vtkCellArray()
    lines.SetData(numpy_to_vtk(offsets), numpy_to_vtk(order))
    tracks.SetLines(lines)
    for name, values in (("particle_id", ids), ("time", times), ("outside", (status == STATUS["outside"]).astype(np.uint8)),
                         ("status", status), ("patch_id", patch_ids)):
        array = numpy_to_vtk(values)
        array.SetName(name)
        tracks.GetPointData().AddArray(array)

    for label, strings in (("patch_names", names), ("status_names", list(STATUS))):
        table = vtk.vtkStringArray()
        table.SetName(label)
        for value in strings:
            table.InsertNextValue(value)
        tracks.GetFieldData().AddArray(table)

    if suffix == ".vtk":
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    else:
        writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(tracks)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        writer.SetFileName(str(temporary))
        if not writer.Write() or writer.GetErrorCode():
            raise OSError(f"Failed to write {path}")
        os.link(temporary, path)


def droplet_relaxation(slip, diameter, liquid_density, gas_density, gas_viscosity,
                       drag_model="schiller-naumann"):
    """Effective relaxation time for the selected spherical droplet drag law."""
    reynolds = gas_density * diameter * np.linalg.norm(slip, axis=1) / gas_viscosity
    # Cd*Re/24 has a finite limit at zero slip.
    if drag_model == "schiller-naumann":
        correction = np.where(reynolds < 1000, 1 + 0.15 * reynolds**0.687,
                              (0.44 / 24) * reynolds)
    elif drag_model == "sphereDrag":
        correction = np.where(reynolds <= 1000, 1 + reynolds**(2 / 3) / 6,
                              (0.424 / 24) * reynolds)
    else:
        raise ValueError(f"Unknown drag model: {drag_model}")
    return (liquid_density * diameter**2 / (18 * gas_viscosity) / correction)[:, None]


def relaxation_step(xyz, speed, gas_velocity, tau, gravity, h):
    """Integrate with gas velocity and drag coefficient frozen over this step."""
    ratio = h / tau
    gain = -np.expm1(-ratio)
    delta = (gas_velocity - speed) + tau * gravity
    # 1 - gain/ratio suffers cancellation on steps much shorter than tau.
    drift = 1 - gain / ratio
    small = ratio < 1e-4
    if np.any(small):
        z = ratio[small]
        drift[small] = z * (0.5 - z / 6 + z**2 / 24)
    return xyz + h * (speed + drift * delta), speed + gain * delta


def write_flow_report(args, start, release_times, crossings):
    width = args.flow_interval or args.dt
    edges = start + width * np.arange(math.ceil((args.end - start) / width))
    # Drop a rounded duplicate of the final edge, then include end exactly.
    edges = np.append(edges[(edges == start) | (edges < np.nextafter(args.end, start))], args.end)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Flow interval is below time precision")
    injection, _ = np.histogram(release_times, bins=edges)
    series = [("injected", args.patch or "", injection)]
    total = np.zeros_like(injection)
    for name, batches in crossings.items():
        times = np.concatenate(batches) if batches else []
        counts, _ = np.histogram(np.clip(times, start, args.end), bins=edges)
        series.append(("outlet", name, counts))
        total += counts
    series.append(("outlet_total", "", total))
    volume = math.pi * args.diameter**3 / 6 if args.diameter else None
    with args.flow_csv.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_s", "end_s", "stream", "patch", "count", "cumulative_count",
                         "number_per_s", "volume_m3_per_s", "mass_kg_per_s"])
        for kind, patch, counts in series:
            cumulative = 0
            for left, right, count in zip(edges[:-1], edges[1:], counts):
                cumulative += count
                rate = count / (right-left)
                volume_rate = rate * volume if volume is not None else None
                mass_rate = volume_rate * args.liquid_density if volume_rate is not None else None
                writer.writerow([left, right, kind, patch, count, cumulative, rate,
                                 volume_rate, mass_rate])


def run_parcels(a, flow, boundaries, seeds, releases, start):
    from parcel_tracking import ParcelTracker
    tracker = ParcelTracker(flow.sampler, boundaries.names, len(seeds), a.diameter,
                            a.liquid_density, a.gas_density, a.gas_viscosity, a.gravity, 1.)
    counts = dict.fromkeys(("deposited", "escaped", "boundary", "outside"), 0)
    crossings = {name: [] for name in sorted(a.outlet)}
    injected_at = np.empty_like(releases)
    live = np.empty(0, dtype=int)
    pending = step = last_bin = 0
    t = start
    with trajectory_output(a.output, boundaries.names) as emit:
        def move(ids, interval, fraction, step_end):
            patch, at = tracker.advance(ids, interval, fraction)
            for index in np.unique(patch[patch >= 0]):
                hit = patch == index
                times = step_end - (1-at[hit])*np.broadcast_to(interval, at.shape)[hit]
                # During the second pass a newborn parcel can hit before its
                # pseudo-injection time on the carrier clock. Keep CSV events ordered.
                times = np.maximum(times, injected_at[ids[hit]])
                status = boundaries.status[index]
                counts[status] += int(hit.sum())
                if emit:
                    emit(times, tracker.position(ids[hit]), ids[hit], status, index)
                if a.flow_csv and status == "escaped":
                    crossings[boundaries.names[index]].append(times)
            return ids[patch < 0]

        while t < a.end and (len(live) or pending < len(seeds)):
            end = min(t+a.dt, a.end)
            dt = end-t
            if dt <= 0:
                raise ValueError("dt is below time precision")
            stop = np.searchsorted(releases, end, side="left")
            if stop > pending:
                born = np.arange(pending, stop)
                tracker.release(born, seeds[born], flow.sampler, a.initial_velocity)
                width = min(dt, max(0., releases[-1]-t))
                injection = t + width*np.arange(len(born))/len(born)
                injected_at[born] = injection
                if emit:
                    emit(injection, seeds[born], born, "active")
                # v2606 InjectionModel::inject moves a new parcel first.
                # Cloud::move then resets stepFraction on the entire cloud.
                interval = end-injection
                born = move(born, interval, (dt-interval)/dt, end)
                live = np.concatenate((live, born))
                pending = stop
            live = move(live, dt, 0., end)
            step += 1
            save_bin = (math.floor(math.nextafter(step*a.dt/a.track_interval, math.inf))
                        if a.track_interval and a.track_interval > a.dt else step)
            if emit and len(live) and (save_bin > last_bin or end == a.end):
                emit(end, tracker.position(live), live, "active")
            last_bin, t = save_bin, end
    if a.flow_csv:
        write_flow_report(a, start, injected_at[:pending], crossings)
    if a.injection_duration:
        print(f"injected={pending} active={len(live)}", *(f"{k}={v}" for k, v in counts.items()))


def run(a):
    if getattr(a, "cell_point", False):
        flow = CellPointFlow(a.case)
    else:
        flow = Flow(a.case, getattr(a, "cell_cache_size", DEFAULT_CELL_CACHE_SIZE))
    t = float(flow.times[0]) if a.start is None else a.start
    if not all(map(math.isfinite, [a.dt, t, a.end, a.tau, *a.gravity])) or a.dt <= 0 or a.tau < 0 or a.end <= t:
        raise ValueError("Require finite parameters, dt > 0, tau >= 0 and start < end")
    if len(flow.times) > 1 and (t < flow.times[0] or a.end > flow.times[-1]):
        raise ValueError("Transient tracking must stay within the saved time range")
    patches = flow.boundary_surfaces()
    outlets, walls = set(a.outlet), set(a.wall)
    requested = walls | outlets | ({a.patch} if a.patch else set())
    if requested - patches.keys():
        raise ValueError(f"Unknown patches: {requested - patches.keys()}. Available: {list(patches)}")
    if walls & outlets:
        raise ValueError("A patch cannot be both --wall and --outlet")
    boundaries = Boundaries(patches, walls | wall_names(a.case), outlets)
    if a.patch:
        xyz = sample_surface(patches[a.patch], a.count, np.random.default_rng(a.seed))
    else:
        with a.seeds.open() as f:
            rows = list(csv.DictReader(f))
        xyz = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
        if not rows or not np.isfinite(xyz).all():
            raise ValueError("Provide finite x,y,z seed coordinates")
    seeds = xyz
    start = t
    duration = a.injection_duration
    if not math.isfinite(duration) or duration < 0 or duration > a.end-start + math.ulp(a.end):
        raise ValueError("Injection duration must fit between start and end")
    # Each ID represents one droplet released at the beginning of an equal time slot.
    release_times = start + duration * (np.arange(len(seeds)) / len(seeds))
    if release_times[-1] >= a.end:
        raise ValueError("Release schedule exceeds available time precision")
    if getattr(a, "parcel_tracking", False):
        if len(flow.times) != 1:
            raise ValueError("The v2606 parcel comparison requires one frozen velocity field")
        return run_parcels(a, flow, boundaries, seeds, release_times, start)
    pending = 0
    ids = np.empty(0, dtype=np.intp)
    xyz = seeds[:0]
    vel = speed = xyz
    droplet = a.diameter is not None
    drag_model = getattr(a, "drag_model", "schiller-naumann")
    inertial = droplet or a.tau > 0
    gravity = np.asarray(a.gravity)
    if droplet:
        gravity = gravity * (1 - a.gas_density / a.liquid_density)
    finalized = dict.fromkeys(("deposited", "escaped", "boundary", "outside"), 0)
    missed = []
    crossings = {name: [] for name in sorted(outlets)}
    with trajectory_output(a.output, boundaries.names) as emit:
        step = 0
        last_bin = 0
        while t < a.end and (len(ids) or pending < len(seeds)):
            end = min(a.end, t+a.dt)
            if end <= t:
                raise ValueError("dt is below time precision")
            h = end-t
            stop = np.searchsorted(release_times, end, side="left") if pending < len(seeds) else pending
            step_start = None
            if stop > pending:
                born = np.arange(pending, stop)
                birth = release_times[pending:stop]
                positions = seeds[pending:stop]
                initial, valid = flow.sample_release(birth, positions)
                if not valid.all():
                    raise ValueError(f"Seeds outside mesh (zero-based): {born[~valid]}")
                if emit:
                    emit(birth, positions, born, "active")
                if birth[-1] > t:
                    step_start = np.concatenate((np.full(len(ids), t), birth))
                    h = (end-step_start)[:, None]
                xyz = np.concatenate((xyz, positions)) if len(ids) else positions
                vel = np.concatenate((vel, initial)) if len(ids) else initial
                if inertial:
                    initial_speed = initial if a.initial_velocity is None else np.broadcast_to(a.initial_velocity, initial.shape)
                    speed = np.concatenate((speed, initial_speed)) if len(ids) else initial_speed
                ids = np.concatenate((ids, born))
                pending = stop
            if not len(ids):
                t = end
                step += 1
                continue
            if inertial:
                tau = (droplet_relaxation(vel - speed, a.diameter, a.liquid_density,
                                          a.gas_density, a.gas_viscosity, drag_model)
                       if droplet else np.asarray([[a.tau]]))
                trial, speed = relaxation_step(xyz, speed, vel, tau, gravity, h)
            else:
                trial = xyz + h*vel
            patch, fraction = boundaries.hits(xyz, trial)
            for index in np.unique(patch[patch >= 0]):
                hit = patch == index
                u = fraction[hit]
                finalized[boundaries.status[index]] += len(u)
                hit_times = (t + h*u if step_start is None else
                             step_start[hit] + h[hit, 0]*u)
                if emit:
                    pos = xyz[hit] + u[:, None]*(trial[hit]-xyz[hit])
                    emit(hit_times, pos, ids[hit], boundaries.status[index], index)
                if a.flow_csv and boundaries.status[index] == "escaped":
                    crossings[boundaries.names[index]].append(hit_times)
            remaining = patch < 0
            trial, ids = trial[remaining], ids[remaining]
            if inertial:
                speed = speed[remaining]
            sampled, ok = flow.sample(end, trial) if len(ids) else (trial, np.empty(0, dtype=bool))
            finalized["outside"] += len(ok) - np.count_nonzero(ok)
            if not ok.all():
                missed.append(trial[~ok])
            if emit:
                emit(end, trial[~ok], ids[~ok], "outside")
            xyz, vel, ids = trial[ok], sampled[ok], ids[ok]
            if inertial:
                speed = speed[ok]
            step += 1
            if emit:
                # Use the step index to avoid accumulating rounding error in the save schedule.
                current_bin = step
                if a.track_interval and a.track_interval > a.dt:
                    current_bin = math.floor(math.nextafter(step*a.dt/a.track_interval, math.inf))
                if current_bin > last_bin or end == a.end:
                    emit(end, xyz, ids, "active")
                last_bin = current_bin
            t = end
    if a.flow_csv:
        write_flow_report(a, start, release_times, crossings)
    if duration:
        print(f"injected={pending} active={len(ids)}", *(f"{k}={v}" for k, v in finalized.items()))
    if missed:
        count = boundaries.count_enclosed(np.concatenate(missed))
        print("outside_inside_mesh=" + ("unknown" if count is None else str(count)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="empty .foam marker in case directory")
    parser.add_argument("seeds", nargs="?", type=Path, help="CSV with x,y,z columns")
    parser.add_argument("output", type=Path, help="trajectory file (.csv, .vtk or .vtp), or flow CSV with --flow-only")
    parser.add_argument("--start", type=float)
    parser.add_argument("--cell-cache-size", type=int, default=DEFAULT_CELL_CACHE_SIZE, metavar="N",
                        help="maximum cached cell geometries for the default VTK sampler (default: %(default)s)")
    parser.add_argument("--cell-point", action="store_true",
                        help="use the v2606 exportCellPoint data in the case instead of VTK nodal U")
    parser.add_argument("--parcel-tracking", action="store_true",
                        help="compare with v2606 face tracking, sphereDrag, analytical U and maxCo=1")
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--dt", required=True, type=float)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--diameter", type=float, help="enable spherical droplet drag, diameter in m")
    model.add_argument("--tau", type=float, default=0, help="Stokes relaxation time in seconds")
    parser.add_argument("--drag-model", choices=("schiller-naumann", "sphereDrag"),
                        default=argparse.SUPPRESS,
                        help="drag law for --diameter (default: schiller-naumann; sphereDrag for OpenFOAM v2606 comparison)")
    parser.add_argument("--gravity", nargs=3, type=float, help="gravity in m/s^2 (droplets default to 0 0 -9.81, --tau to zero)")
    parser.add_argument("--liquid-density", type=float, help="droplet density in kg/m^3 (default 998.2)")
    parser.add_argument("--gas-density", type=float, help="gas density in kg/m^3 (default 1.204)")
    parser.add_argument("--gas-viscosity", type=float, help="gas dynamic viscosity in Pa s (default 1.81e-5)")
    parser.add_argument("--initial-velocity", nargs=3, type=float, help="injection velocity in m/s (default local U)")
    parser.add_argument("--patch", help="OpenFOAM boundary patch for particle injection")
    parser.add_argument("--injection-duration", type=float, default=0, metavar="SECONDS",
                        help="release all particles uniformly over this duration from start (default: instantaneous)")
    parser.add_argument("--count", type=int, help="number of particles released on the patch")
    parser.add_argument("--seed", type=int, help="nonnegative random seed for reproducible positions")
    parser.add_argument("--outlet", action="append", default=[], metavar="NAME",
                        help="classify hits on this patch as escaped (repeatable)")
    parser.add_argument("--wall", action="append", default=[], metavar="NAME",
                        help="additional sticking wall (type wall is detected automatically, repeatable)")
    parser.add_argument("--track-interval", type=float, help="trajectory sampling interval in seconds")
    parser.add_argument("--flow-only", action="store_true", help="use the positional output for the flow CSV only")
    parser.add_argument("--flow-csv", type=Path, help="new CSV for injected and outlet particle rates")
    parser.add_argument("--flow-interval", type=float, help="flow averaging interval in seconds (default: dt)")
    args = parser.parse_args()
    if args.cell_cache_size < 1:
        parser.error("--cell-cache-size must be positive")
    if args.parcel_tracking:
        if args.diameter is None or getattr(args, "drag_model", "sphereDrag") != "sphereDrag":
            parser.error("--parcel-tracking requires --diameter and sphereDrag")
        args.cell_point, args.drag_model = True, "sphereDrag"
    if hasattr(args, "drag_model") and args.diameter is None:
        parser.error("--drag-model requires --diameter")
    if args.flow_only:
        if args.flow_csv or args.track_interval is not None:
            parser.error("--flow-only uses the positional output and excludes --flow-csv and --track-interval")
        args.flow_csv, args.output = args.output, None
    if args.track_interval is not None and (not math.isfinite(args.track_interval) or args.track_interval <= 0):
        parser.error("--track-interval must be finite and positive")
    if args.flow_interval is not None and (not args.flow_csv or not math.isfinite(args.flow_interval)
                                           or args.flow_interval <= 0):
        parser.error("--flow-interval requires --flow-csv and a finite, positive width")
    if args.flow_csv and (args.flow_csv.exists() or (args.output and args.flow_csv.resolve() == args.output.resolve())):
        parser.error("--flow-csv requires a new file distinct from the trajectory output")
    if bool(args.patch) == bool(args.seeds):
        parser.error("provide either a seed CSV or --patch NAME")
    if args.patch and (args.count is None or args.count < 1):
        parser.error("--patch requires a positive --count")
    if not args.patch and (args.count is not None or args.seed is not None):
        parser.error("--count and --seed require --patch")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.diameter is not None:
        if not math.isfinite(args.diameter) or args.diameter <= 0:
            parser.error("--diameter must be finite and positive")
        for key, default in (("liquid_density", 998.2), ("gas_density", 1.204), ("gas_viscosity", 1.81e-5)):
            value = getattr(args, key)
            if value is None:
                setattr(args, key, default)
            elif not math.isfinite(value) or value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    elif any(x is not None for x in (args.liquid_density, args.gas_density, args.gas_viscosity)):
        parser.error("droplet material properties require --diameter")
    if args.initial_velocity is not None:
        if args.diameter is None and args.tau <= 0:
            parser.error("--initial-velocity requires --diameter or positive --tau")
        if not np.isfinite(args.initial_velocity).all():
            parser.error("--initial-velocity components must be finite")
    if args.gravity is None:
        args.gravity = (0, 0, -9.81) if args.diameter is not None else (0, 0, 0)
    run(args)

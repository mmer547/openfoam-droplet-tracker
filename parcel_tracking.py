"""Face-by-face parcel motion for a stationary, positive-volume tet decomposition."""
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


class ParcelTracker:
    def __init__(self, data, patch_names, count, diameter, rho, gas_rho, mu, gravity, max_co):
        self.nodes = data.tets
        self.points = vtk_to_numpy(data.locator.GetDataSet().GetPoints().GetData())
        self.values = data.values(0)
        self.volumes = np.loadtxt(data.directory / "volumes.dat", ndmin=1)
        self.cell = self.nodes[:, 0] - (len(self.points) - len(self.volumes))
        if (self.cell.min() < 0 or self.cell.max() >= len(self.volumes)
                or not np.isfinite(self.volumes).all() or np.any(self.volumes <= 0)):
            raise ValueError("Invalid exported cell volumes")
        self.length = np.cbrt(self.volumes)
        a, b, c, d = self.points[self.nodes].transpose(1, 0, 2)
        ab, ac, ad = b-a, c-a, d-a
        self.transform = np.stack((np.cross(d-b, c-b), np.cross(ac, ad),
                                   np.cross(ad, ab), np.cross(ab, ac)), axis=1)
        self.det = (ab*self.transform[:, 1]).sum(axis=1)
        if np.any(self.det <= 0):
            raise ValueError("Parcel face tracking requires positive-volume tetrahedra")
        self.neighbours = self._neighbours(data.directory, patch_names)
        self.tet = np.full(count, -1, dtype=int)
        self.y = np.zeros((count, 4))
        self.u = np.zeros((count, 3))
        self.diameter, self.rho, self.gas_rho, self.mu = diameter, rho, gas_rho, mu
        self.gravity = np.asarray(gravity) * (1-gas_rho/rho)
        self.max_co = max_co

    def _neighbours(self, directory, patch_names):
        # Each internal triangle occurs twice. External triangles occur once in
        # the tet list and once in the boundary export, with their patch index.
        sides = self.nodes[:, [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]].reshape(-1, 3)
        triangles = [sides]
        owners = [np.repeat(np.arange(len(self.nodes)), 4)]
        for patch, name in enumerate(patch_names):
            path = directory / "patches" / (name + ".dat")
            if path.stat().st_size:
                faces = np.loadtxt(path, dtype=np.int64, ndmin=2)
                triangles.append(faces)
                owners.append(np.full(len(faces), -1-patch))
        keys = np.sort(np.concatenate(triangles), axis=1)
        order = np.lexsort(keys.T)
        if len(order) % 2:
            raise ValueError("Incomplete tetrahedral face connectivity")
        pairs = order.reshape(-1, 2)
        if (np.any(keys[pairs[:, 0]] != keys[pairs[:, 1]])
                or np.any(np.all(keys[pairs[:-1, 1]] == keys[pairs[1:, 0]], axis=1))):
            raise ValueError("Tetrahedral faces must pair with one neighbour or one patch")
        adjacent = np.empty(len(keys), dtype=int)
        adjacent[order] = np.concatenate(owners)[pairs[:, ::-1].ravel()]
        return adjacent[:len(sides)].reshape(-1, 4)

    def release(self, ids, xyz, sampler, initial):
        tet, y = sampler.locate(xyz)
        if np.any(tet < 0):
            raise ValueError(f"Seeds outside mesh (zero-based): {ids[tet < 0]}")
        self.tet[ids] = tet
        self.y[ids] = np.maximum(y, 0)
        self.y[ids] /= self.y[ids].sum(axis=1, keepdims=True)
        self.u[ids] = self.interpolate(ids) if initial is None else initial

    def interpolate(self, ids):
        return (self.values[self.nodes[self.tet[ids]]] * self.y[ids, :, None]).sum(axis=1)

    def position(self, ids):
        return (self.points[self.nodes[self.tet[ids]]] * self.y[ids, :, None]).sum(axis=1)

    def update_velocity(self, ids, dt):
        gas = self.interpolate(ids)
        velocity = self.u[ids]
        re = self.gas_rho*np.linalg.norm(velocity-gas, axis=1)*self.diameter/self.mu
        cd_re = np.where(re > 1000, .424*re, 24*(1 + np.power(re, 2/3)/6))
        beta = .75*self.mu*cd_re/(self.rho*self.diameter**2)
        effective = np.where(beta*dt > 1e-15, (1-np.exp(-beta*dt))/beta, dt)
        self.u[ids] += (beta[:, None]*gas + self.gravity - beta[:, None]*velocity)*effective[:, None]

    def advance(self, ids, duration, initial_fraction):
        """Move through tets and return the first patch and step fraction for each ID."""
        duration = np.broadcast_to(duration, (len(ids),))
        fraction = np.broadcast_to(initial_fraction, (len(ids),)).copy().astype(float)
        patch = np.full(len(ids), -1)
        stalled = set()
        rows = np.flatnonzero(fraction < 1)
        while len(rows):
            pid = ids[rows]
            start = fraction[rows].copy()
            length = self.length[self.cell[self.tet[pid]]]
            travel = duration[rows, None]*self.u[pid]
            f = np.minimum(np.minimum(1-start, self.max_co),
                           self.max_co*length/np.maximum(1e-15*length, np.linalg.norm(travel, axis=1)))
            sub = np.arange(len(rows))
            remaining = np.ones(len(rows))
            destination = np.full(len(rows), -1)
            on_face = np.zeros(len(rows), dtype=bool)
            crossings = 0
            while len(sub):
                # A straight line can visit each positive-volume tet at most once
                # before reaching a cell face; detect a broken adjacency cycle.
                crossings += 1
                if crossings > len(self.nodes):
                    raise RuntimeError("Cyclic tetrahedral path")
                p = pid[sub]
                t = self.tet[p]
                y = self.y[p]
                displacement = (remaining[sub]*f[sub])[:, None]*travel[sub]
                dx = (self.transform[t]*displacement[:, None, :]).sum(axis=2)
                mu = np.full_like(dx, np.inf)
                np.divide(-y, dx, out=mu, where=dx < -self.det[t, None]*1e-15)
                side = mu.argmin(axis=1)
                limit = 1/self.det[t]
                hit_mu = mu[np.arange(len(p)), side]
                hit = (hit_mu >= 0) & (hit_mu < limit)
                amount = np.where(hit, hit_mu, limit)
                y = np.maximum(0., y + amount[:, None]*dx)
                y[hit, side[hit]] = 0
                y[~hit] /= y[~hit].sum(axis=1, keepdims=True)
                self.y[p] = y
                ratio = np.where(hit, amount*self.det[t], 1.)
                fraction[rows[sub]] += remaining[sub]*f[sub]*ratio
                remaining[sub] *= 1-ratio
                face = hit & (side == 0)
                on_face[sub[face]] = True
                destination[sub[face]] = self.neighbours[t[face], 0]
                internal = hit & (side != 0)
                if internal.any():
                    self.cross(p[internal], self.neighbours[t[internal], side[internal]])
                sub = sub[internal]
            self.update_velocity(pid, (fraction[rows]-start)*duration[rows])
            exterior = on_face & (destination < 0)
            patch[rows[exterior]] = -1-destination[exterior]
            inner = on_face & (destination >= 0)
            if inner.any():
                self.cross(pid[inner], destination[inner])
            for index in np.flatnonzero(~exterior & (fraction[rows] == start)):
                state = (pid[index], start[index], self.tet[pid[index]])
                if state in stalled:
                    raise RuntimeError(f"Parcel {pid[index]} repeatedly crossed a face without advancing")
                stalled.add(state)
            rows = rows[(~exterior) & (fraction[rows] < 1)]
        return patch, np.minimum(fraction, 1.)

    def cross(self, ids, neighbour):
        before = self.nodes[self.tet[ids]]
        after = self.nodes[neighbour]
        shared = after[:, :, None] == before[:, None, :]
        self.y[ids] = (shared*self.y[ids, None, :]).sum(axis=2)
        self.tet[ids] = neighbour

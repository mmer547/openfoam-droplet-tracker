"""Compare an exportCellPoint -samplePoints reference with the Python sampler."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track_particles import CellPointFlow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("coordinates", type=Path, help="same xyz rows passed to -samplePoints")
    parser.add_argument("--time", default="0", help="exported OpenFOAM time directory")
    args = parser.parse_args()
    directory = args.case.parent / "postProcessing/trackerCellPoint"
    ref = np.loadtxt(directory / args.time / "samples.dat", ndmin=2)
    xyz = np.loadtxt(args.coordinates, ndmin=2)
    if xyz.shape != (len(ref), 3) or ref.shape[1] != 11:
        parser.error("Expected matching xyz rows and native cell/vertices/weights/U rows")
    flow = CellPointFlow(args.case)
    ids, weights = flow.sampler.locate(xyz)
    velocity, ok = flow.sample(float(args.time), xyz)
    native = ref[:, 0] >= 0
    both = native & ok
    print(f"both={both.sum()} native_only={(native & ~ok).sum()} "
          f"python_only={(ok & ~native).sum()} neither={(~ok & ~native).sum()}")
    if both.any():
        # The exporter emits the centre of cell zero first in the connectivity.
        offset = np.loadtxt(directory / "tets.dat", dtype=int, max_rows=1)[0]
        nodes = flow.sampler.tets[ids[both]]
        same_cell = nodes[:, 0] - offset == ref[both, 0]
        same_tet = same_cell & (nodes[:, 1:] == ref[both, 1:4]).all(axis=1)
        error = np.linalg.norm(velocity[both]-ref[both, 8:11], axis=1)
        print(f"cell_mismatch={(~same_cell).sum()} tet_mismatch={(~same_tet).sum()}")
        if same_tet.any():
            print("max_weight_error =", np.abs(weights[both][same_tet]-ref[both, 4:8][same_tet]).max())
        print("velocity_error_m_s (median, p95, max) =", np.percentile(error, [50, 95, 100]))


if __name__ == "__main__":
    main()

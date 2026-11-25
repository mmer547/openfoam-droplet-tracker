"""Compare cyclone tracker CSV rows with a v2606 solver log and ASCII parcel snapshots."""
import argparse
import csv
import re
from pathlib import Path

import numpy as np


def read_tracks(path):
    born, terminal, active = {}, {}, {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            i, t = int(row["id"]), float(row["time"])
            xyz = np.array([float(row[k]) for k in "xyz"])
            if i not in born:
                if row["status"] != "active":
                    raise ValueError(f"Missing birth for ID {i}")
                born[i] = t
            elif row["status"] == "active":
                active[t, i] = xyz
            elif i not in terminal and row["status"] in ("escaped", "deposited"):
                terminal[i] = (t, row["status"], xyz)
            else:
                raise ValueError(f"Unexpected terminal record for ID {i}: {row}")
    return born, terminal, active


def compare_log(path, born, terminal):
    births = np.sort(list(born.values()))
    walls, exits = [np.sort([t for t, s, _ in terminal.values() if s == status])
                   for status in ("deposited", "escaped")]
    steps = errors = 0
    text = path.read_text()
    if not text.rstrip().endswith("End"):
        raise ValueError("Incomplete OpenFOAM log")
    for tm, chunk in re.findall(r"^Time = ([\d.eE+-]+)\n(.*?)(?=^Time = |\Z)", text, re.M | re.S):
        t = float(tm)
        injected = int(re.search(r"- parcels added\s+= (\d+)", chunk)[1])
        current = int(re.search(r"Current number of parcels\s+= (\d+)", chunk)[1])
        def patch_count(name, event):
            text = chunk.split("Parcel fate: patch "+name+" (number, mass)")[1].split("Parcel fate:")[0]
            return int(re.search(r"- "+event+r"\s+= (\d+)", text)[1])
        wall, out = patch_count("walls", "stick"), patch_count("outlet", "escape")
        n, w, e = births.searchsorted(t), walls.searchsorted(t, "right"), exits.searchsorted(t, "right")
        errors += int((injected, current-wall, wall, out) != (n, n-w-e, w, e)
                      or patch_count("inlet", "escape") != 0)
        steps += 1
    if not steps:
        raise ValueError("No solver time steps found")
    return steps, errors


def labels(path):
    content = path.read_text().split("}", 1)[1]
    uniform = re.search(r"(\d+)\s*\{\s*(\d+)\s*\}", content)
    if uniform:
        return np.full(int(uniform[1]), int(uniform[2]), dtype=int)
    count, values = re.search(r"(\d+)\s*\(\s*([\d\s]*)\)", content).groups()
    result = np.fromstring(values.strip(), sep=" ", dtype=int) if values.strip() else np.empty(0, int)
    if len(result) != int(count):
        raise ValueError(f"Incomplete label list in {path}")
    return result


def compare_state(folder, time, born, terminal, active):
    ids, moving = labels(folder/"origId"), labels(folder/"active")
    xyz = re.findall(r"^\(([^()\n]+)\)\s+\d+\s*$", (folder/"positions").read_text(), re.M)
    if not (len(set(ids)) == len(ids) == len(moving) == len(xyz)):
        raise ValueError(f"Incomplete parcel snapshot at {time}")
    native = {i: ("active" if m else "deposited") for i, m in zip(ids, moving)}
    checks = errors = 0
    for i, birth in born.items():
        if birth < time:
            event = terminal.get(i)
            status = event[1] if event and event[0] <= time else "active"
            errors += status != native.get(i, "escaped")
            checks += 1
    distances = []
    for i, coords in zip(ids, xyz):
        event = terminal.get(i)
        if event and event[0] <= time:
            p = event[2]
        elif (time, i) in active:
            p = active[time, i]
        else:
            raise ValueError(f"No trajectory coordinate for ID {i} at {time}; check the save interval")
        distances.append(np.linalg.norm(p - np.fromstring(coords, sep=" ")))
    return checks, errors, len(distances), max(distances, default=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("case", type=Path, help="OpenFOAM case with ASCII cloud output")
    parser.add_argument("--times", nargs="+", default=["1", "2", "3", "4", "5", "6"])
    args = parser.parse_args()
    born, terminal, active = read_tracks(args.tracks)
    steps, inventory_errors = compare_log(args.log, born, terminal)
    print(f"steps={steps} different_inventory_steps={inventory_errors}")
    failed = bool(inventory_errors)
    for time in args.times:
        folder = args.case/time/"lagrangian/kinematicCloud"
        n, errors, positions, maximum = compare_state(folder, float(time), born, terminal, active)
        print(f"time={time} ids={n} different_fates={errors} positions={positions} max_error_m={maximum:.6g}")
        failed |= bool(errors)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

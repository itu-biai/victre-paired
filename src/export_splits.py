#!/usr/bin/env python3
"""
VICTRE-Paired — export the released split assignment.

Reads only the small `seed` array of every chunk in a copy of the released
dataset and writes `splits.csv` (columns: seed, split) next to constants.py,
so that generate_dataset.py reproduces the released train/val/test membership.

Usage
-----
    python src/export_splits.py --data /path/to/victre-paired
    python src/export_splits.py --data /path/to/victre-paired --out splits.csv
"""

import os, glob, csv, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="dataset root with train/val/test subfolders")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.csv"))
args = ap.parse_args()

rows, seen = [], {}
for split in ("train", "val", "test"):
    for fp in sorted(glob.glob(os.path.join(args.data, split, "*.npz"))):
        with np.load(fp) as d:
            for s in d["seed"]:
                s = int(s)
                if s in seen:
                    raise SystemExit(f"seed {s} appears in both {seen[s]} and {split}")
                seen[s] = split
                rows.append((s, split))

rows.sort()
with open(args.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "split"]); w.writerows(rows)

counts = {sp: sum(1 for _, s in rows if s == sp) for sp in ("train", "val", "test")}
print(f"wrote {len(rows)} rows to {args.out}: {counts}")

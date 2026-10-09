#!/usr/bin/env python3
"""
VICTRE-Paired — the per-density z offset (DELTA), and how it was obtained.

`geom_offz` places the reconstruction volume along the beam axis:

    geom_offz = -(SDD - SID) + ZOUT * geom_vox_z / 2 + DELTA[density]

Most of DELTA is published, not fitted. VICTRE's own reconstruction code puts
the bottom of the volume a fixed distance above the detector:

    d_objbottom_det   = 2.00 cm      FBP_DBTrecon.c:869
    protective cover  = 0.10 cm      MC-GPU template, half of it below the volume
    -----------------------------------------------------------------
    published total   = 19.5 mm

The constants this repository ships sit 0.27-1.31 mm above that. The residual is
systematic (the same sign in every class) and absorbs the difference between
voxel-centre and detector-centre conventions in the LEAP modular-beam setup plus
the rounding in `vox_z = native_z / ZOUT`.

This script measures both: it re-fits DELTA on a probe set and reports the
residual parallax of the published constant, of the re-fit, and of the shipped
values, on patients that took no part in any fit.

    residual parallax = |dr|max, the sub-pixel shift between A(clean) and
    clean_proj, from an FFT cross-correlation per view.

Measured on 225 held-out patients: published constant 1.167 px, shipped
per-density constants 0.600 px. That difference is why the correction exists.

Usage
-----
    python src/fit_delta.py --data /path/to/victre-paired --out delta_fit/
    python src/fit_delta.py --data ... --n-probe 16 --n-holdout 75

Requires a CUDA GPU and `leapctype` (it forward-projects every probe patient).
Writes: sample_seeds.json (which patient was used for what) and delta_fit.csv.
"""

import os, glob, json, csv, argparse
import numpy as np

from constants import DELTA, DENSITIES, ZOUT, SID, SDD, BROKEN_SEEDS

PUBLISHED_DELTA_MM = 19.5          # 20.0 - 0.10/2, see the docstring


def shift_by_xcorr(a, b):
    """Sub-pixel shift between two 2-D arrays, by FFT cross-correlation."""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    c = np.fft.irfft2(A * np.conj(B), s=a.shape)
    p = np.unravel_index(np.argmax(c), c.shape)
    out = []
    for ax, n in zip(p, a.shape):
        m = c[p[0], :] if ax is p[1] else c[:, p[1]]
        i = ax
        y0, y1, y2 = m[(i-1) % n], m[i], m[(i+1) % n]
        den = (y0 - 2*y1 + y2)
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        v = i + d
        out.append(v - n if v > n / 2 else v)
    return out


def residual_parallax(G, gt, proj):
    """Largest |shift| over views between A(gt) and the stored projection."""
    a = G.A(gt)
    return float(max(np.hypot(*shift_by_xcorr(a[v], proj[v])) for v in range(a.shape[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="delta_fit")
    ap.add_argument("--n-probe", type=int, default=16, help="patients per density used for the fit")
    ap.add_argument("--n-holdout", type=int, default=75, help="patients per density used only to report")
    ap.add_argument("--span", type=float, default=1.5, help="search +/- this many mm around the shipped value")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from geometry import Geometry          # imported late: needs CUDA + LEAP

    # ---- index: seed -> chunk, index, density ----
    idx = {}
    for sp in ("train", "val", "test"):
        for fp in sorted(glob.glob(f"{args.data}/{sp}/*.npz")):
            with np.load(fp) as d:
                for b, s in enumerate(d["seed"]):
                    s = int(s)
                    if s in BROKEN_SEEDS:
                        continue
                    idx[s] = dict(chunk=os.path.basename(fp), b=int(b), split=sp,
                                  density=str(d["density"][b]),
                                  native_z=int(d["native_z"][b]), path=fp)
    print(f"indexed {len(idx)} patients")

    # probe and holdout are DISJOINT by construction: the holdout is drawn after
    # the probe is removed, so no patient reported on took part in the fit.
    pools, probe, holdout = {}, {}, {}
    # ORDERING MATTERS AND IS NOT FREE TO CHOOSE. The released DELTA was fitted by
    # validate_dataset.py stage 8 on pool[:16] of a pool ordered by (chunk, position
    # in chunk) -- not by seed. Sorting by seed here would re-fit on a DIFFERENT
    # probe set, which answers a different question: it would still show whether a
    # re-fit lands near the published value, but it would not let a reader reproduce
    # the published fit. The ordering below is therefore the same as
    # validate_dataset.py:357 and H4.
    for dn in DENSITIES:
        pool = sorted((s for s, r in idx.items() if r["density"] == dn),
                      key=lambda s: (idx[s]["chunk"], idx[s]["b"]))
        pools[dn] = pool
        probe[dn] = pool[:args.n_probe]
        holdout[dn] = pool[args.n_probe:args.n_probe + args.n_holdout]
    json.dump({"probe": probe, "holdout": holdout,
               "pool_sizes": {d: len(v) for d, v in pools.items()}},
              open(os.path.join(args.out, "sample_seeds.json"), "w"), indent=1)

    def measure(seed, delta_mm):
        r = idx[seed]
        with np.load(r["path"]) as d:
            gt = d["clean"][r["b"]].astype(np.float32) / 65535
            proj = d["clean_proj"][r["b"]].astype(np.float32) / 65535
            vox_z = float(d["geom_vox_z"][r["b"]])
            offx = float(d["geom_offx"][r["b"]]); offy = float(d["geom_offy"][r["b"]])
        offz = -(SDD - SID) + ZOUT * vox_z / 2.0 + delta_mm
        G = Geometry(vox_z=vox_z, offx=offx, offy=offy, offz=offz)
        return residual_parallax(G, gt, proj)

    # ---- fit on the probe set ----
    rows, refit = [], {}
    for dn in DENSITIES:
        grid = np.arange(DELTA[dn] - args.span, DELTA[dn] + args.span + 1e-9, args.step)
        best, best_v = None, np.inf
        for delta in grid:
            v = float(np.mean([measure(s, float(delta)) for s in probe[dn]]))
            rows.append(dict(stage="probe", density=dn, delta_mm=float(delta), mean_dr=v,
                             n=len(probe[dn])))
            if v < best_v:
                best, best_v = float(delta), v
        refit[dn] = best
        print(f"  {dn:>10s}: re-fit {best:.3f} mm (shipped {DELTA[dn]:.3f}), "
              f"probe mean |dr|max {best_v:.3f} px", flush=True)

    # ---- report on the held-out set: published vs re-fit vs shipped ----
    for dn in DENSITIES:
        for label, delta in (("published", PUBLISHED_DELTA_MM),
                             ("refit", refit[dn]),
                             ("shipped", DELTA[dn])):
            v = [measure(s, float(delta)) for s in holdout[dn]]
            rows.append(dict(stage="holdout", density=dn, which=label,
                             delta_mm=float(delta), mean_dr=float(np.mean(v)),
                             sd_dr=float(np.std(v)), n=len(v)))
            print(f"  {dn:>10s} {label:>9s} delta={delta:7.3f} mm -> "
                  f"|dr|max {np.mean(v):.3f} +/- {np.std(v):.3f} px  (n={len(v)})", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out, "delta_fit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote {args.out}/delta_fit.csv and {args.out}/sample_seeds.json")


if __name__ == "__main__":
    main()

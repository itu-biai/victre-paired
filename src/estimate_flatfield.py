#!/usr/bin/env python3
"""
VICTRE-Paired — flat-field estimation.

VICTRE does not publish the flat field used to normalise its projections, so it
is estimated here from the projections themselves. This script is what produced
`flatfield/ff_base.npy` and `flatfield/coefficients.json`, the two inputs
generate_dataset.py needs.

The estimate is self-controlled: it only ever uses detector pixels that see
*air* in a given view, so no tissue enters the flat field. Two stages:

  1. Base field. For every view, take the pixels outside the breast shadow
     (raw counts above AIR_THR) across many patients and reduce them with a
     per-pixel median. Air is the same physical measurement in every patient,
     so the median over patients removes anatomy and keeps the detector's
     response and the beam's shape.

  2. Per-density polynomial correction. The breast shadow is not in the same
     place for every density class, so some detector regions are seen as air by
     few patients of one class and many of another. A low-order 2-D polynomial
     is fitted, per density and per view, to the ratio (observed air / base
     field) over the pixels that class does expose, and evaluated everywhere.
     This extrapolates smoothly into the region a class never exposes rather
     than leaving a hole.

The extrapolation error is the limitation this introduces, and it is measured
rather than assumed: see the flat-field section of the paper (median 0.01-0.05
of attenuation at 0-20 mm from the exposed region, 95th percentile <= 0.15).

Usage
-----
    python src/estimate_flatfield.py --root /path/to/VICTRE --out flatfield/
    python src/estimate_flatfield.py --root ... --n-per-density 80 --degree 2
"""

import os, glob, json, argparse, re
import numpy as np

try:
    import pydicom
except ImportError:                                            # pragma: no cover
    raise SystemExit("pydicom is required: pip install pydicom")

from constants import NA, DENSITIES

AIR_THR_RAW = 5000          # raw counts above which a pixel is "illuminated air"
NATIVE_H, NATIVE_W = 3000, 1500


def projection_files(root, seed):
    fs = sorted(glob.glob(f"{root}/projections/{seed}/*/*.dcm"),
                key=lambda p: int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber))
    return fs if len(fs) == NA else None


def density_of(root, seed):
    for p in glob.glob(f"{root}/Locations/**/roi_*_{seed}.loc", recursive=True):
        m = re.search(r"/DBT-([A-Za-z]+)/", p)
        if m:
            return m.group(1)
    return None


def air_mask(img):
    """Pixels this view exposes directly: bright, and outside the breast shadow."""
    return img > AIR_THR_RAW


def fit_poly2d(mask, ratio, degree):
    """Least-squares 2-D polynomial through `ratio` where `mask`, evaluated everywhere."""
    h, w = ratio.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ys = (yy / h - 0.5).astype(np.float32)
    xs = (xx / w - 0.5).astype(np.float32)
    terms = [ys**i * xs**j for i in range(degree + 1) for j in range(degree + 1 - i)]
    Aall = np.stack([t.ravel() for t in terms], 1)
    k = mask.ravel()
    if k.sum() < 10 * Aall.shape[1]:
        return np.ones_like(ratio), None
    c, *_ = np.linalg.lstsq(Aall[k], ratio.ravel()[k], rcond=None)
    return (Aall @ c).reshape(h, w).astype(np.float32), [float(x) for x in c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="VICTRE source root (has projections/, Locations/)")
    ap.add_argument("--out", default="flatfield", help="output directory")
    ap.add_argument("--n-per-density", type=int, default=60)
    ap.add_argument("--degree", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = sorted(int(os.path.basename(p)) for p in glob.glob(f"{args.root}/projections/*")
                   if os.path.basename(p).isdigit())
    by_density = {d: [] for d in DENSITIES}
    for s in seeds:
        d = density_of(args.root, s)
        if d in by_density and len(by_density[d]) < args.n_per_density:
            by_density[d].append(s)
    print("patients per density:", {d: len(v) for d, v in by_density.items()})

    # ---- stage 1: base field, per-pixel median over air pixels, all classes ----
    base = np.zeros((NA, NATIVE_H, NATIVE_W), np.float32)
    pool = [s for v in by_density.values() for s in v]
    for view in range(NA):
        stack = []
        for s in pool:
            fs = projection_files(args.root, s)
            if fs is None:
                continue
            img = pydicom.dcmread(fs[view]).pixel_array.astype(np.float32)
            a = air_mask(img)
            img[~a] = np.nan                      # only air contributes
            stack.append(img)
        if not stack:
            raise SystemExit(f"no usable projections for view {view}")
        with np.errstate(all="ignore"):
            base[view] = np.nanmedian(np.stack(stack), 0)
        # a pixel no patient exposes stays NaN; fill it from the row median so the
        # polynomial stage has something finite to correct
        row = np.nanmedian(base[view], 1, keepdims=True)
        base[view] = np.where(np.isfinite(base[view]), base[view], row)
        print(f"  view {view+1}/{NA}: {len(stack)} patients", flush=True)
    np.save(os.path.join(args.out, "ff_base.npy"), base)

    # ---- stage 2: per-density, per-view polynomial correction ----
    coeffs = {}
    for dn, ss in by_density.items():
        coeffs[dn] = []
        for view in range(NA):
            num = np.zeros((NATIVE_H, NATIVE_W), np.float64)
            cnt = np.zeros((NATIVE_H, NATIVE_W), np.int32)
            for s in ss:
                fs = projection_files(args.root, s)
                if fs is None:
                    continue
                img = pydicom.dcmread(fs[view]).pixel_array.astype(np.float32)
                a = air_mask(img)
                num[a] += img[a]
                cnt[a] += 1
            seen = cnt > 0
            ratio = np.ones((NATIVE_H, NATIVE_W), np.float32)
            ratio[seen] = (num[seen] / cnt[seen]) / np.maximum(base[view][seen], 1e-6)
            _, c = fit_poly2d(seen, ratio, args.degree)
            coeffs[dn].append(c)
            print(f"  {dn} view {view+1}/{NA}: {seen.mean()*100:.1f}% of the detector "
                  f"seen as air", flush=True)
    with open(os.path.join(args.out, "coefficients.json"), "w") as f:
        json.dump({"degree": args.degree, "coefficients": coeffs}, f, indent=1)

    print(f"\nwrote {args.out}/ff_base.npy and {args.out}/coefficients.json")
    print("generate_dataset.py reads both; nothing else is needed.")


if __name__ == "__main__":
    main()

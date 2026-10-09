#!/usr/bin/env python3
"""
VICTRE-Paired — two-regime baselines.

Reconstructs every patient in a split with nine classical methods under two
projection regimes:

    ideal : g = A(clean)     (inverse crime -- same operator generates and solves)
    real  : g = clean_proj   (inverse-crime-free -- real Monte-Carlo projections)

and, optionally, a noisy regime (half-dose noisy_proj). Metrics: breast-masked
correlation (primary), scale-matched PSNR, SSIM, RMSE, and mass SDNR. Writes
per-patient rows to tables/baseline_raw.csv (and baseline_rows.npy).

Methods: ATp, FBP, SIRT-20/50/100, SART-2/4/8, ASD-POCS-20. FBP uses an
explicit ramp filter (geometry.py) rather than VICTRE's own reconstruction
code, and rather than LEAP's built-in FBP, which returns a degenerate result
on this modular-beam geometry.

Requires a CUDA GPU and the `leapctype` package (see README -- Installation).
This script only reconstructs and scores; run
figures/make_baseline_figures.py afterwards to produce Table 3 and the paper
figures (F5/F5b/F5c/F6) from its output -- that step needs no GPU.

Usage
-----
    python run_baselines.py --data /path/to/victre-paired --out ./paper --split test
    python run_baselines.py --data /path/to/victre-paired --out ./paper --limit 16   # smoke test
    python figures/make_baseline_figures.py --out ./paper

Resumable: writes to tables/baseline_raw.csv after every chunk; re-running
continues from the last completed (patient, regime, method) triple.
"""

import os, sys, glob, gc, time, csv, argparse

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import BROKEN_SEEDS, SSIM_SLICE_STEP
from noise import decode as decode_noisy
from geometry import Geometry, DEV

# =============================================================================
# CLI
# =============================================================================
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the dataset root")
ap.add_argument("--out", default="./paper", help="output directory")
ap.add_argument("--split", default="test", help="split to evaluate (paper numbers use test)")
ap.add_argument("--limit", type=int, default=0, help="process only the first N patients (0 = all)")
ap.add_argument("--noisy-regime", action="store_true",
                help="also reconstruct the half-dose noisy_proj regime")
ap.add_argument("--sirt-iters", type=int, nargs="+", default=[20, 50, 100])
ap.add_argument("--sart-iters", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--asdpocs-iters", type=int, default=20)
args = ap.parse_args()

DATA, OUTDIR, SPLIT = args.data, args.out, args.split
TAB, FIG = f"{OUTDIR}/tables", f"{OUTDIR}/figures"
for d in (OUTDIR, TAB, FIG):
    os.makedirs(d, exist_ok=True)

T0 = time.time()
def P(*a): print(*a, flush=True)
def elapsed(): return (time.time() - T0) / 60

assert DEV == "cuda", "A CUDA GPU is required (torch.cuda.is_available() is False)"
P(f"device={DEV} | {torch.cuda.get_device_name(0)}")

def free():
    gc.collect(); torch.cuda.empty_cache()

# =============================================================================
# Methods -- thin wrappers over geometry.Geometry, indexed by name
# =============================================================================
METHODS = {"ATp": lambda G, g: G.atp(g), "FBP": lambda G, g: G.fbp(g)}
for n in args.sirt_iters:  METHODS[f"SIRT-{n}"] = (lambda G, g, n=n: G.sirt(g, n))
for n in args.sart_iters:  METHODS[f"SART-{n}"] = (lambda G, g, n=n: G.sart(g, n))
METHODS[f"ASDPOCS-{args.asdpocs_iters}"] = (lambda G, g, n=args.asdpocs_iters: G.asdpocs(g, n))
MORDER = list(METHODS.keys())
P(f"methods ({len(METHODS)}): {MORDER}")

_GC = {}
def geo_for(vox_z, offx, offy, offz):
    """Small cache: only a handful of distinct geometries occur (class x native_z)."""
    k = tuple(round(float(x), 4) for x in (vox_z, offx, offy, offz))
    if k not in _GC:
        if len(_GC) > 8:
            free()
            _GC.pop(next(iter(_GC)))
        _GC[k] = Geometry(*k)
    return _GC[k]

# =============================================================================
# Metrics
# =============================================================================
def _mask(m): return np.asarray(m) > 0.5

def affine_match(pred, gt, mask):
    """Fit a*pred + b to gt by least squares INSIDE the mask, return the matched volume.

    The reconstruction methods return raw volumes (see geometry.py): different
    methods land on different radiometric scales, and FBP in particular shifts the
    whole background because its ramp filter removes the DC component. Comparing
    those directly with PSNR/SSIM/RMSE would score the scale, not the image.

    This is the single, documented normalisation in the pipeline. `corr` and the
    SDNR-style task metrics are already invariant to it and use the raw volume.
    """
    k = _mask(mask)
    if k.sum() < 100:
        return np.asarray(pred, np.float32), np.nan, np.nan
    x = np.asarray(pred, np.float64)[k]
    y = np.asarray(gt, np.float64)[k]
    A = np.stack([x, np.ones_like(x)], 1)
    try:
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return np.asarray(pred, np.float32), np.nan, np.nan
    return (a * np.asarray(pred, np.float64) + b).astype(np.float32), float(a), float(b)


def psnr(pred, gt, mask=None):
    mse = ((pred[_mask(mask)] - gt[_mask(mask)])**2).mean() if mask is not None \
          else ((pred - gt)**2).mean()
    return float(-10 * np.log10(mse + 1e-12))

def rmse(pred, gt, mask=None):
    k = _mask(mask) if mask is not None else slice(None)
    return float(np.sqrt(((pred[k] - gt[k])**2).mean())) if (mask is None or k.sum()) else np.nan


def ssim3d(pred, gt, mask, step=SSIM_SLICE_STEP):
    """2D SSIM averaged over MASKED voxels only, every `step`-th slice.

    The mask is not optional. Without it the average runs over the whole slice,
    where background and air dominate by count and agree trivially between any
    two reconstructions; that inflates the score and inflates it unevenly,
    because methods differ in what they put outside the breast. The same
    argument the manuscript uses against whole-volume metrics applies here.

    The average is weighted by masked voxel count, not by slice: a slice with
    forty breast voxels should not carry the same weight as one with forty
    thousand. This matches the definition used for the published numbers
    (H1_baselines_detectability), so a reuser's SSIM is comparable to Table 3.

    `step` subsamples slices for speed and is stated here rather than left
    implicit: at step=2, 28 of 56 slices are scored.

    One property to know before reusing the number. SSIM is computed through a
    Gaussian window (sigma 1.5), so a masked voxel within about four pixels of
    the mask boundary still sees background through that window: masking the
    AVERAGE does not fully isolate the breast. Measured here, with the mask
    fixed and only the background perturbed:

        background sigma   0.05   0.10   0.30   0.60
        shift in SSIM     0.0003 0.0014 0.0109 0.0311

    i.e. negligible while the background differs on the scale reconstructions
    actually differ, and noticeable only if one method leaves artefacts outside
    the breast as large as the signal inside it. Eroding the mask by five pixels
    removes the leak almost entirely (a 30x reduction) at the cost of 12 % of the
    mask, and we deliberately do NOT do that here: the published numbers
    (Table 3) are computed without erosion, and a reuser's SSIM has to be
    comparable to them. Report the leak, do not silently change the definition.
    """
    C1, C2 = 0.01**2, 0.03**2
    k = _mask(mask)
    num, den = 0.0, 0
    for z in range(0, pred.shape[0], step):
        m = k[z]
        if m.sum() < 50:          # too little breast in this slice to be meaningful
            continue
        x, y = pred[z].astype(np.float64), gt[z].astype(np.float64)
        mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
        vx = gaussian_filter(x*x, 1.5) - mx*mx
        vy = gaussian_filter(y*y, 1.5) - my*my
        vxy = gaussian_filter(x*y, 1.5) - mx*my
        sm = (((2*mx*my + C1)*(2*vxy + C2)) /
              ((mx*mx + my*my + C1)*(vx + vy + C2) + 1e-12))
        num += sm[m].sum()
        den += int(m.sum())
    return float(num / den) if den else np.nan

def mass_sdnr(vol, lesion_coords, ri=4, ro=12):
    masses = lesion_coords[lesion_coords[:, 3] >= 4] if len(lesion_coords) else []
    out = []
    for z, h, w, _ in masses:
        z, h, w = int(round(z)), int(round(h)), int(round(w))
        if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]):
            continue
        core = vol[max(0,z-1):z+2, max(0,h-ri):h+ri+1, max(0,w-ri):w+ri+1]
        ring = vol[z, max(0,h-ro):h+ro+1, max(0,w-ro):w+ro+1]
        if ring.std() > 1e-8:
            out.append((core.mean() - ring.mean()) / ring.std())
    return float(np.mean(out)) if out else np.nan

def corr(a, b, mask=None):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if mask is not None:
        k = _mask(mask)
        if k.sum() < 100: return np.nan
        a, b = a[k], b[k]
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# =============================================================================
# Main loop
# =============================================================================
def dump_rows(rows, path):
    if not rows: return
    keys = sorted({k for r in rows for k in r})
    tmp = path + f".t{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    os.replace(tmp, path)

RAW = f"{TAB}/baseline_raw.csv"
rows, done = [], set()
if os.path.exists(RAW):
    with open(RAW) as f:
        for r in csv.DictReader(f):
            rows.append(r)
            done.add((int(r["seed"]), r["regime"], r["method"]))
    P(f"[resume] {len(done)} (patient, regime, method) triples already present")

files = sorted(glob.glob(f"{DATA}/{SPLIT}/*.npz"))
P(f"{SPLIT}: {len(files)} chunks\n")

REGIMES = ["ideal", "real"] + (["noisy"] if args.noisy_regime else [])
n_done, t0 = 0, time.time()

def to01(a):
    a = np.asarray(a)
    if a.dtype == np.uint16 or a.max() > 1.01:
        a = a.astype(np.float32) / 65535.0
    return a.astype(np.float32)

for ci, fp in enumerate(files):
    try:
        d = np.load(fp)
    except Exception as e:
        P(f"  ! {os.path.basename(fp)}: {e}")
        continue
    for b in range(len(d["seed"])):
        seed = int(d["seed"][b])
        if seed in BROKEN_SEEDS:
            continue
        if args.limit and n_done >= args.limit:
            break
        gt = to01(d["clean"][b])
        mask = np.asarray(d["mask"][b])
        lesion_coords = d["lesion_coords"][b][:int(d["lesion_count"][b])]
        density = str(d["density"][b])
        is_pos = bool(d["is_pos"][b])
        base = dict(seed=seed, density=density, is_pos=int(is_pos),
                   native_z=int(d["native_z"][b]))

        G = geo_for(float(d["geom_vox_z"][b]), float(d["geom_offx"][b]),
                   float(d["geom_offy"][b]), float(d["geom_offz"][b]))

        sources = {"ideal": G.A(gt), "real": to01(d["clean_proj"][b])}
        if args.noisy_regime and "noisy_proj_half" in d.files:
            # noisy arrays use their own quantisation, not /65535 (see noise.py)
            sources["noisy"] = decode_noisy(d["noisy_proj_half"][b])

        for regime in REGIMES:
            if regime not in sources:
                continue
            g = sources[regime]
            for name, fn in METHODS.items():
                if (seed, regime, name) in done:
                    continue
                try:
                    r = fn(G, g)                      # raw reconstruction
                    ra, aff_a, aff_b = affine_match(r, gt, mask)
                    # corr and SDNR are scale-invariant -> raw volume.
                    # PSNR/SSIM/RMSE are not -> affine-matched volume.
                    rec = {**base, "regime": regime, "method": name,
                          "corr": corr(r, gt, mask),
                          "psnr_mask": psnr(ra, gt, mask),
                          "ssim": ssim3d(ra, gt, mask), "rmse": rmse(ra, gt, mask),
                          "aff_a": aff_a, "aff_b": aff_b,
                          "sdnr": mass_sdnr(r, lesion_coords) if is_pos else ""}
                    del r, ra
                    free()
                except Exception as e:
                    rec = {**base, "regime": regime, "method": name,
                          "corr": "", "psnr_mask": "", "ssim": "", "rmse": "",
                          "aff_a": "", "aff_b": "", "sdnr": "", "error": str(e)[:60]}
                rows.append(rec)

        if (seed, "GT", "GT") not in done:
            rows.append({**base, "regime": "GT", "method": "GT", "corr": 1.0,
                        "psnr_mask": "", "ssim": 1.0, "rmse": 0.0,
                        "aff_a": 1.0, "aff_b": 0.0,
                        "sdnr": mass_sdnr(gt, lesion_coords) if is_pos else ""})
            done.add((seed, "GT", "GT"))
        n_done += 1
    d.close()
    free()
    dump_rows(rows, RAW)
    e = (time.time() - t0) / 60
    P(f"  [{ci+1}/{len(files)}] {n_done} patients | {e:.1f} min | "
      f"~{e/max(n_done,1)*(len(files)-ci-1):.0f} min remaining")
    if args.limit and n_done >= args.limit:
        break

dump_rows(rows, RAW)
np.save(f"{OUTDIR}/baseline_rows.npy", np.array(rows, dtype=object), allow_pickle=True)

# ---------------------------------------------------------------------------
# Completeness check. A method that fails on some patients still writes a row,
# with the metric columns blank -- so an unequal count is the only signal that a
# column is quietly averaged over fewer patients than its neighbours. Comparing
# such columns would be comparing different samples.
# ---------------------------------------------------------------------------
from collections import Counter
filled = Counter((r["regime"], r["method"]) for r in rows
                 if r.get("regime") != "GT" and str(r.get("corr", "")) != "")
errored = [r for r in rows if r.get("error")]
if filled:
    counts = sorted(set(filled.values()))
    P("\ncompleteness (filled rows per regime x method):")
    for (rg, m), c in sorted(filled.items()):
        P(f"   {rg:>6s} {m:<14s} {c}")
    if errored:
        P(f"\n{len(errored)} row(s) recorded an error:")
        for r in errored[:10]:
            P(f"   seed {r['seed']} {r['regime']}/{r['method']}: {r['error']}")
    assert len(counts) == 1, (
        f"methods do not have the same number of filled rows: {counts}. "
        f"Columns averaged over different samples are not comparable -- see the "
        f"error rows above.")
    P(f"\nall {len(filled)} regime x method cells have {counts[0]} filled rows")

P(f"\ndone: {n_done} patients, {len(rows)} rows, {(time.time()-t0)/60:.1f} min")
P(f"\nNext: python figures/make_baseline_figures.py --out {OUTDIR}")

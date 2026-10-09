#!/usr/bin/env python3
"""
VICTRE-Paired — the reference evaluation.

Score your reconstructions the way the paper scores its baselines, so that
numbers from different papers are comparable. You supply reconstructions; this
computes the metrics, the matched-position detection statistics, and the
paired confidence intervals.

    python src/evaluate.py --data /path/to/victre-paired --pred /path/to/preds \\
                           --name my-method --out results/

`--pred` holds one `.npy` per patient, named `<seed>.npy`, shaped (56, 408, 336),
in any radiometric scale: the scale is fitted out inside the breast mask before
the scale-dependent metrics, exactly as for the published baselines.

What it reports, and why each one is here
-----------------------------------------
  corr          masked correlation with the reference. Scale-invariant, and the
                headline agreement metric of the paper.
  psnr, ssim,   computed AFTER an affine fit inside the mask. Without that fit
  rmse          they score the radiometric scale rather than the image -- FBP in
                particular shifts the whole background because its ramp filter
                removes the DC component.
  mass SDNR     signal-difference-to-noise at the lesion positions.
  control SDNR  the SAME estimator at matched signal-absent positions. It should
                be ~0; a positive value means the method is inventing structure
                where there is none, which is exactly what the iterative methods
                do on the real projections.
  AUC, d'       mass versus matched control. This is the only number that says
                whether a lesion became easier to see, and agreement metrics can
                move the other way -- a network can raise `corr` well above FBP
                while its AUC drops significantly.

Report agreement AND task together. The paper's own results contain three
counter-examples to ranking by a single metric.

Confidence intervals are bootstrapped over POSITIONS (lesions and controls),
not patients, and the comparison against a baseline is paired on the same
positions. With ~276 patients a Wilcoxon p-value saturates and stops separating
an effect of 0.0004 from one of 0.29, so effect size with an interval is what
this prints.
"""

import os, glob, json, csv, argparse
import numpy as np

from constants import BROKEN_SEEDS, SSIM_SLICE_STEP
from scipy.ndimage import gaussian_filter

RI, RO = 4, 12           # SDNR core and ring half-widths, in voxels


def mask_of(d, b):
    """The released breast mask. Use the stored array, do not re-threshold."""
    return np.asarray(d["mask"][b]) > 0.5


def affine_match(pred, gt, mask):
    # Coerce: the stored mask is uint8. Indexing with uint8 is INTEGER indexing,
    # which silently returns the wrong voxels instead of raising.
    k = np.asarray(mask) > 0.5
    if k.sum() < 100:
        return np.asarray(pred, np.float32)
    x = np.asarray(pred, np.float64)[k]; y = np.asarray(gt, np.float64)[k]
    A = np.stack([x, np.ones_like(x)], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return (c[0] * np.asarray(pred, np.float64) + c[1]).astype(np.float32)


def corr(a, b, m):
    m = np.asarray(m) > 0.5
    x = np.asarray(a, np.float64)[m]; y = np.asarray(b, np.float64)[m]
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def psnr(a, b, m):
    m = np.asarray(m) > 0.5
    return float(-10 * np.log10(((a[m] - b[m]) ** 2).mean() + 1e-12))


def rmse(a, b, m):
    m = np.asarray(m) > 0.5
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean()))


def ssim3d(pred, gt, mask, step=SSIM_SLICE_STEP):
    """2D SSIM averaged over MASKED voxels only, every `step`-th slice.

    THIS MUST MATCH run_baselines.py, which produced the published table. It
    previously did not: this function took no mask and stepped by 4, so a reuser
    following the protocol got a different, inflated SSIM than Table 2. The mask
    is not optional -- without it the average runs over the whole slice, where
    background and air dominate by count and agree trivially between any two
    reconstructions.

    The average is weighted by masked voxel count, not by slice. Both constants
    now come from constants.py so the two scripts cannot drift again.

    One property to know before reusing the number: SSIM is computed through a
    Gaussian window (sigma 1.5), so a masked voxel within about four pixels of
    the mask boundary still sees background through that window. Measured with
    the mask fixed and only the background perturbed, the shift in SSIM is
    0.0003 at background sigma 0.05 and 0.0311 at 0.60. The mask is deliberately
    NOT eroded, because the published numbers are computed without erosion.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = np.asarray(mask) > 0.5
    num, den = 0.0, 0
    for z in range(0, pred.shape[0], step):
        m = k[z]
        if m.sum() < 50:
            continue
        x, y = pred[z].astype(np.float64), gt[z].astype(np.float64)
        mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
        vx = gaussian_filter(x * x, 1.5) - mx * mx
        vy = gaussian_filter(y * y, 1.5) - my * my
        vxy = gaussian_filter(x * y, 1.5) - mx * my
        sm = (((2 * mx * my + C1) * (2 * vxy + C2)) /
              ((mx * mx + my * my + C1) * (vx + vy + C2) + 1e-12))
        num += sm[m].sum()
        den += int(m.sum())
    return float(num / den) if den else np.nan


def sdnr_at(vol, z, h, w):
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]):
        return None
    c = vol[max(0, z-1):z+2, max(0, h-RI):h+RI+1, max(0, w-RI):w+RI+1]
    r = vol[z, max(0, h-RO):h+RO+1, max(0, w-RO):w+RO+1]
    s = r.std()
    return None if s < 1e-8 else float((c.mean() - r.mean()) / s)


def positions(d, b):
    """Lesion (masses only, type >= 4) and matched control positions."""
    out = []
    n = int(d["lesion_count"][b])
    for z, h, w, t in d["lesion_coords"][b][:n]:
        if t >= 4:
            out.append(("mass", z, h, w))
    n = int(d["control_count"][b])
    for z, h, w, _rid, ok in d["control_rois"][b][:n]:
        if ok > 0.5:
            out.append(("control", z, h, w))
    return out


def auc_dprime(mass, ctrl):
    """Mann-Whitney AUC and d'."""
    m, c = np.asarray(mass, float).ravel(), np.asarray(ctrl, float).ravel()
    # len(), not truthiness: boot_auc passes arrays, and `not <array>` raises.
    if len(m) == 0 or len(c) == 0:
        return float("nan"), float("nan")
    order = np.argsort(np.concatenate([m, c]))
    ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order) + 1)
    auc = float((ranks[:len(m)].sum() - len(m) * (len(m) + 1) / 2) / (len(m) * len(c)))
    sp = np.sqrt(0.5 * (m.var() + c.var()))
    return auc, float((m.mean() - c.mean()) / sp) if sp > 1e-12 else float("nan")


def boot_auc(mass, ctrl, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m, c = np.asarray(mass), np.asarray(ctrl)
    vals = [auc_dprime(rng.choice(m, len(m), True), rng.choice(c, len(c), True))[0]
            for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root with train/val/test")
    ap.add_argument("--pred", required=True, help="directory of <seed>.npy reconstructions")
    ap.add_argument("--name", default="method")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="evaluation")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    have = {int(os.path.basename(p)[:-4]): p
            for p in glob.glob(os.path.join(args.pred, "*.npy"))
            if os.path.basename(p)[:-4].isdigit()}
    print(f"{len(have)} prediction files in {args.pred}")

    rows, mass_s, ctrl_s, n_skip = [], [], [], 0
    for fp in sorted(glob.glob(f"{args.data}/{args.split}/*.npz")):
        with np.load(fp) as d:
            for b, s in enumerate(d["seed"]):
                s = int(s)
                if s in BROKEN_SEEDS or s not in have:
                    n_skip += s in BROKEN_SEEDS
                    continue
                gt = d["clean"][b].astype(np.float32) / 65535
                m = mask_of(d, b)
                pred = np.load(have[s]).astype(np.float32)
                if pred.shape != gt.shape:
                    raise SystemExit(f"{have[s]}: shape {pred.shape}, expected {gt.shape}")
                pa = affine_match(pred, gt, m)
                rec = dict(seed=s, density=str(d["density"][b]),
                           corr=corr(pred, gt, m),           # scale-invariant: raw
                           psnr=psnr(pa, gt, m), rmse=rmse(pa, gt, m),
                           ssim=ssim3d(pa, gt, m))
                for kind, z, h, w in positions(d, b):
                    v = sdnr_at(pred, z, h, w)
                    if v is None:
                        continue
                    (mass_s if kind == "mass" else ctrl_s).append(v)
                rows.append(rec)

    if not rows:
        raise SystemExit("no patient matched a prediction file -- check --pred and --split")

    auc, dp = auc_dprime(mass_s, ctrl_s)
    lo, hi = boot_auc(mass_s, ctrl_s, args.boot) if mass_s and ctrl_s else (float("nan"),) * 2
    summary = dict(
        name=args.name, split=args.split, n_patients=len(rows),
        n_mass=len(mass_s), n_control=len(ctrl_s),
        corr=float(np.mean([r["corr"] for r in rows])),
        corr_sd=float(np.std([r["corr"] for r in rows])),
        psnr=float(np.mean([r["psnr"] for r in rows])),
        ssim=float(np.mean([r["ssim"] for r in rows])),
        rmse=float(np.mean([r["rmse"] for r in rows])),
        sdnr_mass=float(np.mean(mass_s)) if mass_s else float("nan"),
        sdnr_control=float(np.mean(ctrl_s)) if ctrl_s else float("nan"),
        auc=auc, auc_lo=lo, auc_hi=hi, dprime=dp)

    with open(os.path.join(args.out, f"{args.name}_per_patient.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(args.out, f"{args.name}_summary.json"), "w"), indent=1)

    print(f"\n{args.name} on {args.split}: {len(rows)} patients"
          + (f" ({n_skip} degenerate patient excluded)" if n_skip else ""))
    print(f"  masked correlation  {summary['corr']:.4f} +/- {summary['corr_sd']:.4f}")
    print(f"  PSNR / SSIM / RMSE  {summary['psnr']:.2f} / {summary['ssim']:.4f} / "
          f"{summary['rmse']:.5f}   (after the affine fit)")
    print(f"  mass SDNR           {summary['sdnr_mass']:+.4f}   (n={len(mass_s)})")
    print(f"  control SDNR        {summary['sdnr_control']:+.4f}   (n={len(ctrl_s)}; "
          f"should be ~0)")
    print(f"  AUC                 {auc:.4f} [{lo:.4f}, {hi:.4f}]    d' {dp:.3f}")
    print(f"\nwrote {args.out}/{args.name}_summary.json")
    print("Report the agreement and the task metric together; the paper has three "
          "cases where they disagree.")


if __name__ == "__main__":
    main()

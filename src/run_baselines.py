"""
run_baselines.py — classical baseline reconstructions under two regimes.

For every test patient (excluding the known broken SEED), eight classical
methods are run under both projection regimes:
  * ideal  — synthetic forward projections A(clean)   (inverse crime)
  * real   — the real Monte-Carlo projections noisy_proj (inverse-crime-free)

Methods: Aᵀp; SIRT-20/50/100; SART-2/4/8; ASD-POCS-20. For each reconstruction
the script records correlation, PSNR (raw masked and scale-matched), SSIM, RMSE
and mass SDNR. A "GT" row per patient records the reference-volume mass SDNR.

Output: rows_test.npy (list of per-row dicts), checkpointed every few chunks,
consumed by figures/make_baseline_figures.py.

Requires a CUDA GPU and LEAP (see geometry.py).
"""
import os
import gc
import glob
import time
import shutil

import numpy as np
import torch

from geometry import Projector
from constants import BROKEN_SEEDS, fmt_seconds as fmt

V6 = "/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v6"
OUT_DIR = "/content/baseline_results"
os.makedirs(OUT_DIR, exist_ok=True)
SPLIT = "test"
CKPT_EVERY = 5


# ── metrics ────────────────────────────────────────────────────────────────
def psnr_mask(r, gt, m):
    mm = m > 0.5
    if mm.sum() < 10:
        return np.nan
    return float(-10 * np.log10(((r[mm] - gt[mm]) ** 2).mean() + 1e-12))


def psnr_scaled(r, gt, m):
    """PSNR after fitting the optimal affine map (a*r + b) to gt over the mask,
    isolating structural fidelity from the intensity scale (scale-invariant)."""
    mm = m > 0.5
    if mm.sum() < 10:
        return np.nan
    x = r[mm].astype(np.float64); y = gt[mm].astype(np.float64)
    xm, ym = x.mean(), y.mean(); vx = ((x - xm) ** 2).mean()
    if vx < 1e-12:
        return np.nan
    a = ((x - xm) * (y - ym)).mean() / vx; b = ym - a * xm
    return float(-10 * np.log10(((a * x + b - y) ** 2).mean() + 1e-12))


def rmse_mask(r, gt, m):
    mm = m > 0.5
    return float(np.sqrt(((r[mm] - gt[mm]) ** 2).mean())) if mm.sum() else np.nan


def corr(r, gt):
    a = r.ravel().astype(np.float64); b = gt.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ssim_global(r, gt, C1=0.01 ** 2, C2=0.03 ** 2):
    """Global SSIM over the volume (data range 1). A single-window approximation;
    sufficient for ranking baselines alongside PSNR and correlation."""
    a = r.astype(np.float64); b = gt.astype(np.float64)
    mua, mub = a.mean(), b.mean(); va, vb = a.var(), b.var()
    cov = ((a - mua) * (b - mub)).mean()
    return float(((2 * mua * mub + C1) * (2 * cov + C2)) /
                 ((mua ** 2 + mub ** 2 + C1) * (va + vb + C2)))


def sdnr(vol, z, h, w, ri=4, ro=12):
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]):
        return np.nan
    c = vol[max(0, z - 1):z + 2, max(0, h - ri):h + ri + 1, max(0, w - ri):w + ri + 1]
    ring = vol[z, max(0, h - ro):h + ro + 1, max(0, w - ro):w + ro + 1]
    return (c.mean() - ring.mean()) / ring.std() if ring.std() > 1e-8 else np.nan


def mass_sdnr(vol, lc, nl):
    """Mean SDNR over a patient's mass lesions (type >= 4); nan if none."""
    m = lc[:nl]; m = m[m[:, 3] >= 4]
    if len(m) == 0:
        return np.nan
    vals = [sdnr(vol, z, h, w) for z, h, w, _ in m]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


# ── baselines (operate on normalized projections g) ────────────────────────
def n01(v):
    v = v - v.min()
    return v / (v.max() + 1e-8)


def bl_atp(P, g):
    return n01(P.AT(g))


def bl_sirt(P, g, it):
    f = torch.zeros(P.ZN, P.NY, P.NX, device=P.device)
    P.L.SIRT((g * P.PROJ_SCALE).contiguous(), f, it)
    return n01(f.permute(0, 2, 1).contiguous())


def bl_sart(P, g, it):
    f = torch.zeros(P.ZN, P.NY, P.NX, device=P.device)
    P.L.SART((g * P.PROJ_SCALE).contiguous(), f, it)
    return n01(f.permute(0, 2, 1).contiguous())


def bl_asdpocs(P, g, it, n_tv=20, n_subsets=1):
    f = torch.zeros(P.ZN, P.NY, P.NX, device=P.device)
    P.L.ASDPOCS((g * P.PROJ_SCALE).contiguous(), f, it, n_subsets, n_tv)
    return n01(f.permute(0, 2, 1).contiguous())


BASELINES = {
    "ATp":       lambda P, g: bl_atp(P, g),
    "SIRT-20":   lambda P, g: bl_sirt(P, g, 20),
    "SIRT-50":   lambda P, g: bl_sirt(P, g, 50),
    "SIRT-100":  lambda P, g: bl_sirt(P, g, 100),
    "SART-2":    lambda P, g: bl_sart(P, g, 2),
    "SART-4":    lambda P, g: bl_sart(P, g, 4),
    "SART-8":    lambda P, g: bl_sart(P, g, 8),
    "ASDPOCS-20": lambda P, g: bl_asdpocs(P, g, 20),
}


def gl(pat, tries=5):
    for i in range(tries):
        r = sorted(glob.glob(pat))
        if r:
            return r
        time.sleep(3)
    return []


def checkpoint(rows):
    lp = os.path.join(OUT_DIR, "rows_test.npy")
    np.save(lp, np.array(rows, dtype=object), allow_pickle=True)


def main():
    P = Projector()
    assert P.device == "cuda", "GPU required"
    print(f"PROJ_SCALE={P.PROJ_SCALE:.3f} | adjoint ratio={P.adjoint_ratio():.4f}", flush=True)

    files = gl(f"{V6}/{SPLIT}/*.npz")
    print(f"{SPLIT}: {len(files)} chunks | two regimes x {len(BASELINES)} baselines\n", flush=True)

    rows = []; t0 = time.time(); npat = 0
    for ci, fp in enumerate(files):
        d = None
        for _ in range(3):
            try:
                d = np.load(fp); break
            except Exception:
                time.sleep(2)
        if d is None:
            print(f"  skip {os.path.basename(fp)}"); continue

        for b in range(len(d["seed"])):
            if int(d["seed"][b]) in BROKEN_SEEDS:
                continue
            gt = d["clean"][b]; mk = d["mask"][b]
            lc = d["lesion_coords"][b]; nl = int(d["lesion_count"][b])
            base = dict(seed=int(d["seed"][b]), dens=str(d["density"][b]),
                        ip=bool(d["is_pos"][b]), nz=int(d["native_z"][b]))

            # reference (GT) mass SDNR
            rows.append({**base, "regime": "—", "method": "GT",
                         "psnr": np.nan, "psnr_mask": np.nan, "psnr_scaled": np.nan,
                         "ssim": np.nan, "rmse": np.nan, "corr": 1.0,
                         "sdnr": mass_sdnr(gt, lc, nl)})

            with torch.no_grad():
                g_ideal = P.A(torch.from_numpy(gt).float().to(P.device))
            g_real = torch.from_numpy(d["noisy_proj"][b]).float().to(P.device)

            for regime, g in [("ideal", g_ideal), ("real", g_real)]:
                for name, fn in BASELINES.items():
                    try:
                        r = fn(P, g).cpu().numpy()
                        rows.append({**base, "regime": regime, "method": name,
                                     "psnr": psnr_mask(r, gt, np.ones_like(mk)),
                                     "psnr_mask": psnr_mask(r, gt, mk),
                                     "psnr_scaled": psnr_scaled(r, gt, mk),
                                     "ssim": ssim_global(r, gt),
                                     "rmse": rmse_mask(r, gt, mk),
                                     "corr": corr(r, gt),
                                     "sdnr": mass_sdnr(r, lc, nl)})
                        del r
                    except Exception:
                        rows.append({**base, "regime": regime, "method": name,
                                     "psnr": np.nan, "psnr_mask": np.nan, "psnr_scaled": np.nan,
                                     "ssim": np.nan, "rmse": np.nan, "corr": np.nan, "sdnr": np.nan})
            del g_ideal, g_real
            npat += 1
        gc.collect(); torch.cuda.empty_cache()

        if (ci + 1) % CKPT_EVERY == 0:
            checkpoint(rows)
        el = time.time() - t0
        print(f"  [{ci + 1}/{len(files)}] {npat} patients | {el / 60:.1f}m | "
              f"ETA ~{el / (ci + 1) * (len(files) - ci - 1) / 60:.1f}m", flush=True)

    checkpoint(rows)
    print(f"\nDONE: {npat} patients, {len(rows)} rows, {fmt(time.time() - t0)}")
    print(f"saved {os.path.join(OUT_DIR, 'rows_test.npy')}")


if __name__ == "__main__":
    main()

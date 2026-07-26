"""
run_baselines.py — classical baseline reconstructions under two regimes.

For every test patient (excluding the known broken SEED), nine classical
methods are run under both projection regimes:
  * ideal  — synthetic forward projections A(clean)   (inverse crime)
  * real   — the real Monte-Carlo projections noisy_proj (inverse-crime-free)

Methods: FBP; Aᵀp; SIRT-20/50/100; SART-2/4/8; ASD-POCS-20.

FBP is implemented from scratch (Ram-Lak/Hann ramp filter along the tube-motion
axis, edge-replicate padding, approximate cosine weighting) rather than adapted
from VICTRE's own reconstruction code, which is GPL3-licensed and is neither
read nor copied here (see project documentation on licensing). Two built-in
LEAP routes were tried first and rejected: `L.FBP(...)` returns a degenerate,
all-zero-correlation result on this modular-beam geometry, and
`L.filterProjections(...)` is a silent no-op (its output is numerically
identical to unfiltered back-projection). Only the from-scratch filter below
produced a genuine filtered back-projection.

For each reconstruction the script records:
  * corr        — correlation over the full volume
  * corr_mask   — correlation over the breast mask only (PRIMARY metric; see note)
  * psnr_mask   — raw PSNR over the breast mask
  * psnr_scaled — PSNR after an optimal affine fit to GT (scale-invariant)
  * ssim        — global SSIM approximation
  * bg          — mean intensity outside the breast mask (ringing/background leakage)
  * sdnr        — mean SDNR over the patient's mass lesions
A "GT" row per patient records the reference volume's own bg and mass SDNR.

Metric note: whole-volume correlation is confounded by background behaviour.
FBP's ramp filter leaks energy into the zero-padded regions of the projections
(dead detector frame, penumbra strip) and so inflates background intensity,
while Aᵀp drives the background to near zero "for free" — background voxels
dominate the volume by count, so whole-volume correlation rewards a clean
background over faithful breast reconstruction. `corr_mask`, computed only
inside the breast, is therefore the primary cross-method metric; `corr` and
`bg` are retained as diagnostics.

Output: rows_test_final.npy (list of per-row dicts), checkpointed every few
chunks, consumed by figures/make_baseline_figures.py.

Requires a CUDA GPU and LEAP (see geometry.py).
"""
import os
import gc
import glob
import time

import numpy as np
import torch
import torch.fft

from geometry import Projector
from constants import BROKEN_SEEDS, DET_PIX, SDD, fmt_seconds as fmt

V6 = "/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v6"
OUT_DIR = "/content/baseline_results"
os.makedirs(OUT_DIR, exist_ok=True)
SPLIT = "test"
CKPT_EVERY = 5


# ── FBP ramp filter ──────────────────────────────────────────────────────
def ramp_filter(g, det_pix=DET_PIX, sdd=SDD, window="hann", cosine_weight=True):
    """Ramp filter applied along the tube-motion axis (axis=1: the detector
    rows, which move with the source in this modular-beam geometry — see
    geometry.py; the alignment offsets in constants.py are confirmed to vary
    along this same axis, an independent check on the convention).

    Edge-replicate padding to 2x length avoids the wraparound artefact of
    zero-padding under the implicit circular convolution of the FFT-based
    filter. A Hann-windowed ramp and an approximate cosine (ray-obliquity)
    weight are used; a DC-removal step was tried and rejected — it improved
    whole-volume correlation but worsened the breast-masked correlation and
    PSNR, i.e. it traded a diagnostic artefact for real signal.
    """
    out = g
    if cosine_weight:
        nr, nc = g.shape[1], g.shape[2]
        u = (torch.arange(nr, device=g.device, dtype=torch.float32) - (nr - 1) / 2) * det_pix
        v = (torch.arange(nc, device=g.device, dtype=torch.float32) - (nc - 1) / 2) * det_pix
        w = sdd / torch.sqrt(sdd ** 2 + u[:, None] ** 2 + v[None, :] ** 2)
        out = out * w[None, :, :]

    n = out.shape[1]
    npad = 1
    while npad < 2 * n:
        npad *= 2
    extra = npad - n
    r = extra // 2
    l = extra - r
    gp = torch.cat([out,
                    out[:, -1:, :].expand(-1, r, -1),
                    out[:, :1, :].expand(-1, l, -1)], dim=1)

    G = torch.fft.rfft(gp, dim=1)
    f = torch.fft.rfftfreq(gp.shape[1], device=g.device)
    H = 2.0 * f
    fn = f / f.max()
    if window == "hann":
        H = H * (0.5 + 0.5 * torch.cos(np.pi * fn))
    elif window == "hamming":
        H = H * (0.54 + 0.46 * torch.cos(np.pi * fn))
    # window == "ramlak": no apodisation

    filt = torch.fft.irfft(G * H.view(1, -1, 1), n=gp.shape[1], dim=1)[:, :n, :]
    return filt.contiguous()


# ── metrics ────────────────────────────────────────────────────────────────
def corr(r, gt):
    a = r.ravel().astype(np.float64); b = gt.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def corr_mask(r, gt, m):
    """PRIMARY metric: correlation restricted to the breast mask. See the
    module docstring for why whole-volume correlation is not used alone."""
    mm = m > 0.5
    if mm.sum() < 10:
        return np.nan
    a = r[mm].astype(np.float64); b = gt[mm].astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


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


def ssim_global(r, gt, C1=0.01 ** 2, C2=0.03 ** 2):
    """Global SSIM over the volume (data range 1). A single-window approximation;
    sufficient for ranking baselines alongside PSNR and correlation."""
    a = r.astype(np.float64); b = gt.astype(np.float64)
    mua, mub = a.mean(), b.mean(); va, vb = a.var(), b.var()
    cov = ((a - mua) * (b - mub)).mean()
    return float(((2 * mua * mub + C1) * (2 * cov + C2)) /
                 ((mua ** 2 + mub ** 2 + C1) * (va + vb + C2)))


def bg_energy(r, m):
    """Mean intensity OUTSIDE the breast mask. GT is close to zero; a large
    value indicates the reconstructor is leaking energy into the background
    (e.g. FBP ringing from the ramp filter hitting zero-padded projection
    regions). Diagnostic for the corr vs corr_mask discrepancy above."""
    mm = m <= 0.5
    return float(r[mm].mean()) if mm.sum() else np.nan


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


def bl_fbp(P, g):
    return n01(P.AT(ramp_filter(g)))


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
    "FBP":        lambda P, g: bl_fbp(P, g),
    "ATp":        lambda P, g: bl_atp(P, g),
    "SIRT-20":    lambda P, g: bl_sirt(P, g, 20),
    "SIRT-50":    lambda P, g: bl_sirt(P, g, 50),
    "SIRT-100":   lambda P, g: bl_sirt(P, g, 100),
    "SART-2":     lambda P, g: bl_sart(P, g, 2),
    "SART-4":     lambda P, g: bl_sart(P, g, 4),
    "SART-8":     lambda P, g: bl_sart(P, g, 8),
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
    lp = os.path.join(OUT_DIR, "rows_test_final.npy")
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

            # reference (GT) row: background energy and mass SDNR
            rows.append({**base, "regime": "—", "method": "GT",
                         "corr": 1.0, "corr_mask": 1.0,
                         "psnr_mask": np.nan, "psnr_scaled": np.nan, "ssim": np.nan,
                         "bg": bg_energy(gt, mk), "sdnr": mass_sdnr(gt, lc, nl)})

            with torch.no_grad():
                g_ideal = P.A(torch.from_numpy(gt).float().to(P.device))
            g_real = torch.from_numpy(d["noisy_proj"][b]).float().to(P.device)

            for regime, g in [("ideal", g_ideal), ("real", g_real)]:
                for name, fn in BASELINES.items():
                    try:
                        r = fn(P, g).cpu().numpy()
                        rows.append({**base, "regime": regime, "method": name,
                                     "corr": corr(r, gt), "corr_mask": corr_mask(r, gt, mk),
                                     "psnr_mask": psnr_mask(r, gt, mk),
                                     "psnr_scaled": psnr_scaled(r, gt, mk),
                                     "ssim": ssim_global(r, gt),
                                     "bg": bg_energy(r, mk),
                                     "sdnr": mass_sdnr(r, lc, nl)})
                        del r
                    except Exception:
                        rows.append({**base, "regime": regime, "method": name,
                                     "corr": np.nan, "corr_mask": np.nan,
                                     "psnr_mask": np.nan, "psnr_scaled": np.nan,
                                     "ssim": np.nan, "bg": np.nan, "sdnr": np.nan})
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
    print(f"saved {os.path.join(OUT_DIR, 'rows_test_final.npy')}")


if __name__ == "__main__":
    main()
